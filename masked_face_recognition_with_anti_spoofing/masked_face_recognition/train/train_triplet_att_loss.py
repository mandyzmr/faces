import time
from collections import defaultdict
from torch.nn.modules.distance import PairwiseDistance
from config import *
from visualization import *
from metrics import *
from train.criterion import *
from tqdm import tqdm

# 先用模型预测emb，然后通过计算dist找到会产生triplet loss的样本，再用mask计算attention loss
def train_on_epoch(model, train_loader, optimizer, device, steps=3, iter_smooth=10):
    epoch_triplet_loss, epoch_att_loss = 0, 0
    n_samples = 0
    step = 0 #由于筛选困难样本的原因，不是每个batch都能算loss，需要额外累计可以计算loss的batch数
    skip = 0 #计算skip掉的batch数
    n_batches = len(train_loader)
    # start = time.time()
    for i, sample in enumerate(train_loader):
        model.train() #训练模式
        anc_img = sample['anc_img'].to(device) #(b,c,h,w)
        pos_img = sample['pos_img'].to(device)
        neg_img = sample['neg_img'].to(device)
        anc_mask = sample['anc_mask'].to(device) #(b,h,w), 不知道为什么自动从array转为tensor
        pos_mask = sample['pos_mask'].to(device)
        neg_mask = sample['neg_mask'].to(device)
        # print(f'Loading time: {time.time()-start}')
        
        # start = time.time()
        # 计算emb, triplet和attention loss
        anc_emb, anc_att_loss, _ = model(anc_img, anc_mask) #(b,128), (b,)
        pos_emb, pos_att_loss, _ = model(pos_img, pos_mask)
        neg_emb, neg_att_loss, _ = model(neg_img, neg_mask)
        anc_emb *= 30 # 伸缩性系数 
        pos_emb *= 30 
        neg_emb *= 30 
        
        # 寻找困难样本OHNM：emb距离差比margin小的样本，即会产生triplet loss的样本
        pos_dist = PairwiseDistance(2).forward(anc_emb, pos_emb) #(b,)
        neg_dist = PairwiseDistance(2).forward(anc_emb, neg_emb)
        hard_idx = torch.where((neg_dist - pos_dist)<DIST_MARGIN) #得到对应样本的idx
        n_hard = len(hard_idx[0])
        
        # 如果没有困难样本，寻找下一个batch
        if n_hard == 0:
            skip+=1 #当跳过的数量太多，改用统计
            # print(f'Skipping Batch {i+1}/{n_batches} due to no hard samples.')
            continue
        
        # 筛选困难样本
        anc_emb = anc_emb[hard_idx]
        pos_emb = pos_emb[hard_idx]
        neg_emb = neg_emb[hard_idx]
        anc_att_loss = anc_att_loss[hard_idx] #(b,)
        pos_att_loss = pos_att_loss[hard_idx]
        neg_att_loss = neg_att_loss[hard_idx]

        # 损失
        triplet_loss = triplet_loss_criterion(anc_emb, pos_emb, neg_emb) #(b,128)->(1,)
        att_loss = torch.mean(torch.cat([anc_att_loss, pos_att_loss, neg_att_loss])) #(3b,)->(1,)
        loss = triplet_loss + att_loss
        loss = loss/steps #使得累积梯度仍为样本loss均值求得
        loss.backward()
        
        # 该batch所有hard样本的loss总和
        n_samples += n_hard
        epoch_triplet_loss += triplet_loss.item()*n_hard 
        epoch_att_loss += att_loss.item()*n_hard
        
        step+=1
        if step%steps==0: #steps个batch才优化一次，相当于batch_size *= steps
            optimizer.step()
            optimizer.zero_grad()
            step=0 #重新归零
            # 当跳过的数量太多时，只在backprob的时候输出
            print(f'Batch {i+1}/{n_batches} - hard samples: {n_samples} (skipped {skip} batches) - triplet loss: {epoch_triplet_loss/n_samples:.4f} - att loss: {epoch_att_loss/n_samples:.4f}')
       
        # print(f'Processing time: {time.time()-start}')
        # if i%iter_smooth==0:
        #     print(f'Batch {i+1}/{n_batches} - hard samples: {n_hard} - triplet loss: {epoch_triplet_loss/n_samples:.4f} - att loss: {epoch_att_loss/n_samples:.4f}')
        # start = time.time()
        
    epoch_triplet_loss = 0 if n_samples==0 else epoch_triplet_loss/n_samples #平均样本loss
    epoch_att_loss = 0 if n_samples==0 else epoch_att_loss/n_samples
    return epoch_triplet_loss, epoch_att_loss, n_samples



def val_on_epoch(model, valid_loader, device, epoch, roc_dir='', heatmap_dir='', masked=False):
    distances, labels = [], []
    val_loss = 0
    n_samples = 0
    for i, (img1, img2, label) in enumerate(tqdm(valid_loader)):
        model.eval() #预测模式，关闭dropout和bn
        img1 = img1.to(device) #(b,c,h,w)
        img2 = img2.to(device)
        label = label.to(device)
        n_samples += img1.shape[0]
        with torch.no_grad():
            img1_emb, _, img1_heatmap = model(img1) #(b,128), (b,c,h,w)
            img2_emb, _, img2_heatmap = model(img2)
            dist = PairwiseDistance(2).forward(img1_emb, img2_emb) #(b,)
            label = label.view(-1) #(b,)
            #记录下来一次性计算acc，转到cpu上才不会爆显存
            distances.append(dist.detach().cpu().numpy()) 
            labels.append(label.detach().cpu().numpy())
    
    # Attention heatmap：对最后一个batch的第一张照片进行可视化
    plot_heatmap(img1[0], img2[0], img1_heatmap[0], img2_heatmap[0], dist[0], label[0], epoch, heatmap_dir)
    
    # 不同验证集：戴口罩和不戴口罩
    distances = np.concatenate(distances) #(m,)
    labels = np.concatenate(labels)
    metrics = evaluate_metrics(distances, labels, epoch, n_folds=N_FOLDS, plot=True, roc_dir=roc_dir,
                               valid_set=f"LFW_{'Masked' if masked else 'Unmasked'}") 
    auc = metrics['auc'] #10折平均auc
    acc = metrics['acc'] #10折最高acc, 及其对应的precision, recall
    recall = metrics['recall']
    tar = metrics['tar'] #tar@far=target
    thres = metrics['thres'] #10折最高acc下的阈值
    print(f'Validation - dist_thres: {np.mean(thres):.2f}(+-{np.std(thres):.2f}) - auc: {auc:.2f} - acc: {np.mean(acc):.2f}(+-{np.std(acc):.2f}) '\
          f'- recall: {np.mean(recall):.2f}(+-{np.std(recall):.2f}) - tar@far=1e-2: {np.mean(tar):.2f}(+-{np.std(tar):.2f})') 
    return auc, np.mean(acc), np.mean(tar)



def predict(model, image, device): #根据face_align预测emb
    model.eval() #[0,255] (h,w,c)
    image = test_transform(image)[np.newaxis,...] #[0,1] (1,c,h,w)
    image = image.to(device)
    with torch.no_grad():
        emb, _, _  = model(image) #(1,128)
        emb = emb.detach().cpu().squeeze(dim=0).numpy() #(128,)
    return emb

