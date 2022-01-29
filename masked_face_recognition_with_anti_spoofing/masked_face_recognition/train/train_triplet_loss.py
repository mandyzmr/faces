import time
from collections import defaultdict
from torch.nn.modules.distance import PairwiseDistance
from config import *
from visualization import *
from metrics import *
from criterion import *

# 先用模型预测emb，然后通过计算dist找到会产生triplet loss的样本
def train_on_epoch(model, train_loader, optimizer, device, iter_smooth = 10):
    epoch_triplet_loss = 0
    n_samples = 0
    n_batches = len(train_loader)
    # start = time.time()
    for i, sample in enumerate(train_loader):
        model.train() #训练模式
        anc_img = sample['anc_img'].to(device) #(b,c,h,w)
        pos_img = sample['pos_img'].to(device)
        neg_img = sample['neg_img'].to(device)
        # print(f'Loading time: {time.time()-start}')
        
        # start = time.time()
        # 计算emb, triplet和attention loss
        anc_emb = model(anc_img) #(b,128), (b,)
        pos_emb = model(pos_img)
        neg_emb = model(neg_img)
        anc_emb *= 10 # 伸缩性系数 https://arxiv.org/pdf/1703.09507.pdf
        pos_emb *= 10 
        neg_emb *= 10 
        
        # 寻找困难样本OHNM：emb距离差比margin小的样本，即会产生triplet loss的样本
        pos_dist = PairwiseDistance(2).forward(anc_emb, pos_emb) #(b,)
        neg_dist = PairwiseDistance(2).forward(anc_emb, neg_emb)
        hard_idx = torch.where((neg_dist - pos_dist)<0.2) #得到对应样本的idx
        n_hard = len(hard_idx[0])
        
        # 如果没有困难样本，寻找下一个batch
        if n_hard == 0:
            print(f'Skipping Batch {i+1}/{n_batches} due to no hard samples.')
            continue
        
        # 筛选困难样本
        anc_emb = anc_emb[hard_idx]
        pos_emb = pos_emb[hard_idx]
        neg_emb = neg_emb[hard_idx]

        # 损失
        triplet_loss = triplet_loss_criterion(anc_emb, pos_emb, neg_emb) #(b,128)->(1,)
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()
        
        # 该batch所有hard样本的loss总和
        n_samples += n_hard
        epoch_triplet_loss += triplet_loss.item()*n_hard 
        
        # print(f'Processing time: {time.time()-start}')
        if i%iter_smooth==0:
            print(f'Batch {i+1}/{n_batches} - hard samples: {n_hard} - triplet loss: {epoch_triplet_loss/n_samples:.3f}') 
        # start = time.time()
        
    epoch_triplet_loss = 0 if n_samples==0 else epoch_triplet_loss/n_samples #平均样本loss
    return epoch_triplet_loss, n_samples



def val_on_epoch(model, valid_loader, device, epoch, roc_dir='', masked=False):
    distances, labels = [], []
    val_loss = 0
    n_samples = 0
    for i, (img1, img2, label) in enumerate(valid_loader):
        model.eval() #预测模式，关闭dropout和bn
        img1 = img1.to(device) #(b,c,h,w)
        img2 = img2.to(device)
        label = label.to(device)
        n_samples += img1.shape[0]
        with torch.no_grad():
            img1_emb = model(img1) #(b,128)
            img2_emb = model(img2)
            dist = PairwiseDistance(2).forward(img1_emb, img2_emb) #(b,)
            label = label.view(-1) #(b,)
            #记录下来一次性计算acc
            distances.append(dist.detach().cpu().numpy()) 
            labels.append(label.detach().cpu().numpy())
    
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
          f'- recall: {np.mean(recall):.2f}(+-{np.std(recall):.2f}) - tar@far=1e-3: {np.mean(tar):.2f}(+-{np.std(tar):.2f})') 
    return auc, np.mean(acc), np.mean(recall)



def predict(model, image, device): #根据face_align预测emb
    model.eval() #[0,255] (h,w,c)
    image = test_transform(image)[np.newaxis,...] #[0,1] (1,c,h,w)
    image = image.to(device)
    with torch.no_grad():
        emb = model(image) #(1,128)
        emb = emb.detach().cpu().squeeze(dim=0).numpy() #(128,)
    return emb


def predict_in_bulk(model, image, device): #批量预测
    model.eval() #[0,255] (h,w,c)
    image = test_transform(image)[np.newaxis,...] #[0,1] (1,c,h,w)
    image = image.to(device)
    with torch.no_grad():
        emb = model(image) #(1,128)
        emb = emb.detach().cpu().squeeze(dim=0).numpy() #(128,)
    return emb
