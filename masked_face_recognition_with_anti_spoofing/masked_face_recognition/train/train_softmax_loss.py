import time
from tqdm import tqdm 
from collections import defaultdict
from torch.nn.modules.distance import PairwiseDistance
from config import *
from criterion import *
from metrics import *


# 先用模型预测emb，然后通过计算dist找到会产生triplet loss的样本
def train_on_epoch(model, train_loader, optimizer, device, iter_smooth=100):
    epoch_loss = 0
    epoch_acc = 0
    n_batches = len(train_loader)
    # start = time.time()
    for i, (image, label) in enumerate(train_loader):
        model.train() #训练模式
        image = image.to(device) #(b,c,h,w)
        label = label.to(device).view(-1) #(b,)
        # print(f'Loading time: {time.time()-start}')
        
        # start = time.time()
        # 计算emb和softmax loss
        emb, y_pred  = model(image) #(b,128), (b,500)
        loss = arcface_loss_criterion(y_pred, label)
        metrics = confusion_matrix_multi(y_pred, label)
        acc = metrics['acc']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print(f'Processing time: {time.time()-start}')
        epoch_loss += loss.item()
        epoch_acc += acc
        if i%iter_smooth==0:
            print(f'Batch {i+1}/{n_batches} - Arcface loss: {epoch_loss/(i+1):.3f} - acc: {epoch_acc/(i+1):.3f}') 
        # start = time.time()
    
    #仅在drop last的时候有效，否则求loss时reduction=‘none'，再除以样本总数
    epoch_loss /= n_batches 
    epoch_acc /= n_batches 
    return epoch_loss, epoch_acc


def val_on_epoch(model, valid_loader, device):
    predictions, labels = [], []
    val_loss = 0
    n_samples = 0
    for i, (image, label) in enumerate(valid_loader):
        model.eval() #预测模式，关闭dropout和bn
        image = image.to(device)
        label = label.to(device)
        n_samples += image.shape[0]
        with torch.no_grad():
            _, y_pred  = model(image) 
            label = label.view(-1) #(b,)
            #由于val batch没有drop last，先求样本loss总和，再求样本loss均值
            #arcface loss只是为了进一步优化，验证时用正常softmax
            loss = F.cross_entropy(y_pred, label, reduction='sum') 
            val_loss += loss.item() #样本loss总和
            
            #记录下来一次性计算acc
            predictions.append(y_pred.detach().cpu()) 
            labels.append(label.detach().cpu())
    
    predictions = torch.cat(predictions) #(m,2)
    labels = torch.cat(labels) #(m,)
    val_loss /= n_samples #样本loss均值
    metrics = confusion_matrix_multi(predictions, labels) 
    acc = metrics['acc']
    precision = metrics['precision']
    recall = metrics['recall']
    print(f'Validation - loss: {val_loss:.3f} - acc: {acc:.3f} - precision: {precision:.3f} - recall: {recall:.3f}')   
    return val_loss, acc, precision, recall


def val_on_lfw(model, valid_loader, device, epoch, roc_dir='', masked=False):
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
            img1_emb = model(img1) #(b,128) 关闭softmax后，直接预测emb
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
    print(f'Validation - dist_thres: {np.mean(thres):.3f}(+-{np.std(thres):.3f}) - auc: {auc:.3f} - acc: {np.mean(acc):.3f}(+-{np.std(acc):.3f}) '\
          f'- recall: {np.mean(recall):.3f}(+-{np.std(recall):.3f}) - tar@far=1e-2: {np.mean(tar):.3f}(+-{np.std(tar):.3f})') 
    return auc, np.mean(acc), np.mean(recall)


def predict(model, image, device): #根据face_align预测emb
    model.eval() #[0,255] (h,w,c)
    image = test_transform(image)[np.newaxis,...] #[0,1] (1,c,h,w)
    image = image.to(device)
    with torch.no_grad():
        emb = model(image) #(1,128) 模型调节为不用任何loss的状态，直接输出emb
        emb = emb.detach().cpu().squeeze(dim=0).numpy() #(128,)
    return emb

