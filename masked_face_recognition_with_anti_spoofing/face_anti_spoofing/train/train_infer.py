import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from metrics import *
import time

def train_on_epoch(model, train_loader, criterion, optimizer, device, iter_smooth = 20):
    n_batches = len(train_loader)
    epoch_loss = 0
    epoch_acc = 0
    if device.__str__() =='cuda':
        non_blocking=True
    else:
        non_blocking=False
    # start = time.time()
    for i, (image, label) in enumerate(train_loader):
        model.train() #训练模式
        image = image.to(device, non_blocking=non_blocking) #(b,h,w,c)
        label = label.to(device, non_blocking=non_blocking) #(b,1)
        # print(f'{i}: Loading - {time.time()-start}')
        
        # start0 = time.time()
        y_pred = model(image) #(b,2)
        label = label.view(-1) #(b,)
        loss  = criterion(y_pred, label)
        acc = accuracy(y_pred, label)
        # print(f'Forward - {time.time()-start0}')

        # start1 = time.time()
        optimizer.zero_grad()
        # for param in model.parameters():
        #     param.grad=None
        loss.backward()
        optimizer.step()
        # print(f'Backward - {time.time()-start1}')

        # 输出内容
        epoch_loss += loss.item() #累计epoch内batch loss
        epoch_acc += acc.item()
        # print(f'Processing time: {time.time()-start0}')
        if i%iter_smooth==0:
            print(f'Batch {i+1}/{n_batches} - loss: {epoch_loss/(i+1):.4f} - acc: {epoch_acc/(i+1):.4f}')         
        # start = time.time()

    epoch_loss /= n_batches #平均epoch loss
    epoch_acc /= n_batches #仅在drop last的时候有效，否则求loss时reduction=‘none'，再除以样本总数
    return epoch_loss, epoch_acc


def val_on_epoch(model, val_loader, criterion, device):
    y_pred_all, y_true_all = [], []
    val_loss = 0
    n_samples = 0
    for i, (image, label) in enumerate(val_loader):
        model.eval() #预测模式，关闭dropout和bn
        b, n, c, h, w = image.shape #多个patches
        n_samples += b 
        image = image.view(b*n, c, h, w).to(device) #把n个patches分开，输入模型
        label = label.to(device)
        with torch.no_grad():
            y_pred = model(image)
            y_pred = y_pred.view(b,n,2) #对每个样本衍生的n个patches求样本类别概率均值 
            y_pred = torch.mean(y_pred, dim=1, keepdim=False) #(b,2)
            label = label.view(-1) #(b,)
            #由于val batch没有drop last，先求样本loss总和，再求样本loss均值
            loss  = criterion(y_pred, label, reduction='sum') 
            val_loss += loss.item() #样本loss总和
            #记录下来一次性计算acc
            y_pred = F.softmax(y_pred, dim=1)
            y_pred_all.append(y_pred.detach().cpu().numpy()) 
            y_true_all.append(label.detach().cpu().numpy())
    
    val_loss /= n_samples #样本loss均值
    y_pred_all = np.concatenate(y_pred_all) #(m,2)
    y_true_all = np.concatenate(y_true_all) #(m,)
    acer, acc = ACER(y_pred_all, y_true_all) 
    fpr, tpr, auc = TPR_FPR(y_pred_all, y_true_all) #得到TPR@FPR=1e-3
    print(f'Validation - loss: {val_loss:.4f} - acer - {acer:.4f} - acc: {acc:.4f} - auc: {auc:.4f}')   
    return val_loss, acer, acc, auc


def preprocess_for_prediction(path):
    image = cv2.imread(path) 
    image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
    image = augumentor(image, target_shape=(PATCH_SIZE,PATCH_SIZE,3), 
                       is_infer=True, n_patches=36) #(n,h,w,3)
    image = np.transpose(image, (0, 3, 1, 2)) #(n,3,h,w)
    image = image.astype(np.float32)
    image = image / 255.0
    image = torch.FloatTensor(image)
    image = image.unsqueeze(0) #(1,n,3,h,w)
    fake_label = None
    return image, fake_label

    
def predict(model, test_loader, device): 
    y_pred_all = [] 
    for i, (image, label) in enumerate(test_loader):
        model.eval() #预测模式，关闭dropout和bn
        b, n, c, h, w = image.shape #多个patches
        image = image.view(b*n, c, h, w).to(device) #把n个patches分开，输入模型
        with torch.no_grad():
            y_pred = model(image)
            y_pred = y_pred.view(b,n,2) #对每个样本衍生的n个patches求样本类别概率均值 
            y_pred = torch.mean(y_pred, dim=1, keepdim=False) #(b,2)
            y_pred = F.softmax(y_pred, dim=1) #(b,2) 概率
            y_pred_all.append(y_pred.detach().cpu().numpy())
        
    y_pred_all = np.concatenate(y_pred_all) #(m,2)
    return y_pred_all[:,1] #只返回概率 (m,)


def submission(filepath, save_path, prob):
    data = pd.read_csv(filepath, sep=' ', names=['color','depth','ir','prob'])
    data['prob'] = prob
    data.to_csv(save_path, header=None, index=None, sep=' ')
    print('Probability has been updated')
    return data


def ensemble(predictions, save_path):
    tmp_pred  = 0
    for path in predictions:
        pred = pd.read_csv(path, sep=' ', names=['color','depth','ir','prob'])
        tmp_pred += pred['prob']
                
    tmp_pred /= len(predictions)
    data = submission(path, 'ensemble.txt', tmp_pred)
    return data
