import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate
from sklearn.metrics import auc

def accuracy(y_pred, y_true): #用概率更高值来决定类别
    y_pred = torch.argmax(y_pred, axis=1)
    acc = y_pred.eq(y_true) 
    acc = acc.detach().cpu().numpy() 
    acc = np.mean(acc)
    return acc


def confusion_matrix(y_pred, y_true, threshold=0.5, beta=1):
    '''
    Parameters
    ----------
    beta (int): when beta>1, recall-oriented; 
                when beta<1, precision-oriented
    '''
    y_pred = y_pred[:,1]>threshold #用阈值来决定类别
    tp = np.logical_and(y_true==1, y_pred==1).sum()
    fp = np.logical_and(y_true==0, y_pred==1).sum()
    tn = np.logical_and(y_true==0, y_pred==0).sum()
    fn = np.logical_and(y_true==1, y_pred==0).sum()
    
    fpr = 0 if (tn+fp)==0 else fp/(tn+fp)
    precision = 0 if (tp+fp)==0 else tp/(tp+fp)
    recall = 0 if (tp+fn)==0 else tp/(tp+fn)
    f1 = (1+beta**2)*tp/((1+beta**2)*tp+fn+fp)
    acc = (tp+tn)/(tp+fp+tn+fn)
    metrics = {'fpr':fpr,'precision':precision, 'recall':recall, 'f1':f1, 'acc':acc}
    return metrics


def ACER(y_pred, y_true):
    metrics = confusion_matrix(y_pred, y_true)
    apcer = metrics['fpr'] #把假人预测为真人的概率
    npcer = 1-metrics['tpr'] #把真人预测为假人的概率
    acer = (apcer + npcer) / 2.0 #平均错误概率
    acc = metrics['acc'] #一般准确率
    return acer, acc


# 求得当FPR=target时，对应的TPR@FPR=1e-3
def TPR_FPR(y_pred, y_true, fpr_target = 1e-3): #活体检测侧重于最小化fpr，即把假人预测为真人的概率，验证更严格
    threshold = np.arange(0.0, 1.0, 0.001) #在[0,1]之间有多个阈值，根据阈值不同，metrics结果也不一样
    fpr = np.zeros(len(threshold))
    tpr = np.zeros(len(threshold))
    for i, thres in enumerate(threshold):
        metrics = confusion_matrix(y_pred, y_true, thres)
        fpr[i] = metrics['fpr']
        tpr[i] = metrics['recall']
    roc_auc_score = auc(fpr, tpr)
    
    f = interpolate.interp1d(fpr, threshold, kind= 'slinear') #得到threshold=f(fpr)的函数关系
    # 阈值越大越严格，FPR随着阈值减小而变大，当遍历到最小阈值时，代表所有样本都预测为P，除非真实样本确实都为P，否则np.max(fpr)=1
    fpr_target = np.clip(fpr_target, None, np.max(fpr)) #如果target>np.max(fpr)，代表只需要满足最宽松的条件，即最小阈值
    if fpr_target <= np.min(fpr): #但是如果target比可能实现的最小值都小，就只能提高target，同时打印出来告知
        fpr_target = np.min(fpr)
        print(f'FPR target -> {fpr_target}')
    thres = f(fpr_target) #求得在目标fpr下的对应阈值

    metrics = confusion_matrix(y_pred, y_true, thres)
    FPR = metrics['fpr']
    TPR = metrics['recall']
    
    return FPR, TPR, roc_auc_score



if __name__ == '__main__':
    y_pred = torch.rand(100,2)
    y_pred = F.softmax(y_pred, dim=1).numpy()
    y_true = np.random.choice(2,100)
    print(TPR_FPR(y_pred, y_true))
