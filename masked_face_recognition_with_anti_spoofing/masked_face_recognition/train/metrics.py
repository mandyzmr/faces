import torch
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import os
from config import *
import glob
import imageio

#用概率更高值来决定多分类类别
def confusion_matrix_multi(y_pred, y_true): 
    y_pred = torch.argmax(y_pred, axis=1)
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro',zero_division=0) #由于存在imbalanced
    precision = precision_score(y_true, y_pred, average='macro',zero_division=0) 
    f1 = f1_score(y_true, y_pred, average='macro',zero_division=0)
    metrics = {'precision':precision, 'recall':recall, 'acc':acc, 'f1':f1}
    return metrics


def confusion_matrix_dist(dist, y_true, threshold=0.5, beta=1):
    '''
    Parameters
    ----------
    beta (int): when beta>1, recall-oriented; 
                when beta<1, precision-oriented
    '''
    y_pred = dist<threshold #用阈值来决定类别
    tp = np.logical_and(y_true==1, y_pred==1).sum()
    fp = np.logical_and(y_true==0, y_pred==1).sum()
    tn = np.logical_and(y_true==0, y_pred==0).sum()
    fn = np.logical_and(y_true==1, y_pred==0).sum()
    
    fpr = 0 if (tn+fp)==0 else fp/(tn+fp)
    precision = 0 if (tp+fp)==0 else tp/(tp+fp)
    recall = 0 if (tp+fn)==0 else tp/(tp+fn) 
    acc = (tp+tn)/(tp+fp+tn+fn)
    f1 = (1+beta**2)*tp/((1+beta**2)*tp+fn+fp)
    metrics = {'fpr':fpr, 'precision':precision, 'recall':recall, 'acc':acc, 'f1':f1}
    return metrics


def calculate_roc(distances, labels, n_folds=10):
    thresholds = np.arange(min(distances), max(distances), 0.001) #根据dist生成用于遍历的阈值
    indices = np.arange(len(labels)) #samples idx
    n_thresholds = len(thresholds) #遍历的thresholds数
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=False) #10折交叉验证
    tpr = np.zeros((n_folds, n_thresholds))
    fpr = np.zeros((n_folds, n_thresholds))
    acc = np.zeros((n_folds, n_thresholds))
    precision = np.zeros(n_folds)
    recall = np.zeros(n_folds)
    best_thres = np.zeros(n_folds)
    
    for fold_i, (train_idx, test_idx) in enumerate(kfold.split(indices, labels)): #分成10折的indices
        # 求得在目标acc(最高)下的对应阈值
        for thres_i, thres in enumerate(thresholds):
            metrics = confusion_matrix_dist(distances[train_idx], labels[train_idx], thres)
            acc[fold_i, thres_i] = metrics['acc']
        best_thres_idx = np.argmax(acc[fold_i]) 

        # 求得该阈值下的metrics和平均tpr/fpr
        for thres_i, thres in enumerate(thresholds):
            metrics = confusion_matrix_dist(distances[test_idx], labels[test_idx], thres)
            tpr[fold_i, thres_i] = metrics['recall']
            fpr[fold_i, thres_i] = metrics['fpr']
        metrics = confusion_matrix_dist(distances[test_idx], labels[test_idx], thresholds[best_thres_idx])
        precision[fold_i] = metrics['precision']    
        recall[fold_i] = metrics['recall']
        best_thres[fold_i] = thresholds[best_thres_idx]
    acc = np.max(acc, axis=1) #(n_folds,) 每个折的最大值
    tpr = np.mean(tpr, axis=0) #(n_thresholds,) 10折下遍历threshold的平均值
    fpr = np.mean(fpr, axis=0)
    metrics_cv = {'tpr':tpr, 'fpr':fpr,'precision':precision, 'recall':recall, 'acc':acc, 'thres':best_thres}
    return metrics_cv


def calculate_tpr_fpr(distances, labels, fpr_target=1e-2, n_folds=10):
    thresholds = np.arange(min(distances), max(distances), 0.001) #根据dist生成用于遍历的阈值
    indices = np.arange(len(labels)) #samples idx
    n_thresholds = len(thresholds) #遍历的thresholds数
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=False) #10折交叉验证

    tpr = np.zeros(n_folds) #TAR (True Accept Rate): 摄像头前的人通过人脸识别系统的概率，即TPR
    fpr = np.zeros(n_folds) #FAR

    for fold_i, (train_idx, test_idx) in enumerate(kfold.split(indices, labels)): #分成10折的indices
        FPR = np.zeros(n_thresholds)
        for thres_i, thres in enumerate(thresholds):
            metrics = confusion_matrix_dist(distances[train_idx], labels[train_idx], thres)
            FPR[thres_i] = metrics['fpr']
        
        # 求得在目标fpr下的对应阈值
        f = interpolate.interp1d(FPR, thresholds, kind= 'slinear') #得到threshold=f(fpr)的函数关系
        # 阈值越小越严格，FPR随着阈值增大而变大，当遍历到最大阈值时，代表所有样本都预测为P，除非真实样本确实都为P，否则np.max(fpr)=1
        fpr_target = np.clip(fpr_target, None, np.max(FPR)) #如果target>np.max(FPR)，代表只需要满足最宽松的条件，即最大阈值
        if fpr_target <= np.min(FPR): #但是如果target比可能实现的最小值都小，就只能提高target，同时打印出来告知
            fpr_target = np.min(FPR)
            print(f'FPR target -> {fpr_target:.3e}')
        thres = f(fpr_target) 

        # 求得该阈值下的metrics
        metrics = confusion_matrix_dist(distances[test_idx], labels[test_idx], thres)
        tpr[fold_i] = metrics['recall'] 
        fpr[fold_i] = metrics['fpr']
        
    return tpr, fpr


#对valid_loader做10折交叉验证
def evaluate_metrics(distances, labels, epoch, fpr_target=1e-2, n_folds=10, plot=True, roc_dir='', valid_set='LFW_Unmasked'):    
    #总体评价：找到acc最高时的dist threshold，得到平均tpr/fpr/auc和每折下的precision/recall/thres
    metrics_cv = calculate_roc(distances, labels, n_folds)
    tpr = metrics_cv['tpr'] #(n_thresholds,)
    fpr = metrics_cv['fpr']
    roc_auc_score = auc(fpr, tpr)
    metrics_cv['auc']=roc_auc_score
    if plot:
        plot_roc(fpr, tpr, roc_auc_score, epoch, roc_dir, valid_set)
    
    #目标评价：找到target fpr时的dist threshold，得到每折下的tpr/fpr，为区分用tar/far表示
    tar, far = calculate_tpr_fpr(distances, labels, fpr_target, n_folds)
    metrics_cv['tar']=tar
    metrics_cv['far']=far
    return metrics_cv


def plot_roc(fpr, tpr, roc_auc_score, epoch, roc_dir='', valid_set='LFW_Unmasked'):
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{valid_set} (Epoch {epoch} @ AUC = {roc_auc_score:.2%})')
    plt.gca().set_aspect('equal')
    plt.savefig(os.path.join(roc_dir, f'epoch{epoch:02}_{valid_set}_auc_{roc_auc_score:.2f}.png'))
    # plt.show()


def generate_rou_gif(rou_dir=''):
    for valid_set in ['LFW_Unmasked', 'LFW_Masked']:
        filenames = glob.glob(os.path.join(rou_dir,f'epoch*_{valid_set}_auc_*.png')) #完整路径，同时避免.DS_stores等隐藏文件
        filenames = sorted(filenames) #确保按顺序
        plots = []
        for filename in filenames:
            image = plt.imread(filename)
            image = (image*255).astype('uint8')
            plots.append(image)
        imageio.mimsave(os.path.join(rou_dir, f'roc_{valid_set}.gif'), plots, 'GIF-FI', fps=2)
        # display(IPyImage(open(os.path.join(rou_dir, f'roc_{valid_set}.gif'), 'rb').read())) #显示动画
