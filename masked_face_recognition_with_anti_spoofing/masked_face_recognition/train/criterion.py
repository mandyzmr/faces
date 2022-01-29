import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms 
import numpy as np
from config import *

triplet_loss_criterion = nn.TripletMarginLoss(margin=0.2) # l2 norm dist

class AttentionLoss(nn.Module):
    def forward(self, att_map, mask):
        # 得到不同层提取的att map形状：(b,1,h,w)的tensor
        h,w = att_map.shape[2:]
        # 得到和att map尺寸一样的mask：(b,h,w)
        mask = mask/255 #从[0,255]转成[0,1]
        mask = torch.unsqueeze(mask, dim=1).float() #(b,1,h,w)的float32 tensor
        mask = nn.Upsample((h,w), mode='bilinear')(mask) #可以批量操作，而cv2.resize只能对单张图片操作
        att_loss = F.binary_cross_entropy(att_map, mask, reduction='none') 
        att_loss = torch.mean(att_loss, dim=[1,2,3]) #求每个样本的loss，方便后续对样本筛选 (b,)
        return att_loss

att_loss_criterion = AttentionLoss()

class ArcFaceLoss(nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, margin=0.5, scale=30):
        super(ArcFaceLoss, self).__init__()
        self.margin=margin
        self.scale=scale
        
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, -1,1) #logit=cos(theta)，先截取范围，有时候clip不行用clamp
        label_onehot = F.one_hot(y_true, num_classes=N_CLASSES)
        logits, _ = torch.max(y_pred*label_onehot, axis=1, keepdim=True) # 利用one hot找到每个样本正确类别下的logit (b,)
        theta = torch.acos(logits) #通过logit=cos(theta)得到theta，有时候arccos不行用acos
        y_pred = (torch.cos(theta + self.margin) - logits)*label_onehot + y_pred #把对应类别的logit从cos(theta)替换为cos(theta+m)
        y_pred = y_pred * self.scale #乘上伸缩系数
        loss = F.cross_entropy(y_pred, y_true) #样本平均loss
        return loss

arcface_loss_criterion = ArcFaceLoss()


if __name__=='__main__':
    # Triplet loss
    anc = torch.rand(2,128)
    pos = torch.rand(2,128)
    neg = torch.rand(2,128)
    print(triplet_loss_criterion(anc, pos, neg))

    # Attention loss
    att_map = torch.rand(2,1,3,3)
    mask = np.random.rand(2,5,5)
    print(att_loss_criterion(att_map, mask)) #(b,)
    
    # Arcface loss
    y_pred = torch.rand(10,3)
    y_true = torch.from_numpy(np.random.choice(3, 10))
    print(arcface_loss_criterion(y_pred, y_true))
