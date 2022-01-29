import torch
from torch import nn
from torchvision import models
from torchvision.models.resnet import BasicBlock 
from torchsummary import summary
import torch.nn.functional as F
from net import *


class ResNet18(Net):
    def __init__(self, num_classes=2, dropout_p=0.5, is_first_bn=False, backbone=True):
        super(ResNet18,self).__init__() 
        # 数据normalization
        self.is_first_bn = is_first_bn #先通过bn归一化
        if self.is_first_bn:
            self.first_bn = nn.BatchNorm2d(3) #图片的in_chans是3
            
        # 仅用做backbone：8倍下采样，得到128输出
        self.dropout_p = dropout_p
        self.backbone=backbone #是否用整个结构，还是只有前3个block
        self.encoder  = models.resnet18(pretrained=True) #预训练参数
        self.block = BasicBlock #用于获取expansion
        
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Sequential(self.encoder.conv1, #(7,7), s=2, 64
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool) 
        self.conv2 = self.encoder.layer1 #64
        self.conv3 = self.encoder.layer2 #128
        
        # 用完整体：32倍下采样，得到512输出
        if not self.backbone:
            self.conv4 = self.encoder.layer3 #256
            self.conv5 = self.encoder.layer4 #512
            self.fc = nn.Linear(512, num_classes)

    def forward(self, x): #(224,224,3)
        batch_size,C,H,W = x.shape
        x = self.normalize(x)
        x = self.conv1(x) #(5664),56,
        x = self.conv2(x) #(56,56,64)
        x = self.conv3(x) #(28,28,128)
        
        if not self.backbone:
            x = self.conv4(x) #(14,14,256)
            x = self.conv5(x) #(7,7,512)
            x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size,-1)
            fea = F.dropout(x, p=self.dropout_p, training=self.training)
            logit = self.fc(fea)
            return logit
        return x


if __name__ == '__main__':
	model = ResNet18(backbone=False)
	summary(model,(3,224,224))
