import numpy as np
import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
from collections import OrderedDict
from net import *
from senet import *
from resnet18 import *


class FaceBagNet_ResNet18(Net):
    '''
    Sub-Model: ResNet18分别提取3个模态的特征
    Fusion: ResNet的Basic Block处理融合特征
    '''
    def __init__(self, num_classes=2):
        super(FaceBagNet_ResNet18,self).__init__()       
        # 每个模态都是通过预训练resnet18的submodel提取特征
        self.color_RES  = ResNet18(num_classes=num_classes,is_first_bn=True)
        self.depth_RES = ResNet18(num_classes=num_classes,is_first_bn=True)
        self.ir_RES = ResNet18(num_classes=num_classes,is_first_bn=True)

        self.res_4 = self._make_layer(BasicBlock, 3*128, 256, 2, stride=2)
        self.res_5 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)

        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(512, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, num_classes))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 把每个样本的多模态分别抽取出来
        batch_size,C,H,W = x.shape #(b, 3+3+3, h, w)
        color = x[:, 0:3,:,:] 
        depth = x[:, 3:6,:,:]
        ir = x[:, 6:9,:,:]
        
        # 只需前3个ResBlock分别提取多模态特征
        color_feas = self.color_RES(color) #(b,128,h,w)
        depth_feas = self.depth_RES(depth)
        ir_feas = self.ir_RES(ir)
        
        # 特征拼接融合
        fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1)
        x = self.res_4(fea)
        x = self.res_5(x)
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        x = self.fc(x)
        return x


class FaceBagNet_SENet(Net):
    '''
    Sub-Model: SeResNeXt分别提取3个模态的特征，然后SEModule筛选特征
    Fusion: SEResNeXt Bottleneck处理融合特征
    
    Parameters
    ----------
    name (str): 可选SENet154, SE-ResNet, SE-ResNeXt18, SE-ResNeXt34, SE-ResNeXt50 
    
    '''
    def __init__(self, name='SE-ResNeXt18', num_classes=2):
        super(FaceBagNet_SENet,self).__init__()
        # 每个模态都是通过SENet的submodel提取特征
        self.color_SENet  = SENet_backbone(name, num_classes=num_classes, is_first_bn=True)
        self.depth_SENet = SENet_backbone(name, num_classes=num_classes,is_first_bn=True)
        self.ir_SENet = SENet_backbone(name, num_classes=num_classes,is_first_bn=True)
        
        fea_chans = 128*self.color_SENet.block.expansion #每个模态得到128*4=512的特征channel
        self.color_SE = SEModule(fea_chans,reduction=16)
        self.depth_SE = SEModule(fea_chans,reduction=16)
        self.ir_SE = SEModule(fea_chans,reduction=16)

        self.bottleneck = nn.Sequential(nn.Conv2d(fea_chans*3, fea_chans, kernel_size=1, padding=0),
                                         nn.BatchNorm2d(fea_chans),
                                         nn.ReLU(inplace=True))

        
        self.in_chans=fea_chans #bottleneck的输出
        self.res_4 = self._make_layer(SEResNeXtBottleneck, 256, 2, groups=32, stride=2)
        self.res_5 = self._make_layer(SEResNeXtBottleneck, 512, 2, groups=32, stride=2)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(512*SEResNeXtBottleneck.expansion, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, num_classes))

    def _make_layer(self, block, out_chans, n_blocks, groups, stride=1,
                    downsample_kernel_size=1, downsample_padding=0, reduction=16):
        downsample = None
        if stride != 1 or self.in_chans != out_chans * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_chans, out_chans * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(out_chans * block.expansion),
            )

        layers = []
        layers.append(block(self.in_chans, out_chans, groups, reduction, stride, downsample))
        self.in_chans = out_chans * block.expansion
        for i in range(1, n_blocks):
            layers.append(block(self.in_chans, out_chans, groups, reduction))
        return nn.Sequential(*layers)
    

    def forward(self, x):
        # 把每个样本的多模态分别抽取出来
        batch_size,C,H,W = x.shape #(b, 3+3+3, h, w)
        color = x[:, 0:3,:,:] 
        depth = x[:, 3:6,:,:]
        ir = x[:, 6:9,:,:]
        
        # 只需前3个ResBlock分别提取多模态特征
        color_feas = self.color_SENet(color) #(b,128*4,h,w)
        depth_feas = self.depth_SENet(depth)
        ir_feas = self.ir_SENet(ir)
        
        # 对多模态特征应用attention机制，筛选重要特征
        color_feas = self.color_SE(color_feas) #(b,128*4,h,w)
        depth_feas = self.depth_SE(depth_feas)
        ir_feas = self.ir_SE(ir_feas)

        # 特征拼接融合
        fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1)
        x = self.bottleneck(fea) #缩小特征channel
        x = self.res_4(x)
        x = self.res_5(x)
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        x = self.fc(x)
        return x


class FaceBagNet_ResNet18_SENet(Net):
    '''
    Sub-Model: ResNet18/SENet分别提取3个模态的特征，然后SEModule筛选特征
    Fusion: ResNet的Basic Block处理融合特征
    '''
    def __init__(self, name='SE-ResNeXt18', num_classes=2):
        super(FaceBagNet_ResNet18_SENet,self).__init__()       
        # 每个模态都是通过submodel提取特征
        if name=='ResNet18':
            submodel = ResNet18(num_classes=num_classes,is_first_bn=True)
        else:
            submodel = SENet_backbone(name, num_classes=num_classes, is_first_bn=True)
        
        self.backbone = name
        self.color_SUB  = submodel
        self.depth_SUB = submodel
        self.ir_SUB = submodel

        fea_chans = 128*self.color_SUB.block.expansion #每个模态得到128*4=512的特征channel
        self.color_SE = SEModule(fea_chans,reduction=16)
        self.depth_SE = SEModule(fea_chans,reduction=16)
        self.ir_SE = SEModule(fea_chans,reduction=16)

        if self.backbone.startswith('SE'): #只有SENet需要把expansion后的channel压缩
            self.bottleneck = nn.Sequential(nn.Conv2d(fea_chans*3, 128*3, kernel_size=1, padding=0),
                                         nn.BatchNorm2d(128*3),
                                         nn.ReLU(inplace=True))
        
        self.res_4 = self._make_layer(BasicBlock, 3*128, 256, 2, stride=2)
        self.res_5 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)

        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(512, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, num_classes))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 把每个样本的多模态分别抽取出来
        batch_size,C,H,W = x.shape #(b, 3+3+3, h, w)
        color = x[:, 0:3,:,:] 
        depth = x[:, 3:6,:,:]
        ir = x[:, 6:9,:,:]
        
        # 只需前3个ResBlock分别提取多模态特征
        color_feas = self.color_SUB(color) #(b,128,h,w)
        depth_feas = self.depth_SUB(depth)
        ir_feas = self.ir_SUB(ir)
        
        # 对多模态特征应用attention机制，筛选重要特征
        color_feas = self.color_SE(color_feas) #(b,128*4,h,w)
        depth_feas = self.depth_SE(depth_feas)
        ir_feas = self.ir_SE(ir_feas)

        # 特征拼接融合
        x = torch.cat([color_feas, depth_feas, ir_feas], dim=1)
        if self.backbone.startswith('SE'):
            x = self.bottleneck(x) #缩小特征channel
        x = self.res_4(x)
        x = self.res_5(x)
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        x = self.fc(x)
        return x


def get_model(name='SE-ResNeXt18', modality='fusion', num_classes=2, attention=False, bottleneck=False):
    if modality!='fusion': #单模态模型
        if name=='ResNet18':
            model = ResNet18(num_classes, is_first_bn=True, backbone=False)
        elif name.startswith('SE'):
            model = SENet_backbone(name, num_classes, is_first_bn=True, backbone=False)
    else: #多模态模型
        if name=='ResNet18' and attention==False:
            model = FaceBagNet_ResNet18(num_classes)
        elif name=='ResNet18' and attention==True:
            model = FaceBagNet_ResNet18_SENet(name, num_classes)
        elif name.startswith('SE') and bottleneck==False:
            model = FaceBagNet_ResNet18_SENet(name, num_classes)
        elif name.startswith('SE') and bottleneck==True:
            model = FaceBagNet_SENet(name, num_classes)
    return model


if __name__ == '__main__':
    model = get_model(modality='color')
    summary(model, (3, 32, 32))
