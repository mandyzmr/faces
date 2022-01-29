import numpy as np
import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F
from collections import OrderedDict
from net import *

class SEModule(nn.Module):
    def __init__(self, in_chans, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #Squeeze (1,1,c)
        self.fc1 = nn.Conv2d(in_chans, in_chans // reduction, kernel_size=1) #Excitation (1,1,c/r)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_chans // reduction, in_chans, kernel_size=1) #(1,1,c)
        self.sigmoid = nn.Sigmoid() #每个feature的权重

    def forward(self, x):
        inputs = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return inputs * x #加权features


class Bottleneck(nn.Module): #不同bottleneck共用的forward方法
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        x = self.se_module(x) + residual
        x = self.relu(x)
        return x
    

#正常ResNet bottleneck，只是先用SE模块筛选特征后，再和short cut相加
class SEResNetBottleneck(Bottleneck): 
    expansion = 4
    def __init__(self, in_chans, out_chans, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=stride, bias=False) #(h,w,c)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, groups=groups, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chans * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(out_chans * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


#和ResNet Bottleneck唯一不同就是3x3Conv用了32分组卷积
class SEResNeXtBottleneck(Bottleneck): 
    expansion = 4
    def __init__(self, in_chans, out_chans, groups, reduction, stride=1, 
                 downsample=None, base_width=4): #分组卷积时，每组至少生成4个filters
        super(SEResNeXtBottleneck, self).__init__()
        width = int(np.floor(out_chans * (base_width / 64)) * groups)
        self.conv1 = nn.Conv2d(in_chans, width, kernel_size=1, stride=1, bias=False) #(h,w,c)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_chans * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chans * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(out_chans * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

        
# 从1x1 Conv开始做channel expansion，2-4-4倍，由于channel太多做64分组卷积
class SEBottleneck(Bottleneck): 
    expansion = 4
    def __init__(self, in_chans, out_chans, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans * 2, kernel_size=1, bias=False) #(h,w,c)
        self.bn1 = nn.BatchNorm2d(out_chans * 2)
        self.conv2 = nn.Conv2d(out_chans * 2, out_chans * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans * 4)
        self.conv3 = nn.Conv2d(out_chans * 4, out_chans * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chans * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(out_chans * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride



class SENet(Net):
    def __init__(self, block, layers, groups, reduction=16, dropout_p=0.2,
                 in_chans=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000, backbone=True, is_first_bn = False):
        super(SENet, self).__init__()

        # 数据normalization
        self.is_first_bn = is_first_bn #先通过bn归一化
        if self.is_first_bn:
            self.first_bn = nn.BatchNorm2d(3) #图片的in_chans是3
        
        # 仅用做backbone：8倍下采样，得到128*4输出
        self.backbone = backbone
        self.in_chans = in_chans #根据不同block，layer0输出的shape不一样
        self.dropout_p = dropout_p
        self.block=block
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, in_chans, 3, stride=1, padding=1, bias=False)),
                ('bn3', nn.BatchNorm2d(in_chans)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, in_chans, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(in_chans)),
                ('relu1', nn.ReLU(inplace=True)),
            ]

        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, 64, layers[0], groups)
        self.layer2 = self._make_layer(block, 128, layers[1], groups, 2, 
                                       downsample_kernel_size, downsample_padding)
        
        # 用完整体：32倍下采样，得到512*4输出
        if not self.backbone:
            self.layer3 = self._make_layer(block, 256, layers[2], groups, 2, 
                                       downsample_kernel_size, downsample_padding)
            self.layer4 = self._make_layer(block, 512, layers[3], groups, 2, 
                                           downsample_kernel_size, downsample_padding)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            
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
    
    def forward(self, x): #(224,224,3)
        batch_size,C,H,W = x.shape
        x = self.normalize(x)
        x = self.layer0(x) #(56,56,64/128) -> in_chans
        x = self.layer1(x) #(56,56,64*4)
        x = self.layer2(x) #(28,28,128*4)
        
        # 完整SENet下半部分
        if not self.backbone:
            x = self.layer3(x) #(14,14,256*4)
            x = self.layer4(x) #(7,7,512*4)
            x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size,-1)
            if self.dropout_p is not None:
                x = F.dropout(x, p=self.dropout_p, training=self.training) 
            logit = self.fc(x)
            return logit
        return x
    

def SENet_backbone(name='SE-ResNeXt18', num_classes=2, backbone=True, is_first_bn=True):
    structure = {
     'SENet154': SENet(SEBottleneck, [2, 2, 2, 2], groups=64, reduction=16, dropout_p=0.2,
                     in_chans=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, 
                     num_classes=num_classes, backbone=backbone, is_first_bn=is_first_bn),
     'SE-ResNet18': SENet(SEResNetBottleneck, [2, 2, 2, 2], groups=1, reduction=16, dropout_p=None,
                     in_chans=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, 
                     num_classes=num_classes, backbone=backbone, is_first_bn=is_first_bn),
     'SE-ResNeXt18': SENet(SEResNeXtBottleneck, [2, 2, 2, 2], groups=32, reduction=16, dropout_p=None,
                     in_chans=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, 
                     num_classes=num_classes, backbone=backbone, is_first_bn=is_first_bn),
     'SE-ResNeXt34': SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16, dropout_p=None, 
                      in_chans=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0,
                      num_classes=num_classes, backbone=backbone, is_first_bn=is_first_bn),
     'SE-ResNeXt50': SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16, dropout_p=None, 
                      in_chans=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0,
                      num_classes=num_classes, backbone=backbone, is_first_bn=is_first_bn)
    }
    model = structure[name]
    return model



if __name__ == '__main__':
    model = SENet_backbone('SE-ResNeXt18', backbone=False)
    summary(model, (3,32,32))
