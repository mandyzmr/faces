import numpy as np
from torch import nn
from torchsummary import summary
from torchvision.models.resnet import BasicBlock, Bottleneck #原版直接使用
from torch.utils import model_zoo
from attention import *
from train.criterion import AttentionLoss

class BasicBlock_CBAM(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        # 在block的最后增加一个CBAM模块
        self.channel_att = ChannelAttention(planes)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.channel_att(x)*x
        x = self.spatial_att(x)*x
        
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class Bottleneck_CBAM(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CBAM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # 在block的最后增加一个CBAM模块
        self.channel_att = ChannelAttention(planes * 4)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.channel_att(x)*x
        x = self.spatial_att(x)*x

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNet_CBAM(nn.Module):
    def __init__(self, block, layers, embedding_dim=128, n_classes=500, 
                 att_criterion=False, softmax_criterion=False):
        super(ResNet_CBAM, self).__init__()
        # Stem
        self.inplanes = 64 #stem后的channel
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.spatial_att = SpatialAttention() #新增att，因为后续的layer block都已经增加了CBAM模块
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Resnet block
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #由于修改了结构，和预训练参数的shape会不一样，最好直接改层名，使其不会被匹配上
        self.fc_emb = nn.Linear(512 * block.expansion, embedding_dim) 
        
        # 在最后2个layer输出增加att模块，是主要用于可视化，以及检查权重和mask之间的att loss，让关注度集中在没有被遮挡的脸部区域
        self.spatial_att1 = SpatialAttention() 
        self.spatial_att2 = SpatialAttention()
        self.att_loss = AttentionLoss() 
        self.att = att_criterion #是否计算mask的att loss
        self.softmax = softmax_criterion #是否计算softmax loss
        if self.softmax: 
            self.classifier = nn.Linear(embedding_dim, n_classes) #根据这个人的emb，找到这个人
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        h,w = x.shape[2:]  #(b,3,h,w)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.spatial_att(x)*x
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        att1 = self.spatial_att1(x)
        x = att1*x #(b,256*expansion,14,14)
        x = self.layer4(x)
        att2 = self.spatial_att2(x)
        x = att2*x #(b,512*expansion,7,7)
        heatmap = x #后续可以给最后一层Conv画热力图
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_emb(x)
        
        # l2 normalized emb
        emb = x/torch.norm(x, dim=1, keepdim=True) #emb/norm(emb)
        
        #先用softmax训练，再用triplet和att loss对emb做fine tune
        if self.softmax:
            #用normalized emb和weight求cos(theta)，后续计算arcface loss
            self.classifier.weight.data = self.classifier.weight.data/torch.norm(self.classifier.weight.data, dim=1, keepdim=True)
            logits = self.classifier(emb) #(b,n_classes)
            return emb, logits
        elif self.att:
            att_loss=None #初始值，只有训练的时候才计算att loss
            if self.training:
                att1_loss = self.att_loss(att1, mask) #(b,h,w)
                att2_loss = self.att_loss(att2, mask)
                att_loss = att1_loss + att2_loss
            return emb, att_loss, heatmap
        else:
            return emb #如果不计算att loss的话，直接返回emb


def get_resnet_cbam(name='resnet18', embedding_dim=128, n_classes=500, basicblock=BasicBlock_CBAM, bottleneck=Bottleneck_CBAM, 
                    softmax_criterion=False, att_criterion=False, pretrained=False):
    structure = {
        'resnet18': ResNet_CBAM(basicblock, [2,2,2,2], embedding_dim, n_classes, att_criterion, softmax_criterion),
        'resnet34': ResNet_CBAM(basicblock, [3,4,6,3], embedding_dim, n_classes, att_criterion, softmax_criterion),
        'resnet50': ResNet_CBAM(bottleneck, [3,4,6,3], embedding_dim, n_classes, att_criterion, softmax_criterion),
        'resnet101': ResNet_CBAM(bottleneck, [3,4,23,3], embedding_dim, n_classes, att_criterion, softmax_criterion),
        'resnet152': ResNet_CBAM(bottleneck, [3,8,36,3], embedding_dim, n_classes, att_criterion, softmax_criterion)}
    
    pretrained_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',}
    
    model = structure[name]
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(pretrained_urls[name])
        state_dict = model.state_dict()
        for key, value in pretrained_state_dict.items(): 
            if key in state_dict: #由于新模型会删除一些旧结构，不需要加载多余的参数
                state_dict[key]=pretrained_state_dict[key] #用预训练参数覆盖
        model.load_state_dict(state_dict)
    return model



if __name__=='__main__':
    model = get_resnet_cbam('resnet18', basicblock=BasicBlock, bottleneck=Bottleneck, att_criterion=True, softmax_criterion=False)
    image = torch.rand(1,3,256,256)
    mask = np.random.rand(1,256,256)
    emb, att_loss, heatmap = model(image, mask)
    # emb, logits = model(image, mask)
    # emb = model(image, mask)
    print(emb.shape)
    # print(logits.shape)
    print(att_loss) 
    print(heatmap.shape)   
    # print(summary(model, [(3,256,256), (256,256)]))


