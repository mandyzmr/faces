import torch
from torch import nn
from torchvision import models
from torchsummary import summary
from config import *


# 没有Attention机制的baseline
class ResNet(nn.Module):
    '''
    triplet loss: 用A/P/N的emb计算相同和不同人之间的dist，用margin对比距离差
    softmax_loss: 把人作为类别，计算cross entropy loss
    '''
    def __init__(self, name='resnet18', embedding_dim=128, n_classes=500, 
                 softmax_criterion=False, pretrained=False):
        super(ResNet, self).__init__()
        # 加载完预训练参数后，把最后一层fc重置，从1000改为emb
        self.model = self.get_resnet(name, pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)
        self.softmax = softmax_criterion
        if self.softmax: 
            self.classifier = nn.Linear(embedding_dim, n_classes) #根据这个人的emb，找到这个人
    
    def get_resnet(self, name, pretrained):
        structure = {
         'resnet18': models.resnet18,
         'resnet34': models.resnet34,
         'resnet50': models.resnet50,
         'resnet101': models.resnet101,
         'resnet152': models.resnet152
        }
        model = structure[name](pretrained=pretrained)
        return model    

    def forward(self, x):
        emb = self.model(x) #得到人脸emb (b,emb)
        # l2 normalized emb
        emb = emb/torch.norm(emb, dim=1, keepdim=True) #emb/norm(emb)
        
        #先用softmax训练，再用triplet对emb做fine tune
        if self.softmax: #用normalized emb和weight求cos(theta)，后续计算arcface loss
            self.classifier.weight.data = self.classifier.weight.data/torch.norm(self.classifier.weight.data, dim=1, keepdim=True)
            logits = self.classifier(emb) #(b,n_classes)
            return emb, logits
        else:
            return emb #测试就直接返回emb


if __name__=='__main__':
	model = ResNet('resnet50')
	print(summary(model, (3, 224, 224)))


