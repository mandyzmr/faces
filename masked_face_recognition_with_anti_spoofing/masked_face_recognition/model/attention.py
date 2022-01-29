import torch
from torch import nn
from torchsummary import summary


class ChannelAttention(nn.Module):
    def __init__(self, in_chans, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #Squeeze (1,1,c)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Pooling共享Conv
        self.fc1 = nn.Conv2d(in_chans, in_chans // reduction, kernel_size=1, bias=False) #Excitation (1,1,c/r)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_chans // reduction, in_chans, kernel_size=1, bias=False) #(1,1,c)
        self.sigmoid = nn.Sigmoid() #每个feature的权重

    def forward(self, x):
        inputs = x
        avg_pool = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_pool = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        x = self.sigmoid(avg_pool+max_pool)
        # x = inputs*x #加权features
        return x #attention weights，后续可视化heapmap，以及计算和mask之间的loss


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1 #Same Conv
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False) 
        self.sigmoid = nn.Sigmoid() #每个pixel的权重

    def forward(self, x):
        inputs = x
        avg_pool = torch.mean(x, dim=1, keepdim=True) #(b,1,h,w)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        # x = inputs*x #加权features
        return x 


class FaceAttention(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(FaceAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_chans, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


if __name__=='__main__':
    block = ChannelAttention(128)
    print(summary(block, (128, 4,4)))

