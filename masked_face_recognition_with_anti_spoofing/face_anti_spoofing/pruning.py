from torch.nn.utils import prune
from torch import nn
import numpy as np
from torchvision import models


def sparsity(model): #模型参数稀疏程度%
    num, zero_num = 0., 0.
    for p in model.parameters(): #遍历参数
        num += p.numel() #参数值个数
        zero_num += (p == 0).sum() #参数值为0的个数
    return zero_num / num


def prune_value(model, amount=0.3): #以参数值为单位进行归零
    print(f'Before pruning: {sparsity(model):.2%} global sparsity')
    for m in model.modules(): #从外层到内层逐步遍历
        if isinstance(m, nn.Conv2d):
            #从L1 norm最小的参数开始归零，返回weight_orig和对应的weight_mask
            prune.l1_unstructured(m, name='weight', amount=amount)  
            prune.remove(m, 'weight')  #把mask和原参数值相乘
    print(f'After pruning: {sparsity(model):.2%} global sparsity')
    return model


def prune_filter(model, amount=0.1): #以filter为单位进行裁剪
    print(f'Before pruning: {sparsity(model):.2%} global sparsity')
    pre_layer = None #前层Conv
    pre_layer_bn = None #前层Bn
    pre_pruned_idx = None #前层被裁掉的filter dix
    
    for m in model.modules(): #只适用于串联结构模型
        # 主要目的是prune Conv2d的filter
        if isinstance(m, nn.Conv2d):
            out_chans, in_chans, f, _ = m.weight.shape 
            print(m, m.weight.shape,'->')
            # 如果由于前层Conv/Bn去掉filters，导致input feature map缺失，需要先把受前层影响的卷积核去掉
            if pre_layer is not None and pre_pruned_idx is not None:
                if m.groups == 1: #当前层准备做正常卷积
                    m.weight.data = np.delete(m.weight.data, pre_pruned_idx, axis=1) #裁掉
                    m.in_channels = m.weight.shape[1] #需要手动更改in_chans
                    # print(m.weight.shape,'->')
                else: #但是如果当前层准备做group卷积，前层缺少的feature map会影响某些分组，导致运算无法正常进行
                    pre_layer.weight.data = backup_conv_w #恢复前层Conv原参数，不做裁剪
                    if pre_layer.bias.data.any(): #如果存在bias也复原，虽然BN前的Conv通常没有设置bias，但如果不接BN就很有可能带bias
                        pre_layer.bias.data = backup_conv_b 
                    pre_layer.out_channels = backup_conv_w.shape[0] 
                    
                    if pre_layer_bn is not None: #如果Conv后还跟了BN，一起恢复
                        pre_layer_bn.weight.data = backup_bn
                        pre_layer_bn.num_features = backup_bn.shape[0]
                    
            # 再把当前层从L2 norm最小的filter开始去掉
            l2_norm, idx = (m.weight**2).sum(dim=[1,2,3]).sort() #根据filters的L2 norm排序，为简化计算免去np.sqrt
            pruned_num = int(out_chans*amount) #根据每层filters数，裁剪amount%
            pruned_idx = idx[:pruned_num] #需要裁掉的filter idx
            print(pruned_num)
            backup_conv_w = m.weight.data.clone() #由于会影响下一层，先为后续运算备份数据
            m.weight.data = np.delete(m.weight.data, pruned_idx, axis=0) #裁掉
            m.out_channels = m.weight.shape[0] #需要手动更改out_chans
            if m.bias.data.any(): #如果存在bias，因为BN前的Conv通常没有设置bias
                backup_conv_b = m.bias.data
                m.bias.data = np.delete(m.bias.data, pruned_idx, axis=0) #裁掉
            pre_layer = m #作为下一层的前层
            pre_pruned_idx = pruned_idx 
            print(m.weight.shape, m.bias.shape, m.out_channels)
                      
        # 通常跟在pruned Conv后，out_chans会受影响
        if isinstance(m, nn.BatchNorm2d): 
            if pre_layer is not None and pre_pruned_idx is not None: #以防是在模型开头做normalization
                backup_bn = m.weight.data
                m.weight.data = np.delete(m.weight.data, pre_pruned_idx, axis=0) #裁掉, BN只是针对每个(out_chans,)有running mean/std
                m.num_features = m.weight.data.shape[0]
                pre_layer_bn = m #额外保存作为前层bn，因为如果下一层Group Conv需要复原前层，那么需要把Conv和Bn同时恢复，不能只恢复Bn
        
        #由于FC通常在模型最后，前层一般是pruned Conv/bn+GAP，维持裁剪后的out_chans
        #但是如果前面有Flatten重置out_chans就不适用
        if isinstance(m, nn.Linear): 
            if pre_layer is not None and pre_pruned_idx is not None: #以防是在模型开头做normalization
                m.weight.data = np.delete(m.weight.data, pre_pruned_idx, axis=1) #裁掉
                m.in_features = m.weight.data.shape[1]
                pre_layer = None #由于FC不做裁剪，不会影响后续层
                pre_pruned_idx = None
            
    print(f'After pruning: {sparsity(model):.2%} global sparsity')
    return model


if __name__ == '__main__':
	model = models.alexnet()
	prune_value(model, 0.1)

