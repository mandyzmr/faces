import torch
from torch import nn

# 模型共用的网络函数
class Net(nn.Module):
    def normalize(self, x):
        # 通过bn或者RGB的mean,std进行数据normalization
        if self.is_first_bn:
            x = self.first_bn(x)
        else:
            mean=[0.485, 0.456, 0.406] #rgb
            std =[0.229, 0.224, 0.225]
            x = torch.cat([
                (x[:,[0]]-mean[0])/std[0],
                (x[:,[1]]-mean[1])/std[1],
                (x[:,[2]]-mean[2])/std[2],
            ],1)
        return x
    
    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['backup']:
            self.train()
            if is_freeze_bn==True: #如果不训练bn层，即服从标准正态分布，不需要学习mean和std
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval() #设置不需要backprop
                        m.weight.requires_grad = False 
                        m.bias.requires_grad   = False

def load_model_checkpoint(model, pretrained_path, device):
    # 预加载自己训练的模型
    checkpoint = torch.load(pretrained_path, map_location=device) #用gpu加载state_dict

    # 若之前保存的时候，在nn.DataParallel的状况下保存了，key会有module.
    state_dict = model.state_dict()
    for key, value in checkpoint['state_dict'].items():
        if key[7:] in state_dict: # module.layer.xxx
            state_dict[key[7:]] = checkpoint['state_dict'][key]
    model.load_state_dict(state_dict) #参数在cpu
    
    # 若之前保存的时候，有model.module.state_dict()，可以直接加载
    # model.load_state_dict(checkpoint['state_dict']) #参数在cpu

    model.to(device) #再次分配到gpu
    # print(f"Loaded checkpoint: acc - {checkpoint['acc']:.4f}, acer - {checkpoint['acer']:.4f}")
    return model
   