import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchvision import models


# 自带Cosine退火只包括SGDR的前半部分，自定义class把restart也加上
class CosineAnnealingLR_with_Restart(optim.lr_scheduler._LRScheduler): #继承共享scheduler
    def __init__(self, optimizer, model, out_dir, T_max=50, cycle_num=5, eta_min=1e-3, take_snapshot=False, last_epoch=-1):
        self.T_max = T_max #每个周期的epochs数，对应current_epoch属于[1,self.Te]
        self.current_epoch = last_epoch
        self.cycle = 0 #计算周期数，共cycle_num个
        self.eta_min = eta_min #最小的lr
        self.model = model
        self.out_dir = out_dir
        self.take_snapshot = take_snapshot #保存每个周期的模型
        self.lr_history = [] #得到变化历史
        super(CosineAnnealingLR_with_Restart, self).__init__(optimizer, last_epoch) #自带属性，代表当前epoch，可以直接self.引用
        
    def get_lr(self): #lr循环从base_lr下降到eta_min
        # 后面(1+cosx)取值范围是[0,2]，所以new_lrs范围是[eta_min, base_lr]
        new_lrs = [self.eta_min + (base_lr - self.eta_min)/2 *
                   (1 + np.cos(np.pi * self.current_epoch / self.T_max)) #根据当前
                   for base_lr in self.base_lrs] #optimizer的lr就是base_lrs，只有一个值[0.1]
        self.lr_history.append(new_lrs[0])
        return new_lrs

    def get_last_lr(self):
        last_lr = self.optimizer.param_groups[0]['lr']
        return [last_lr] #为了和其他scheduler统一，得到[lr]
    
    def step(self): #epoch参数逐步deprecate
        self.last_epoch += 1 #默认运行1次代表前进1个epoch
        self.current_epoch += 1 #以周期为单位的epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr #计算当前epoch的lr，并更新参数

        # restart
        if self.current_epoch == self.T_max: #一个周期结束
            self.cycle += 1 #计算周期数
            if self.take_snapshot: #如果保存模型
                ckpt_path = os.path.join(self.out_dir, f'cycle{self.cycle:02}_final_model.pth')
                torch.save({
                    'state_dict': self.model.state_dict(), #模型参数w/b信息
                    'optimizer_state_dict': optimizer.state_dict(), #包括bn的running mean和std等信息
                }, ckpt_path)

            self.current_epoch = 0 #重置周期内epoch count


def plot_lr(scheduler_fn, name, epochs=300):
    model = models.alexnet() 
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.005)
    scheduler = scheduler_fn(optimizer, model, '', 
                T_max=50, cycle_num=5, take_snapshot=False)
    x = []
    y = []
    for e in range(epochs):
        optimizer.step() #先optimizer进一步，再到lr scheduler
        scheduler.step()
        lr = scheduler.get_last_lr() #返回当下[lr]
        x.append(e)
        y.append(lr)
    plt.plot(x, y)
    plt.title(name)
    plt.show()


def load_optimizer_checkpoint(optimizer, pretrained_path, device):
    # 继续训练会要记录上次optimizer的状况
    checkpoint = torch.load(pretrained_path, map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #参数在cpu
    #param_groups会跟随model.to(device)而转换到gpu，但是state仍然在cpu
    for state in optimizer.state.values(): #取出里面的mementum_buffer
        for k, v in state.items():
            state[k] = v.to(device) #再次分配到gpu
    return optimizer


def load_scheduler_checkpoint(lr_scheduler, pretrained_path, device):
    # 继续训练会要记录上次lr_scheduler的状况
    checkpoint = torch.load(pretrained_path, map_location=device)
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return lr_scheduler


if __name__ == '__main__':
    plot_lr(CosineAnnealingLR_with_Restart, 'cosine_annealing_lr', epochs=250)
