import torch
import sys
from config import *
from model.resnet_cbam import *
from model.resnet import *
from preprocess.single_dataset import *
from preprocess.pair_dataset import *
from preprocess.triplet_dataset import *


def get_dataset(softmax_criterion=False, att_criterion=False, masked_face=False, fusion_face=False):
    if softmax_criterion: #用softmax loss训练
        train_dataset = SingleDataset(IMAGE_SIZE, transform=train_transform, masked=masked_face,
                                        mode='train', train_size=TRAIN_SIZE)
        train_loader  = DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE, 
                                   drop_last=True, num_workers=num_workers, pin_memory=pin_memory) 
        valid_dataset = SingleDataset(IMAGE_SIZE, transform=test_transform, masked=masked_face, 
                                        mode='test', train_size=TRAIN_SIZE)
        valid_loader  = DataLoader(valid_dataset, shuffle=False, batch_size = BATCH_SIZE//8, 
                                   drop_last=False, num_workers=num_workers, pin_memory=pin_memory)
        return (train_dataset, train_loader), (valid_dataset, valid_loader)
    
    elif att_criterion: #用triplet和attention loss训练
        train_dataset = TripletDataset(IMAGE_SIZE, transform=train_transform, masked=masked_face, fusion=fusion_face, 
                                     mask=True, bbox=False, n_triplets=N_TRIPLETS) #有mask
        train_loader  = DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE, 
                                   drop_last=False,  num_workers=num_workers, pin_memory=pin_memory) #用triplet训练时，因为每个batch还需要做进一步样本筛选，不需要drop last
        valid_dataset = PairDataset(img_size=IMAGE_SIZE, transform = test_transform, masked=False)
        valid_loader  = DataLoader(valid_dataset, shuffle=False, batch_size = BATCH_SIZE//8, 
                                   drop_last=False,  num_workers=num_workers, pin_memory=pin_memory) 
        valid_masked_dataset = PairDataset(img_size=IMAGE_SIZE, transform = test_transform, masked=True)
        valid_masked_loader  = DataLoader(valid_masked_dataset, shuffle=False, batch_size = BATCH_SIZE//8, 
                                          drop_last=False,  num_workers=num_workers, pin_memory=pin_memory) 
        return (train_dataset, train_loader), (valid_dataset, valid_loader), (valid_masked_dataset, valid_masked_loader)

    else: #用triplet loss训练
        train_dataset = TripletDataset(IMAGE_SIZE, transform=train_transform, masked=masked_face, fusion=fusion_face,
                                     mask=False, bbox=False, n_triplets=N_TRIPLETS) #没有mask
        train_loader  = DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE, 
                                   drop_last=False,  num_workers=num_workers, pin_memory=pin_memory) 
        valid_dataset = PairDataset(img_size=IMAGE_SIZE, transform = test_transform, masked=False)
        valid_loader  = DataLoader(valid_dataset, shuffle=False, batch_size = BATCH_SIZE//8, 
                                   drop_last=False,  num_workers=num_workers, pin_memory=pin_memory) 
        valid_masked_dataset = PairDataset(img_size=IMAGE_SIZE, transform = test_transform, masked=True)
        valid_masked_loader  = DataLoader(valid_masked_dataset, shuffle=False, batch_size = BATCH_SIZE//8, 
                                          drop_last=False,  num_workers=num_workers, pin_memory=pin_memory) 
        return (train_dataset, train_loader), (valid_dataset, valid_loader), (valid_masked_dataset, valid_masked_loader)
    


def get_model(name='resnet18', embedding_dim=128, n_classes=500, att_model=False, cbam=False,
             softmax_criterion=False, att_criterion=False, pretrained=False):
    if att_model: #带attention模块的ResNet
        if cbam: #用CBAM模块的block
            model = get_resnet_cbam(name, embedding_dim, n_classes, BasicBlock_CBAM, Bottleneck_CBAM, 
                                    softmax_criterion, att_criterion, pretrained)
        else:
            model = get_resnet_cbam(name, embedding_dim, n_classes, BasicBlock, Bottleneck, 
                                    softmax_criterion, att_criterion, pretrained)
    else: #不带attention
        model = ResNet(name, embedding_dim, n_classes, softmax_criterion, pretrained)
    return model 


def load_model_checkpoint(model, checkpoint, device):
    # 预加载自己训练的模型
    model.load_state_dict(checkpoint['state_dict'], strict=False) #加载到cpu，用softmax训练后，加载除了classifier以外的层数进行微调
    model.to(device) #分配给gpu
    return model
    

def load_optimizer_checkpoint(optimizer, checkpoint, device):
    # 继续训练会要记录上次optimizer的状况
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #参数在cpu
    # param_groups会跟随model.to(device)而转换到gpu，但是state仍然在cpu
    for state in optimizer.state.values(): #取出里面的mementum_buffer
        print(state.keys())
        for k, v in state.items():
            state[k] = v.to(device) #再次分配到gpu
    return optimizer


def load_scheduler_checkpoint(lr_scheduler, checkpoint, device):
    # 继续训练会要记录上次lr_scheduler的状况
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return lr_scheduler