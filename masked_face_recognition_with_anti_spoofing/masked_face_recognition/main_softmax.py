import os
import pickle
import time
import argparse
import sys
sys.path.append('./preprocess') #增加相对路径
sys.path.append('./model') 
sys.path.append('./train')
sys.path.append('/home/aistudio/external_library') #ai studio应用环境
from utils import *
from summary import *
from config import *
from train.train_softmax_loss import *
from train.optimizer import *


def run_train(config):
    # ------setup------
    out_dir = 'running_log' #保存训练日志和模型的路径
    model_name = f"{config.model}_{'att_' if config.attention else ''}{'cbam_' if config.cbam else ''}{'masked_' if config.masked_face else ''}{str(EMBEDDING_DIM)}"
    out_dir = os.path.join(out_dir, model_name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
    roc_dir  = os.path.join(out_dir, 'roc_plots')
    for path in [checkpoint_dir, roc_dir]:
        if not os.path.exists(path): 
            os.makedirs(path) #多层创建文件夹
   
    log = Logger()
    log.open(os.path.join(out_dir, f'{model_name}_softmax_training.txt'), mode='a') #增加内容
    log.write(f'Training log @ {out_dir}\n')
    log.write(f'Device: {device}\n')
    log.write('\n')

    #------dataset------
    log.write('** Dataset setting **\n')
    (train_dataset, train_loader), (valid_dataset, valid_loader) = get_dataset(softmax_criterion=True, 
                            att_criterion=False, masked_face=config.masked_face)
    log.write(f"Dataset: VGGFace2 {'Masked ' if config.masked_face else ''}Faces\n")
    log.write(f'training_samples = {train_dataset.__len__()}\n')
    log.write(f'val_samples = {valid_dataset.__len__()}\n')
    log.write(f'batch_size = {BATCH_SIZE}\n')
    log.write('\n')
    
    #------model------
    log.write('** Model setting **\n')
    model = get_model(name=config.model, embedding_dim=EMBEDDING_DIM, n_classes=N_CLASSES, att_model=config.attention, 
                        cbam=config.cbam, softmax_criterion=True, att_criterion=False, 
                        pretrained=PRETRAINED)
    log.write(f'Model: {type(model).__name__}\n')
    log.write(f"pretrained: {'True' if PRETRAINED else 'False'}\n")
    model.to(device) #先分配给gpu
    log.write('\n')
    
    #------train------
    log.write('** Training setting **\n')
    optimizer = get_optimizer(model, name=OPTIMIZER)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    log.write(f'criterion = arcface softmax loss\n')
    log.write(f'optimizer = {type(optimizer).__name__}\n')
    log.write(f'epochs = {config.epochs}\n')
    log.write('\n')

    #------pretrained------
    epoch_start = 0
    max_acc = 0.0
    pretrained_path = config.pretrained_model
    if pretrained_path is not None: 
        pretrained_path = os.path.join(checkpoint_dir, pretrained_path)
        log.write(f'Loading initial_checkpoint: {pretrained_path}\n')
        checkpoint = torch.load(pretrained_path, map_location=device) #用gpu加载state_dict
        model = load_model_checkpoint(model, checkpoint, device) #加载完之后，再用DP
        # 当训练意外中断时，加载方便继续训练，但是如果optimizer更换了或者scheduler更换了，则不需要加载
        # optimizer = load_optimizer_checkpoint(optimizer, checkpoint, device)
        # lr_scheduler = load_scheduler_checkpoint(lr_scheduler, checkpoint, device)
        epoch_start = checkpoint['epoch']
        max_acc = checkpoint['acc']
        log.write('\n')

    model = nn.DataParallel(model) #gpu多于1个时，并行运算

    #------log------
    log.write('** Start training here! **\n')
    pattern1="{: ^12}|{:-^14}|{:-^36}|\n" #开头第一行
    pattern2="{: ^6}"*2+"|"+"{: ^7}"*2+"|"+"{: ^9}"*4+"|"+"{: ^12}\n" #标题行
    pattern3="{: ^6}"+"{: ^6.0e}"+"|"+"{: ^7.3f}"*2+"|"+"{: ^9.3f}"*4+"|"+"{: ^12}\n" #内容行
    log.write(pattern1.format('',' TRAIN ',' VALID '))
    log.write(pattern2.format('epoch','lr','loss','acc','loss','acc','precision', 'recall', 'time'))
    log.write("-"*74+'\n')
    
    history = defaultdict(list)
    val_loss, val_acc, val_precision, val_recall = 0,0,0,0 #前半周期不做validation
    start = time.time() #计时
    for e in range(epoch_start, epoch_start+config.epochs): #继续从上次的epoch训练 
        print(f'Epoch {e+1}/{epoch_start+config.epochs}')
        # 根据epoch先调整lr
        lr = adjust_learning_rate(optimizer, e)
        # lr = lr_scheduler.get_last_lr()[0]

        # 训练
        train_loss, train_acc = train_on_epoch(model, train_loader, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证
        if valid_loader: 
            val_loss, val_acc, val_precision, val_recall = val_on_epoch(model, valid_loader, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)

            end = time.time() #每个epoch结束后计算一次累计时间
            log.write(pattern3.format(e+1, lr, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, time_to_str(end - start)))
            
            if val_acc > max_acc:
                max_acc = val_acc #更新最大acc值
                ckpt_path = f"global_max_acc_model_{e+1}.pth" #所有epoch模型
                torch.save({
                    'epoch':e+1, 
                    'acc': val_acc,
                    'model':type(model.module).__name__,
                    'state_dict': model.module.state_dict(), #模型参数w/b信息
                    'optimizer_state_dict': optimizer.state_dict(), #包括bn的running mean和std等信息
                    # 'scheduler_state_dict': lr_scheduler.state_dict(),
                }, os.path.join(checkpoint_dir, ckpt_path))
                log.write(f'Saving epoch {e+1} max acc model: {max_acc:.4f}\n')
                run_validate(config, ckpt_path, e+1)
        else:
            end = time.time() #每个epoch结束后计算一次累计时间
            log.write(pattern3.format(e+1, lr, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, time_to_str(end - start)))
        
        # 更新lr
        # lr_scheduler.step()

    # 保存每个epoch的metrics结果，方便后续可视化查看训练情况
    pickle.dump(history, open(os.path.join(out_dir, f'{model_name}_softmax_history.pkl'),'wb'))


def run_validate(config, pretrained_path, e=-1): #在LFW上测试，为了便于同时使用在训练过程中，不从config获取
    out_dir = 'running_log' #保存训练日志和模型的路径
    model_name = f"{config.model}_{'att_' if config.attention else ''}{'cbam_' if config.cbam else ''}{'masked_' if config.masked_face else ''}{str(EMBEDDING_DIM)}"
    out_dir = os.path.join(out_dir, model_name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
    roc_dir  = os.path.join(out_dir, 'roc_plots')
    
    #------dataset------
    (_, _), (valid_dataset, valid_loader), (valid_masked_dataset, valid_masked_loader) = get_dataset(softmax_criterion=False, 
                        att_criterion=False, masked_face=config.masked_face, fusion_face=False) #只需要取valid set
    
    #------model------
    model = get_model(name=config.model, embedding_dim=EMBEDDING_DIM, n_classes=N_CLASSES, att_model=config.attention, 
                        cbam=config.cbam, softmax_criterion=False, att_criterion=False, pretrained=False) #直接加载自己训练的
    model.to(device) #先分配给gpu
    if pretrained_path is not None: 
        pretrained_path = os.path.join(checkpoint_dir, pretrained_path)
        print(f'Loading initial_checkpoint: {pretrained_path}\n')
        checkpoint = torch.load(pretrained_path, map_location=device) #用gpu加载state_dict
        model = load_model_checkpoint(model, checkpoint, device) #加载完之后，再用DP

    model = nn.DataParallel(model) 

    print('Validating on unmasked LFW ...')
    val_auc, val_acc, val_recall = val_on_lfw(model, valid_loader, device, e, roc_dir, masked=False)
    print('Validating on masked LFW ...')
    val_masked_auc, val_masked_acc, val_masked_recall = val_on_lfw(model, valid_masked_loader, device, e, roc_dir, masked=True)


def run_test(config): #预测emb
    out_dir = 'running_log' #保存训练日志和模型的路径
    model_name = f"{config.model}_{'att_' if config.attention else ''}{'cbam_' if config.cbam else ''}{str(EMBEDDING_DIM)}"
    out_dir = os.path.join(out_dir, model_name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
    
    model = get_model(name=config.model, embedding_dim=EMBEDDING_DIM, n_classes=N_CLASSES, att_model=config.attention, 
                        cbam=config.cbam, softmax_criterion=False, att_criterion=False, pretrained=False) #直接加载自己训练的
    model.to(device) #先分配给gpu
    
    pretrained_path = config.pretrained_model
    if pretrained_path is not None: 
        pretrained_path = os.path.join(checkpoint_dir, pretrained_path)
        print(f'Loading initial_checkpoint: {pretrained_path}\n')
        model = load_model_checkpoint(model, pretrained_path, device) 
    model = nn.DataParallel(model) 
    
    image = cv2.imread(config.image_path)
    emb = predict(model, image, device) #得到真人的概率
    return emb


def main(config):
    if config.mode == 'train':
        run_train(config)
    elif config.mode == 'validate':
        return run_validate(config, config.pretrained_model,2)
    elif config.mode == 'test':
        return run_test(config)


if __name__ == '__main__':
    # 在终端传入参数运行模型
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--masked_face', type=bool, default=False)
    parser.add_argument('--image_path', type=str, default=None, help='Image local path for testing')

    # model
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18','resnet34','resnet50','resnet101','resnet152'])
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--cbam', type=bool, default=False)

    # train
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--mode', type=str, default='train', choices=['train','validate','test'])
    parser.add_argument('--pretrained_model', type=str, default=None) #预训练模型路径 global_min_acer_model.pth

    config = parser.parse_args()
    print(config)
    main(config)

