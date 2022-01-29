import os
import pickle
import time
import argparse
import multiprocessing
import sys
sys.path.append('./preprocess') #增加相对路径
sys.path.append('./model') 
sys.path.append('./train')
from collections import defaultdict
from preprocess.dataset import *
from model.facebagnet import *
from train.train_infer import *
from train.metrics import *
from train.cosine_anneal_lr_scheduler import *
from utils import *



def run_train(config):
    out_dir = 'running_log' #保存训练日志和模型的路径
    if config.modality=='fusion':
        config.model_name = f"FaceBagNet_{config.model}_{'att_' if config.attention else ''}{'neck_' if config.bottleneck else ''}{str(config.patch_size)}"
    else:
        config.model_name = f"{config.model}_{config.modality}_{str(config.patch_size)}"
    out_dir = os.path.join(out_dir,config.model_name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir): 
        os.makedirs(checkpoint_dir)
   
    # ------setup------
    log = Logger()
    log.open(os.path.join(out_dir, f'{config.model_name}_training.txt'), mode='a') #增加内容
    log.write(f'Training log @ {out_dir}\n')

    # ------device------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_cores = multiprocessing.cpu_count()
    log.write(f'Device: {device}\n')
    log.write('\n')

    #------dataset------
    log.write('** Dataset setting **\n')
    if device.__str__() == 'cuda':
        num_workers = 4
        pin_memory = True #使用锁页内存加速复制数据到gpu
    else:
        num_workers = cpu_cores
        pin_memory = False

    train_dataset = FDDataset(mode = 'train', modality=config.modality, patch_size=config.patch_size)
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size = config.batch_size,
                                drop_last = True, num_workers = num_workers, pin_memory=pin_memory)

    valid_dataset = FDDataset(mode = 'val', modality=config.modality, patch_size=config.patch_size)
    valid_loader  = DataLoader(valid_dataset, shuffle=False, batch_size  = config.batch_size//32,  #防止加载数据太多爆内存
                               drop_last = False, num_workers = num_workers, pin_memory=pin_memory)
    assert(len(train_dataset)>=config.batch_size)
    log.write(f'training_samples = {train_dataset.__len__()}\n')
    log.write(f'val_samples = {valid_dataset.__len__()}\n')
    log.write(f'batch_size = {config.batch_size}\n')
    log.write(f'modality = {config.modality}\n')
    log.write(f'patch_size = {config.patch_size}\n') 
    log.write('\n')
    
    #------model------
    log.write('** Model setting **\n')
    model = get_model(name=config.model, modality=config.modality, num_classes=2, 
                      attention=config.attention, bottleneck=config.bottleneck)
    log.write(f'Model: {type(model).__name__}\n')
    model.to(device) #先分配给gpu
    log.write('\n')
    
    #------train------
    log.write('** Training setting **\n')
    criterion = F.cross_entropy
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=config.initial_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = CosineAnnealingLR_with_Restart(optimizer, model, checkpoint_dir,
                T_max=config.epoch_inter, cycle_num=config.cycle_num) #先用于在固定epochs内下降lr
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch:10**(-epoch/10))
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9) #lr = gamma**epoch *lr #再进一步逐步减小lr
    log.write(f'criterion = {criterion.__name__}\n')
    log.write(f'optimizer = {type(optimizer).__name__}\n')
    log.write(f'scheduler = {type(scheduler).__name__}\n')
    log.write(f'cycle_num = {config.cycle_num}\n')
    log.write(f'epochs_per_cycle = {config.epoch_inter}\n')
    log.write('\n')

    #------pretrained------
    pretrained_path = config.pretrained_model
    if pretrained_path is not None: 
        pretrained_path = os.path.join(checkpoint_dir, pretrained_path)
        log.write(f'Loading initial_checkpoint: {pretrained_path}\n')
        model = load_model_checkpoint(model, pretrained_path, device) #加载完之后，再用DP
        # 当训练意外中断时，加载方便继续训练；若要在既有模型上，用新的lr等进行训练，不建议加载
        optimizer = load_optimizer_checkpoint(optimizer, pretrained_path, device)
        # scheduler = load_scheduler_checkpoint(scheduler, pretrained_path, device)
        log.write('\n')

    model = nn.DataParallel(model) #gpu多于1个时，并行运算

    #------log------
    log.write('** Start training here! **\n')
    pattern1="{: ^18}|{:-^16}|{:-^32}|\n" #开头第一行
    pattern2="{: ^6}"*3+"|"+"{: ^8}"*2+"|"+"{: ^8}"*4+"|"+"{: ^12}\n" #标题行
    pattern3="{: ^6}"*2+"{: ^6.4f}"+"|"+"{: ^8.4f}"*2+"|"+"{: ^8.4f}"*4+"|"+"{: ^12}\n" #内容行
    log.write(pattern1.format('',' TRAIN ',' VALID '))
    log.write(pattern2.format('cycle','epoch','lr','loss','acc','loss','acer','acc','auc','time'))
    log.write("-"*81+'\n')
    result = pd.DataFrame(columns=['model','cycle','epoch','lr','train_loss','train_acc','val_loss','val_acer','val_acc','val_auc']) #记录最优轮数的metrics
    
    history = {}
    start = time.time() #计时
    global_min_acer = 1.0 #所有周期内的最小acer
    torch.backends.cudnn.benchmark=True #cuDNN autotuner

    for i in range(config.cycle_num): #继续从上次的epoch训练
        print(f'*** Cycle {i+1} ***')
        min_acer = 1.0 #本周期内的最小acer
        val_loss, val_acer, val_acc, val_auc = 0,0,0,0 #前半周期不做validation
        metrics = defaultdict(list)
        for e in range(config.epoch_inter):        
            print(f'Epoch {e+1}/{config.epoch_inter} ', end='')
            lr = scheduler.get_last_lr()[0]
            print(f'(lr = {lr:.4f})')

            # 训练
            train_loss, train_acc = train_on_epoch(model, train_loader, criterion, optimizer, device)
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            
            # 验证
            if e >= config.epoch_inter // 2: #由于前半周期肯定不断进步，只在下半周期做validation
                val_loss, val_acer, val_acc, val_auc = val_on_epoch(model, valid_loader, criterion, device)
                metrics['val_loss'].append(val_loss)
                metrics['val_acer'].append(val_acer)
                metrics['val_acc'].append(val_acc)
                metrics['val_auc'].append(val_auc)
                end = time.time() #每个epoch结束后计算一次累计时间
                log.write(pattern3.format(i+1, e+1, lr, train_loss, train_acc, 
                          val_loss, val_acer, val_acc, val_auc, time_to_str(end - start)))
            
                # 保存周期内、全局最优模型，和周期末模型
                if val_acer < min_acer:
                    min_acer = val_acer #更新最小acer值
                    ckpt_path = os.path.join(checkpoint_dir, f'cycle{i+1:02}_min_acer_model.pth') #每个周期仅保存一个最优模型
                    torch.save({
                        'epoch':e+1,
                        'acer': min_acer,
                        'acc': val_acc,
                        'auc': val_auc, 
                        'state_dict': model.module.state_dict(), #模型参数w/b信息
                        'optimizer_state_dict': optimizer.state_dict(), #包括bn的running mean和std等信息
                        'scheduler_state_dict': scheduler.state_dict(), #包括last_epoch等用于计算后续lr的信息
                    }, ckpt_path)
                    log.write(f'Saving cycle {i+1} min acer model: {min_acer:.4f}\n')
                    result.loc[i] = [config.model_name, i+1, e+1, lr, train_loss, train_acc, val_loss, val_acer, val_acc, val_auc]

                if val_acer < global_min_acer:
                    global_min_acer = val_acer #更新最小acer值
                    ckpt_path = os.path.join(checkpoint_dir, f'global_min_acer_model.pth') #仅保存一个全局最优模型
                    torch.save({
                        'epoch':e+1,
                        'acer': global_min_acer,
                        'acc': val_acc,
                        'auc': val_auc, 
                        'state_dict': model.module.state_dict(), #模型参数w/b信息
                        'optimizer_state_dict': optimizer.state_dict(), #包括bn的running mean和std等信息
                        'scheduler_state_dict': scheduler.state_dict(), #包括last_epoch等用于计算后续lr的信息
                    }, ckpt_path)
                    log.write(f'Saving global min acer model: {global_min_acer:.4f}\n')
                    result.loc[config.cycle_num] = [config.model_name, f'Global - {i+1}', e+1, lr, train_loss, train_acc, val_loss, val_acer, val_acc, val_auc]
            else:
                end = time.time() #每个epoch结束后计算一次累计时间
                log.write(pattern3.format(i+1, e+1, lr, train_loss, train_acc, 
                          val_loss, val_acer, val_acc, val_auc, time_to_str(end - start)))

            scheduler.step() #更新学习率
        history[i+1]=metrics 
    
    # 保存每个周期，每个epoch的metrics结果，方便后续可视化查看训练情况
    pickle.dump(history, open(os.path.join(out_dir, f'{config.model_name}_history.pkl'),'wb'))
    # 保存每个周期和全局最优结果，方便后续多个模型间进行比较
    result.to_csv(os.path.join(out_dir, f'{config.model_name}_result.txt'), index=False)
    return history, result



def run_test(config):
    out_dir = 'running_log' #保存训练日志和模型的路径
    if config.modality=='fusion':
        config.model_name = f"FaceBagNet_{config.model}_{'att_' if config.attention else ''}{'neck_' if config.bottleneck else ''}{str(config.patch_size)}"
    else:
        config.model_name = f"{config.model}_{config.modality}_{str(config.patch_size)}"
    out_dir = os.path.join(out_dir,config.model_name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
    
    # ------device------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_cores = multiprocessing.cpu_count()
    
    #------dataset------
    if device.__str__() == 'cuda':
        num_workers = 16
        pin_memory = True #使用锁页内存加速复制数据到gpu
    else:
        num_workers = cpu_cores
        pin_memory = False
    
    test_dataset = FDDataset(mode = 'test', modality=config.modality, patch_size=config.patch_size)
    test_loader  = DataLoader(test_dataset, shuffle=False, batch_size = config.batch_size,
                              drop_last = False, num_workers = num_workers, pin_memory=pin_memory)
    print(f'Loading {test_dataset.__len__()} test samples...')
    
    #------model------
    model = get_model(name=config.model, modality=config.modality, num_classes=2, 
                      attention=config.attention, bottleneck=config.bottleneck)
    model.to(device) #先分配给gpu
    
    pretrained_path = config.pretrained_model
    if pretrained_path is not None: 
        pretrained_path = os.path.join(checkpoint_dir, pretrained_path)
        print(f'Loading initial_checkpoint: {pretrained_path}\n')
        model = load_model_checkpoint(model, pretrained_path, device) 
    
    model = nn.DataParallel(model) 
    
    #------log------
    print('** Start predicting here! **\n')
    model.eval()
    prob = predict(model, test_loader, device) #得到真人的概率
    save_path = os.path.join(out_dir, f'{config.model_name}_prediction.txt')
    submission(TEST_LIST, save_path, prob)


def main(config):
    if config.mode == 'train':
        run_train(config)

    if config.mode == 'infer_test':
        config.pretrained_model = 'global_min_acer_model.pth'
        run_test(config)


if __name__ == '__main__':
    # 在终端传入参数运行模型
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--modality', type=str, default='fusion', choices=['fusion','color','depth','ir'])
    parser.add_argument('--patch_size', type=int, default=48)

    # model
    parser.add_argument('--model', type=str, default='ResNet18', 
                        choices=['ResNet18', 'SENet154', 'SE-ResNet', 'SE-ResNeXt18', 'SE-ResNeXt34', 'SE-ResNeXt50'])
    parser.add_argument('--attention', type=bool, default=False, help='with SE Module to screen out multi-modal features')
    parser.add_argument('--bottleneck', type=bool, default=False, help='with SE-ResNeXt Bottleneck instead of ResNet Basic Block')

    # train
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--initial_lr', type=float, default=0.1)
    parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--epoch_inter', type=int, default=50, help='epochs per lr schedule cycle')
    parser.add_argument('--mode', type=str, default='train', choices=['train','infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None) #预训练模型路径 global_min_acer_model.pth

    config = parser.parse_args()
    print(config)
    main(config)

