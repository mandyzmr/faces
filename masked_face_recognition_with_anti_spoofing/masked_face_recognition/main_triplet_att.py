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
from train.train_triplet_att_loss import *
from train.optimizer import *


def run_train(config):
    # ------setup------
    out_dir = 'running_log' #保存训练日志和模型的路径
    model_name = f"{config.model}_att_{'cbam_' if config.cbam else ''}{'masked_' if config.masked_face else ''}{str(EMBEDDING_DIM)}"
    out_dir = os.path.join(out_dir, model_name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
    heatmap_dir  = os.path.join(out_dir, 'heatmaps')
    roc_dir  = os.path.join(out_dir, 'roc_plots')
    for path in [checkpoint_dir, heatmap_dir, roc_dir]:
        if not os.path.exists(path): 
            os.makedirs(path) #多层创建文件夹
   
    log = Logger()
    log.open(os.path.join(out_dir, f'{model_name}_triplet_att_training.txt'), mode='a') #增加内容
    log.write(f'Training log @ {out_dir}\n')
    log.write(f'Device: {device}\n')
    log.write('\n')

    #------dataset------
    log.write('** Dataset setting **\n')
    (train_dataset, train_loader), (valid_dataset, valid_loader), (valid_masked_dataset, valid_masked_loader) = get_dataset(softmax_criterion=False, 
                        att_criterion=True, masked_face=config.masked_face, fusion_face=config.fusion_face)
    log.write(f"Train Dataset: VGGFace2 {'Masked ' if config.masked_face else ''}{'Fusion ' if config.fusion_face else ''}Faces with corresponding masks\n")
    log.write(f'Val Dataset: LFW Masked and Unmasked Faces\n')
    log.write(f'training_samples = {train_dataset.__len__()}\n')
    log.write(f'val_unmasked_samples = {valid_dataset.__len__()}\n')
    log.write(f'val_masked_samples = {valid_masked_dataset.__len__()}\n')
    log.write(f'batch_size = {BATCH_SIZE}\n')
    log.write('\n')
    
    #------model------
    log.write('** Model setting **\n')
    model = get_model(name=config.model, embedding_dim=EMBEDDING_DIM, n_classes=N_CLASSES, att_model=True, 
                        cbam=config.cbam, softmax_criterion=False, att_criterion=True, 
                        pretrained=PRETRAINED)
    log.write(f'Model: {type(model).__name__}\n')
    log.write(f"Pretrained: {'True' if PRETRAINED else 'False'}\n")
    log.write('\n')
    
    #------train------
    log.write('** Training setting **\n')
    optimizer = get_optimizer(model, name=OPTIMIZER)
    log.write(f'criterion = triplet loss and attention loss\n')
    log.write(f'optimizer = {type(optimizer).__name__}\n')
    log.write(f'epochs = {config.epochs}\n')
    log.write('\n')

    #------pretrained------
    epoch_start = 0
    max_auc = 0.0
    pretrained_path = config.pretrained_model
    if pretrained_path is not None: 
        pretrained_path = os.path.join(checkpoint_dir, pretrained_path)
        log.write(f'Loading initial_checkpoint: {pretrained_path}\n')
        checkpoint = torch.load(pretrained_path, map_location=device)
        model = load_model_checkpoint(model, checkpoint, device) #加载完之后，再用DP
        # 当训练意外中断时，加载方便继续训练
        # optimizer = load_optimizer_checkpoint(optimizer, checkpoint, device)
        # epoch_start = checkpoint['epoch']
        # max_auc = checkpoint['auc']
        log.write('\n')
    else:
        model.to(device) #先分配给gpu
    model = nn.DataParallel(model) #gpu多于1个时，并行运算

    #------log------
    log.write('** Start training here! **\n')
    pattern1="{: ^12}|{:-^45}|{:-^60}|\n" #开头第一行
    pattern2="{: ^6}"*2+"|"+"{: ^13}"*3+"|"+"{: ^7}{: ^13}"*2+"{: ^13}"*2+"|"+"{: ^12}\n" #标题行
    pattern3="{: ^6}"+"{: ^6.4f}"+"|"+"{: ^13}"+"{: ^13.3f}"*2+"|"+"{: ^7.3f}{: ^13.3f}"*2+"{: ^13.3f}"*2+"|"+"{: ^12}\n" #内容行
    log.write(pattern1.format('',' TRAIN ',' VALID '))
    log.write(pattern2.format('epoch','lr','hard_samples','triplet_loss','att_loss', 'auc','masked_auc','acc','masked_acc',
                                'tar@far=1e-2','masked_tar','time'))
    log.write("-"*113+'\n') 
    
    history = defaultdict(list)
    val_auc, val_masked_auc, val_acc, val_masked_acc, val_tar, val_masked_tar = 0,0,0,0,0,0 #前半周期不做validation
    start = time.time() #计时
    for e in range(epoch_start, epoch_start+config.epochs): #继续从上次的epoch训练      
        print(f'Epoch {e+1}/{epoch_start+config.epochs}')
        # 根据epoch调整lr
        lr = adjust_learning_rate(optimizer, e)
        
        # 训练
        train_triplet_loss, train_att_loss, n_hard = train_on_epoch(model, train_loader, optimizer, device)
        history['train_att_loss'].append(train_att_loss)
        history['train_triplet_loss'].append(train_triplet_loss)

        # 同时在戴口罩和不戴口罩的LFW上验证
        if valid_loader and valid_masked_loader and e >= config.epochs //2: #由于前半周期肯定不断进步，只在下半周期做validation
            val_auc, val_acc, val_tar = val_on_epoch(model, valid_loader, device, e, roc_dir, heatmap_dir, masked=False)
            history['val_auc'].append(val_auc)
            history['val_acc'].append(val_acc)
            history['val_tar'].append(val_tar)
            
            val_masked_auc, val_masked_acc, val_masked_tar = val_on_epoch(model, valid_masked_loader, device, e, roc_dir, heatmap_dir, masked=True)
            history['val_masked_auc'].append(val_masked_auc)
            history['val_masked_acc'].append(val_masked_acc)
            history['val_masked_tar'].append(val_masked_tar)
            end = time.time() #每个epoch结束后计算一次累计时间
            log.write(pattern3.format(e+1, lr, n_hard, train_triplet_loss, train_att_loss, val_auc, val_masked_auc, 
                                      val_acc, val_masked_acc, val_tar, val_masked_tar, time_to_str(end - start)))

            # 遮挡状态下的人脸识别系统，是为了能准确识别不戴口罩的人脸的前提下，提高识别戴口罩人脸的准确率
            # 因此需要平衡二者的表现，取二者均值auc为保存模型的标准
            mean_auc = np.mean([val_auc, val_masked_auc]) 
            if mean_auc > max_auc:
                max_auc = mean_auc #更新最大auc值
                ckpt_path = os.path.join(checkpoint_dir, f'global_max_auc_model.pth') #仅保存一个最优模型
                torch.save({
                    'epoch':e+1, 
                    'auc':max_auc,
                    'model':type(model.module).__name__,
                    'state_dict': model.module.state_dict(), #模型参数w/b信息
                    'optimizer_state_dict': optimizer.state_dict(), #包括bn的running mean和std等信息
                }, ckpt_path)
                log.write(f'Saving epoch {e+1} max auc model: {mean_auc:.4f}\n')
        else:
            end = time.time() #每个epoch结束后计算一次累计时间
            log.write(pattern3.format(e+1, lr, n_hard, train_triplet_loss, train_att_loss, val_auc, val_masked_auc, 
                                      val_acc, val_masked_acc, val_tar, val_masked_tar, time_to_str(end - start)))

    # 保存每个epoch的metrics结果，方便后续可视化查看训练情况
    pickle.dump(history, open(os.path.join(out_dir, f'{model_name}_triplet_att_history.pkl'),'wb'))
    generate_heatmap_gif(heatmap_dir)
    generate_rou_gif(rou_dir)
    

def run_test(config):
    out_dir = 'running_log' #保存训练日志和模型的路径
    model_name = f"{config.model}_att_{'cbam_' if config.cbam else ''}{'masked_' if config.masked_face else ''}{str(EMBEDDING_DIM)}"
    out_dir = os.path.join(out_dir, model_name)
    checkpoint_dir = os.path.join(out_dir, 'checkpoint')
    
    model = get_model(name=config.model, embedding_dim=EMBEDDING_DIM, n_classes=N_CLASSES, att_model=True, 
                        cbam=config.cbam, softmax_criterion=False, att_criterion=True, 
                        pretrained=PRETRAINED)
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

    if config.mode == 'infer':
        return run_test(config)


if __name__ == '__main__':
    # 在终端传入参数运行模型
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--masked_face', type=bool, default=False)
    parser.add_argument('--fusion_face', type=bool, default=False)
    parser.add_argument('--image_path', type=str, default=None, help='Image local path for testing')

    # model
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18','resnet34','resnet50','resnet101','resnet152'])
    parser.add_argument('--cbam', type=bool, default=False)

    # train
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--mode', type=str, default='train', choices=['train','infer'])
    parser.add_argument('--pretrained_model', type=str, default=None) #预训练模型路径 global_min_acer_model.pth

    config = parser.parse_args()
    print(config)
    main(config)

