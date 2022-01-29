import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from casia_surf_dataset.load_data import *
from augmentation import *


class FDDataset(Dataset):
    def __init__(self, mode='train', modality='fusion', patch_size=32, balance = True):
        super(FDDataset, self).__init__()
        self.mode = mode #train/val/test
        self.modality = modality #color/depth/ir/fusion
        self.balance = balance
        self.patch_size = patch_size 
        self.set_mode(self.mode)

    def set_mode(self, mode):
        self.mode = mode
        path_list = {'train':TRAIN_LIST,
                     'val': VAL_LIST,
                     'test': TEST_LIST}
        
        print(f'Loading {self.mode} dataset...')
        self.data_list = load_data_list(os.path.join(DATA_ROOT, path_list[self.mode])) #用pandas导入数据
        self.num_data = len(self.data_list)

    def preprocess(self, image_path):
        image = cv2.imread(os.path.join(DATA_ROOT, image_path))
        image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
        if self.mode == 'train': #根据mode，得到单个/多个patches
            is_infer=False #(h,w,c)
        elif self.mode in ['val','test']:
            is_infer=True #(n,h,w,c)
        image = augumentor(image, target_shape=(self.patch_size, self.patch_size, 3), 
                               is_infer=is_infer, n_patches=36)
        return image    
    
    def __getitem__(self, idx):
        # 取出样本
        if self.balance: #如果要均衡样本，证明是train
            label = np.random.choice([0,1], p=[0.5,0.5]) #pos和neg相同采样比例
            data = self.data_list[self.data_list['label']==label] #选择pos/neg
            idx = np.random.choice(len(data)) #随机选取样本
        else: #否则，取出指定idx的样本
            data = self.data_list #候选数据
        color_path, depth_path, ir_path, label = data.iloc[idx] #样本路径和label
         
        
        # 处理标签
        if self.mode in ['train','val']:
            label = torch.LongTensor([int(label)]) #数字int64
        elif self.mode == 'test': #为了统一格式，设置test的label为Nan方便导入
            label = color_path+' '+depth_path+' '+ir_path #字符串
            
        
        # 根据模态和mode，得到统一尺寸的单模态图/多模态图，以及单个/多个patches
        if self.modality=='fusion':
            color = self.preprocess(color_path) #(h,w,3)/(n,h,w,3)
            depth = self.preprocess(depth_path)
            ir = self.preprocess(ir_path) 
            image = np.concatenate([color, depth, ir], axis=-1) #(h,w,9)/(n,h,w,9)
        elif self.modality=='color':
            image = self.preprocess(color_path) #(h,w,3)/(n,h,w,3)
        elif self.modality=='depth':
            image = self.preprocess(depth_path)
        elif self.modality=='ir':
            image = self.preprocess(ir_path)
          
        
        # 根据mode，得到单个/多个patches
        if self.mode == 'train':
            if self.modality=='fusion': # MFE对1个随机模态dropout归零
                if np.random.randint(0,2) == 0: #50%概率不会dropout模态
                    mfe_idx = np.random.choice(3) #从3个模态种随机选择1个
                    if np.random.randint(0,2) == 0: #25%概率会dropout 1个模态
                        image[:,:,3*mfe_idx:3*(mfe_idx+1)] = 0
                    else:
                        for i in range(3):
                            if i != mfe_idx: #25%概率会dropout 2个模态
                                image[:,:,3*mfe_idx:3*(mfe_idx+1)] = 0
            image = np.transpose(image, (2, 0, 1)) #(9,h,w)/(3,h,w)            
                            
        elif self.mode in ['val','test']:
            image = np.transpose(image, (0, 3, 1, 2)) #(n,9,h,w)/(n,3,h,w)

        
        # 归一化输出patches和label
        image = image.astype(np.float32)
        image = image / 255.0
        image = torch.FloatTensor(image) #返回float32
        return image, label 

    
    def __len__(self):
        return self.num_data