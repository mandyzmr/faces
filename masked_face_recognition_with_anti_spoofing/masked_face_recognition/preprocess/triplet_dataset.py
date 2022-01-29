import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from load_data import *
from config import *
from tqdm import tqdm 

# 针对三元组的数据集：Anchor, Positive, Negative, Label1, Label2
class TripletDataset(Dataset):
    def __init__(self, img_size=256, transform=None, masked=True, fusion=False, mask=False, bbox=False, n_triplets=1000): 
        if masked: #基于不同数据集
            self.data_list = TRAIN_MASKED_FACE_PATH+'.csv' 
        else:
            self.data_list = TRAIN_FACE_PATH+'.csv' 
        if fusion: #以masked为主的基础上，稍微混合不一样版本的
            self.pos_fusion, self.neg_fusion = 0.4, 0.5
        else:
            self.pos_fusion, self.neg_fusion = 0.0, 0.0 

        self.img_size = img_size
        self.transform = transform
        self.masked = masked
        self.mask = mask
        self.bbox = bbox
        
        # 如果有配好的triplets数据就直接用，要不就生成一下
        triplets_path = os.path.join(DATA_ROOT, 
                        f"{self.data_list[9:-4]}_triplets_{'fusion_' if fusion else ''}{n_triplets}.npy") #省去vggface2_(xxx).csv
        if os.path.exists(triplets_path):
            print("Loading triplets dataset...")
            self.triplets = np.load(triplets_path)
            print(f"Triplets dataset with {n_triplets} samples loaded.")
        else:
            print('Generating triplets dataset...')
            self.triplets = self.generate_triplets(triplets_path, n_triplets)

            
    def generate_triplets(self, triplets_path, n_triplets):
        data = load_train_list(os.path.join(DATA_ROOT, self.data_list))
        labels = np.unique(data["label"])
        n_ppl = len(labels) #人数
        images = {i:data[data['label']==i]['path'].to_list() for i in labels} #得到每个人的图片路径列表
        triplets = [] 
        type1, type2, type3 = 0, 0, 0 #三种类型样本的数量
        
        # 由于组合太多，只随机创建1000组samples
        for _ in tqdm(range(n_triplets)): 
            # 随机选两个类当做正类负类
            pos_class, neg_class = np.random.choice(labels, 2, replace=False)
            
            # 如果选出来的正类里的图片数少于2就重新选一个正类
            while len(images[pos_class]) < 2:
                pos_class, neg_class = np.random.choice(labels, 2, replace=False)
    
            # 创建数据集
            pos_name = data[data['label']==pos_class]['name'].iloc[0]
            neg_name = data[data['label']==neg_class]['name'].iloc[0]
            anc_idx, pos_idx = np.random.choice(len(images[pos_class]), 2, replace=False)
            neg_idx = np.random.choice(len(images[neg_class]))
            
            anc_path = images[pos_class][anc_idx]
            pos_path = images[pos_class][pos_idx]
            neg_path = images[neg_class][neg_idx]
            if np.random.rand() < self.pos_fusion: #40%概率下P换成不一样版本的，或者不换
                pos_path = pos_path.replace(os.path.splitext(self.data_list)[0], TRAIN_FACE_PATH if self.masked else TRAIN_MASKED_FACE_PATH) 
                if np.random.rand() < self.neg_fusion: #50%概率下N换成不一样版本的，或者不换
                    neg_path = neg_path.replace(os.path.splitext(self.data_list)[0], TRAIN_FACE_PATH if self.masked else TRAIN_MASKED_FACE_PATH) 
                    type3+=1 #统计不同类型的样本数量
                else:
                    type2+=1
            else:
                type1+=1
                
            triplets.append([anc_path, pos_path, neg_path, 
                             pos_class, neg_class, 
                             pos_name, neg_name])

        np.save(triplets_path, triplets)
        print(f"Triplets dataset with {n_triplets} samples generated ({type1} Type1 + {type2} Type2 + {type3} Type3) and saved to: {triplets_path}")
        return triplets
    
    def __len__(self):
        return len(self.triplets)             
          
    def get_mask(self, image_path): #由于image_path已经有可能发生变化，随机masked/unmasked
        if TRAIN_MASKED_FACE_PATH in image_path:
            mask_path = image_path.replace(TRAIN_MASKED_FACE_PATH, TRAIN_MASK_PATH)
        else:
            mask_path = image_path.replace(TRAIN_FACE_PATH, TRAIN_MASK_PATH)    
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) #(h,w)
        return mask
    
    def get_bbox(self, image_path):
        if TRAIN_MASKED_FACE_PATH in image_path:
            bbox_path = image_path.replace(TRAIN_MASKED_FACE_PATH, TRAIN_MASK_BBOX_PATH) 
        else:
            bbox_path = image_path.replace(TRAIN_FACE_PATH, TRAIN_MASK_BBOX_PATH) 
        bbox_path = os.path.splitext(bbox_path)[0]+'.txt' #从图片转为txt格式
        bbox = np.genfromtxt(bbox_path, encoding='utf-8',delimiter=',',dtype='int')
        return bbox
    
    def __getitem__(self, idx):
        anc_path, pos_path, neg_path, pos_class, neg_class, pos_name, neg_name = self.triplets[idx]
        anc_img = cv2.imread(anc_path)
        pos_img = cv2.imread(pos_path)
        neg_img = cv2.imread(neg_path)
        if self.transform:
            anc_img = self.transform(anc_img) #float32 [c,h,w]
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        
        pos_class = torch.LongTensor([int(pos_class)]) #int64
        neg_class = torch.LongTensor([int(neg_class)])
        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img,
                  'pos_class': pos_class, 'neg_class': neg_class}    
        
        if self.mask: #根据搭配的triplets，返回对应人脸非遮挡部位的mask
            anc_mask = self.get_mask(anc_path)
            pos_mask = self.get_mask(pos_path)
            neg_mask = self.get_mask(neg_path)
            sample.update({'anc_mask': anc_mask, 'pos_mask': pos_mask, 'neg_mask': neg_mask})
        
        if self.bbox: #根据搭配的triplets，返回对应人脸非遮挡部位的bbox
            anc_bbox = self.get_bbox(anc_path)
            pos_bbox = self.get_bbox(pos_path)
            neg_bbox = self.get_bbox(neg_path)
            sample.update({'anc_bbox': anc_bbox, 'pos_bbox': pos_bbox, 'neg_bbox': neg_bbox})
       
        return sample


def show_triplets(sample, version='img'): #显示图片/mask
    images = [sample[f'pos_{version}'], sample[f'anc_{version}'], sample[f'neg_{version}']]
    titles = ['Pos', 'Anc', 'Neg']
    plt.subplots(1,3)
    for i in range(3):
        plt.subplot(1,3,i+1)
        if version=='img':
            img = np.transpose(images[i].numpy(), (1,2,0)) #(h,w,c)
            plt.imshow(img[...,::-1]) #rgb
        elif version=='mask':
            plt.imshow(images[i], cmap='gray') #(h,w)
        plt.title(titles[i])
        plt.axis(False)
    plt.show()


if __name__=='__main__':
    train_dataset = TripletDataset(img_size=256, transform = train_transform, masked=False,
                             mask=False, bbox=False) #这两个设置，决定sample是否返回图片对应的mask和bbox
    train_loader  = DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE,
                           drop_last=False, num_workers=0) #用triplet训练时，因为每个batch还需要做进一步样本筛选，不需要drop last

    train_masked_dataset = TripletDataset(img_size=256, transform = train_transform, masked=True,
                             mask=True, bbox=False)
    train_masked_loader  = DataLoader(train_masked_dataset, shuffle=True, batch_size = BATCH_SIZE, 
                           drop_last=False, num_workers=0) 
    
    idx = np.random.choice(train_masked_dataset.__len__())
    sample = train_masked_dataset[idx]
    show_triplets(sample, 'img')
    show_triplets(sample, 'mask')



