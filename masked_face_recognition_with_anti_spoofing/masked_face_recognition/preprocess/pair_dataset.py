import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from load_data import *
from config import *
from tqdm import tqdm


# 针对二元组的数据集：Image1, Image2, IsSame
class PairDataset(Dataset):
    def __init__(self, data_list='pairs.txt', img_size=256, transform=None, masked=False):
        self.data_list = data_list #基于不同数据集
        self.img_size = img_size
        self.transform = transform

        # 初始化模型
        self.detector = detector # 人脸检测模型
        self.predictor = predictor #68个关键点检测模型
         
        # 如果有配好的LFW数据就直接用，要不就生成一下
        lfw_path=os.path.join(DATA_ROOT, 
                 f"lfw_funneled_{'masked_' if masked else ''}pairs.npy")
        if os.path.exists(lfw_path):
            print("Loading LFW dataset...")
            self.pairs = np.load(lfw_path)
            print(f"LFW dataset with {self.pairs.shape[0]} pairs loaded.")
        else:
            print('Generating LFW dataset...')
            self.pairs = self.generate_pairs(lfw_path)

            
    def add_extension(self, path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            return None #如果不是jpg/png，跳过该pair
            
    
    def generate_pairs(self, lfw_path):
        same_pairs, diff_pairs = load_test_list(os.path.join(DATA_ROOT, self.data_list))
        pairs = []
        skip_num = 0 #记录路径不存在或检测到的人脸非1个的pairs数
        directory = os.path.splitext(lfw_path)[0].replace('_pairs','') #根据lfw_path判断是戴/不戴口罩的pairs
        for i in tqdm(range(same_pairs.shape[0])):
            name, id1, id2, _ = same_pairs.iloc[i]
            path1 = self.add_extension(os.path.join(directory, name, f'{name}_{int(id1):04}'))
            path2 = self.add_extension(os.path.join(directory, name, f'{name}_{int(id2):04}'))
            label = 1 #相同人
            if path1 and path2: #只有当路径存在
                img1 = self.get_face_align(path1) #检测人脸
                img2 = self.get_face_align(path2)
                if (img1 is not None) and (img2 is not None): #只有当检测到人脸的情况，才加入到数据集
                    pairs.append([path1, path2, label])
                    # cv2.imwrite(path1, img1) #或者直接把原图替换为正脸截图
                    # cv2.imwrite(path2, img2)
                else:
                    skip_num+=1
            else:
                skip_num+=1
                
        for i in tqdm(range(diff_pairs.shape[0])):
            name1, id1, name2, id2 = diff_pairs.iloc[i]
            path1 = self.add_extension(os.path.join(directory, name1, f'{name1}_{int(id1):04}'))
            path2 = self.add_extension(os.path.join(directory, name2, f'{name2}_{int(id2):04}'))
            label = 0 #不同人
            if path1 and path2:
                img1 = self.get_face_align(path1) #检测人脸
                img2 = self.get_face_align(path2)
                if (img1 is not None) and (img2 is not None):
                    pairs.append([path1, path2, label])
                    # cv2.imwrite(path1, img1) #或者直接把原图替换为正脸截图
                    # cv2.imwrite(path2, img2)
                else:
                    skip_num+=1 
            else:
                skip_num+=1
        
        if skip_num>0:
            print(f'Skipped {skip_num} pairs')
        pairs = np.array(pairs)
        np.random.shuffle(pairs) #因为前半部分都是1，后半都是0，需要先打乱顺序
        np.save(lfw_path, pairs)
        print(f"LFW dataset with {len(pairs)} pairs generated and saved to: {lfw_path}")
        return pairs 
    
    
    def __len__(self):
        return len(self.pairs)             
          
        
    def get_face_align(self, image_path):
        image = cv2.imread(image_path) #BGR
        face_align=None #原始值
    
        # 人脸检测，返回bbox坐标
        bboxes = self.detector(image, upsample_num_times=1) #返回bbox坐标[[(x1,y1) (x2,y2)],[...]]
        if len(bboxes) == 1:    
            # 对检测到的人脸提取人脸关键点
            landmarks = self.predictor(image, bboxes[0]) #若有多组landmarks的话，用dlib.full_objection_detections().append(landmarks)
            face_align = dlib.get_face_chip(image, face=landmarks, #关键点对齐人脸，旋转恢复水平正面
                                            size=self.img_size) #resize得到放大居中的人脸框截图
        return face_align
        
            
    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = self.get_face_align(path1) #如果pairs里的路径是原图，仍然要做对齐检测，如果已做也会返回即有结果，
        img2 = self.get_face_align(path2)
        # img1 = cv2.imread(path1)
        # img2 = cv2.imread(path2)
        
        if self.transform:
            img1 = self.transform(img1) #float32 [c,h,w]
            img2 = self.transform(img2)
            
        label = torch.LongTensor([int(label)]) #int64
        return img1, img2, label


def show_pairs(sample):
    img1, img2, label = sample
    images = [img1, img2]
    plt.subplots(1,2, figsize=(5,2))
    for i in range(2):
        plt.subplot(1,2,i+1)
        img = np.transpose(images[i].numpy(), (1,2,0)) #(h,w,c)
        plt.imshow(img[...,::-1])
        plt.axis(False)
    plt.suptitle(f'Same person: {"True" if label else "False"}')
    plt.show()


if __name__=='__main__':
    valid_dataset = PairDataset(img_size=256, transform = test_transform, masked=False)
    valid_loader  = DataLoader(valid_dataset, shuffle=False, batch_size = BATCH_SIZE, drop_last=False, 
                      num_workers=0) 

    valid_masked_dataset = PairDataset(img_size=256, transform = test_transform, masked=True)
    valid_masked_loader  = DataLoader(valid_masked_dataset, shuffle=False, batch_size = BATCH_SIZE, drop_last=False, 
                      num_workers=0) 
    
    idx = np.random.choice(valid_dataset.__len__())
    sample = valid_dataset[idx]
    show_pairs(sample)



