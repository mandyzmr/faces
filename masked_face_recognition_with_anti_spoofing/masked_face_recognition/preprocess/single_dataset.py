import cv2
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from load_data import *
from config import *


# 针对单张图片的数据集：image, label
class SingleDataset(Dataset):
    def __init__(self, img_size=256, transform=None, masked=True, mode='train', train_size=0.8):
        if masked: #基于不同数据集
            data_list = TRAIN_MASKED_FACE_PATH+'.csv' 
        else:
            data_list = TRAIN_FACE_PATH+'.csv' 
        self.transform = transform
        self.data = load_train_list(os.path.join(DATA_ROOT, data_list))
        if mode=='train':
            self.data, _ = train_test_split(self.data, train_size=train_size, random_state=0, 
                                            shuffle=True, stratify=self.data['label'])
        elif mode=='test':
            _, self.data = train_test_split(self.data, train_size=train_size, random_state=0, 
                                            shuffle=True, stratify=self.data['label'])
        
    def __len__(self):
        return len(self.data)             
    
    def __getitem__(self, idx):
        img_path, name, label = self.data.iloc[idx]
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img) #float32 [c,h,w]
        label = torch.LongTensor([int(label)]) #int64
        return img, label


def show_single(sample): #显示图片/mask
    image = np.transpose(sample[0].numpy(), (1,2,0)) #(h,w,c)
    plt.imshow(image[...,::-1]) #rgb
    plt.title(f'Label: {sample[1].numpy()[0]}')
    plt.axis(False)
    plt.show()


if __name__=='__main__':
	train_dataset = SingleDataset(img_size=256, transform = train_transform, masked=True)
	train_loader  = DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE, 
                           		drop_last=True, num_workers=0) 

	idx = np.random.choice(train_dataset.__len__())
	sample = train_dataset[idx]
	show_single(sample)



