import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = 'casia_surf_dataset' #在数据文件夹下
TRAIN_LIST = 'train_list.txt' 
VAL_LIST = 'val_private_list.txt'
TEST_LIST = 'test_private_list.txt'


def load_data_list(path): #加载数据
    data = pd.read_csv(path, sep=' ', names=['color','depth','ir','label'])
    n_pos = len(data[data['label']==1])
    n_neg = len(data[data['label']==0])
    print(f'We have loaded {data.shape[0]} samples: {n_pos} real faces, and {n_neg} fake faces.')
    return data


def show_sample(idx, data): #查看样本的多模态图
    samples = data.iloc[idx]
    titles = data.columns
    plt.subplots(1,3)
    for i in range(3):
        plt.subplot(1,3,i+1)
        img = plt.imread(samples[i])
        plt.imshow(img)
        plt.title(f'{titles[i]}: {img.shape}')
        plt.axis(False)
    plt.show()


if __name__ == '__main__':
	train = load_data_list(TRAIN_LIST)
	print(train.head()) #查看数据
	n_train = train.shape[0]
	idx = np.random.choice(n_train) #随机抽取样本
	show_sample(idx, train)
