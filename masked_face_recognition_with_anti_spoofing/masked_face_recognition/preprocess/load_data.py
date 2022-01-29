import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import *


# 根据VGGFace2数据创建data list，包括图片路径，以及对应人name和label
def create_train_list(path):
    data = []
    ppl_list = os.listdir(path) # 得到以人名命名的folder列表
    ppl_list.sort()
    for name_id, name in enumerate(ppl_list):
        name_path = os.path.join(path, name) #由于mac可能有.DS_Store等隐藏文件
        if os.path.isdir(name_path): #跳过文件
            image_list = os.listdir(name_path)
            image_list.sort()
            for image in image_list: #每个人的图片路径
                data.append([os.path.join(path, name, image), name, name_id])

    data = pd.DataFrame(data, columns = ['path','name','label'])
    data.to_csv(f'{path}.csv', index=False)
    return data


def load_train_list(path):  
    data = pd.read_csv(path, sep=',')
    print(f'We have loaded {data.shape[0]:,} samples belonging to {len(np.unique(data["name"]))} people.\n')
    return data


# 加载LFW列表，包含两种pairs：同一个人的2张图片id，或者不同人各自的图片id
def load_test_list(path):  
    data = pd.read_csv(path, delimiter='\t', skiprows=1, names=['name1','id1-1','id1-2/name2','id2'])
    same_pairs = data[data['id2'].isnull()].reset_index(drop=True) #
    diff_pairs = data[data['id2'].notnull()].reset_index(drop=True) #
    diff_pairs['id2']=diff_pairs['id2'].astype('int') #由于和nan同列，变为float，需要转回int
    print(f'We have loaded {same_pairs.shape[0]} pairs belonging to the same person, and {diff_pairs.shape[0]} pairs that are not.')
    return same_pairs, diff_pairs


if __name__=='__main__':
    # 查看vggface
    train = create_train_list(os.path.join(DATA_ROOT, TRAIN_PATH))
    train = load_train_list(os.path.join(DATA_ROOT,TRAIN_LIST))
    print(train)

    # idx = np.random.choice(len(train))
    # plt.imshow(plt.imread(train['path'][idx]))
    # plt.title(train['name'][idx])
    # plt.show()

    # # 查看lfw
    # same, diff = load_test_list(os.path.join(DATA_ROOT,VAL_LIST))
    # print(same.head())
    # print(diff.head())

    # idx = np.random.choice(len(same))
    # name = same['name1'][idx]
    # file = f'{name}_{int(same["id1-1"][idx]):04}.jpg'
    # plt.imshow(plt.imread(os.path.join(DATA_ROOT, VAL_PATH, name, file)))
    # plt.title(same['name1'][idx])
    # plt.show()



    