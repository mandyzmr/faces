import os
import sys
#由于创建数据的脚本create_masked_dataset.sh在上一级路径，通过添加上一级路径import config
#但是sys.path.append('..')不管用，所以直接添加绝对路径
sys.path.append(os.getcwd()) 
sys.path.append('/home/aistudio/external_library') #ai studio应用环境
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import dlib
import copy
from load_data import *
from config import *


# 创建戴口罩数据集
def create_masked_dataset(path, start_p=0.0, end_p=1.0, random_p=None, unmasked=False, masked=False, mask=False,
                          face_path=TRAIN_FACE_PATH, masked_face_path=TRAIN_MASKED_FACE_PATH, #可以改成val版本的
                          mask_path=TRAIN_MASK_PATH, mask_bbox_path=TRAIN_MASK_BBOX_PATH):
    # 创建数据集方式
    data = []
    ppl_list = os.listdir(path) # 得到以人名命名的folder列表
    ppl_list.sort()
    n_ppl = len(ppl_list) #原数据集总人数
    
    # Method 1: 如果需要全部转换，由于速度会太慢，可以通过shell后台运行脚本，同步处理多组数据
    if (start_p is not None) and (end_p is not None):
        ppl_list = ppl_list[int(start_p*n_ppl):int(end_p*n_ppl)]
        
    # Method 2: 如果只需要抽取其中一部分做成小数据集：
    elif random_p is not None:
        ppl_list = np.random.choice(ppl_list, int(random_p*n_ppl), replace=False)
    ppl_list.sort() #随机抽取组成新数据集的人
    skip_num = 0 #记录有多少图片因为检测不到/超过1个以上人脸，而被跳过
    
    # 基于原数据集，遍历每个人的每张照片，进行改造
    print(f'Creating new datasets for {len(ppl_list)} people from {ppl_list[0]} to {ppl_list[-1]}...')
    for name in tqdm(ppl_list):
        name_path = os.path.join(path, name)
        if os.path.isdir(name_path): #是文件夹的情况下
            image_list = os.listdir(name_path)
            image_list.sort()
            if unmasked: #要创建不戴口罩版本
                if not os.path.exists(os.path.join(DATA_ROOT, face_path, name)):
                    os.makedirs(os.path.join(DATA_ROOT, face_path, name))
            if masked: #要创建戴口罩版本
                if not os.path.exists(os.path.join(DATA_ROOT, masked_face_path, name)):
                    os.makedirs(os.path.join(DATA_ROOT, masked_face_path, name))
            if mask: #要创建戴口罩对应的mask (validation不需要)
                if not os.path.exists(os.path.join(DATA_ROOT, mask_path, name)):
                    os.makedirs(os.path.join(DATA_ROOT, mask_path, name))
                if not os.path.exists(os.path.join(DATA_ROOT, mask_bbox_path, name)):
                    os.makedirs(os.path.join(DATA_ROOT, mask_bbox_path, name))
        
            for image in image_list: #每个人的图片路径
                image_path = os.path.join(path, name, image)
                image_face_path = os.path.join(DATA_ROOT, face_path, name, image)
                image_masked_face_path = os.path.join(DATA_ROOT, masked_face_path, name, image)
                image_mask_path = os.path.join(DATA_ROOT, mask_path, name, image)
                image_mask_bbox_path = os.path.join(DATA_ROOT, mask_bbox_path, name, os.path.splitext(image)[0]+'.txt')
                skip_num = create_masked_face(image_path, image_face_path, image_masked_face_path, image_mask_path, image_mask_bbox_path,
                                   unmasked, masked, mask, skip_num)
    print(f'Datasets generated with {skip_num} images skipped!')


# 创建戴口罩图片
def create_masked_face(image_path, image_face_path, image_masked_face_path, image_mask_path, image_mask_bbox_path,
                       unmasked=False, masked=False, mask=False, skip_num=0):
    image = cv2.imread(image_path) #BGR
    face_align, mask_align = None, None #设置初始值
    if image is None:
        print(image_path)
        skip_num += 1

    else:
        # 人脸检测，返回bbox坐标
        bboxes = detector(image, upsample_num_times=1) #返回bbox坐标[[(x1,y1) (x2,y2)],[...]]
        if len(bboxes) == 1:  #若只检测到1个人脸，由于检测到多张脸时，情况会很复杂，未免混乱都跳过         
            # 对原图检测到的人脸提取人脸关键点
            landmarks = predictor(image, bboxes[0]) #若有多组landmarks的话，用dlib.full_objection_detections().append(landmarks)
            face_align = dlib.get_face_chip(image, face=landmarks, #关键点对齐人脸，旋转恢复水平正面
                                            size=IMAGE_SIZE) #resize得到放大居中的人脸框截图
            face_align_unmasked = copy.deepcopy(face_align) #备份一个不被口罩遮挡的对齐截图

            if masked: #如果创建戴口罩数据
                bboxes = detector(face_align, upsample_num_times=1) #对水平的正脸截图重新提取人脸关键点
                if len(bboxes) == 1:  
                    # 关键点位置：脸型：0-16，左眉：17-21，右眉：22-26，鼻梁：27-30，鼻子：31-35
                    #           左眼：顺时针36-41，右眼：顺时针42-47，嘴巴：48-68
                    landmarks = predictor(face_align, bboxes[0]) 

                    # 下半脸：创建多边形随机颜色口罩
                    bottom = []
                    for i in range(2, 15, 1): # 耳下脸部关键点
                        bottom.append([landmarks.part(i).x, landmarks.part(i).y])
                    bottom = np.array(bottom)
                    colors = [(200, 183, 144), (163, 150, 134), (172, 170, 169), \
                            (167, 168, 166), (173, 171, 170), (161, 161, 160), \
                            (170, 162, 162)] #不同颜色的口罩
                    color = colors[np.random.randint(0,len(colors),[])]
                    cv2.fillConvexPoly(face_align, bottom, color) 
                    
                    if mask: #如果需要返回对应的mask和bbox
                        # 求眼角和眉毛之间的平均距离，即眼睑的高度
                        eyebrow_list = [18,20,23,25] #对齐眼角的眉毛位置
                        eyes_list = [36,39,42,45] #眼睛的左右眼角
                        eyebrow = 0
                        eyes = 0
                        for eb, ey in zip(eyebrow_list, eyes_list):
                            eyebrow += landmarks.part(eb).y
                            eyes += landmarks.part(ey).y
                        extras = int(eyes/4 - eyebrow/4) 

                        # 上半脸：创建mask，用白色凸包convex hull标识没有被遮盖的上半脸
                        upper = []
                        for i in [0,1,2,14,15,16,17,18,19,20,23,24,25,26]: #耳侧脸部和眉毛关键点
                            if i in eyebrow_list: #眉毛关键点往上移动两个眼睑的距离，即把额头的位置也放到提取人脸特征的区域范围
                                y = (landmarks.part(i).y-2*extras) if (landmarks.part(i).y-2*extras) > 0 else 0 #超过图片上方取0
                            else:
                                y = landmarks.part(i).y
                            upper.append([landmarks.part(i).x, y])
                        upper = np.array(upper)
                        hull = cv2.convexHull(upper) #找到能涵盖所有关键点的最小凸包
                        mask_align = np.zeros(face_align.shape, dtype=np.uint8) #全黑mask底图
                        mask_align = cv2.fillPoly(mask_align, [hull], [255, 255, 255]) #用白色填充凸包

                        # 获取上半脸的Bbox坐标
                        h, w, c = face_align.shape
                        xmin, ymin = np.clip(np.min(upper, axis=0), 0, None) #不小于0
                        xmax = np.clip(np.max(upper, axis=0)[0], None, w) #不超过图片范围
                        ymax = np.clip(np.max(upper, axis=0)[1], None, h)
                else:
                    face_align=None #避免首次检测到人脸，结果再次检测失败的情况，为了所有状态都能一一对应，若第二次失败则不创建该组数据

        # 只有成功创建新图时，才保存图片，后续创建dataset时直接从新路径读取，所以不会存在检测不到人脸的图片
        if face_align is not None: 
            if unmasked: #要创建不带口罩版本
                cv2.imwrite(image_face_path, face_align_unmasked) #原图
            if masked: #要创建戴口罩版本
                cv2.imwrite(image_masked_face_path, face_align) #保存口罩图片
            if mask: 
                cv2.imwrite(image_mask_path, mask_align) #mask
                with open(image_mask_bbox_path, 'w') as f: #记录不被遮挡脸部的Bbox坐标
                    f.write(f'{xmin},{ymin},{xmax},{ymax}\n')
        else:
            skip_num += 1 
    return skip_num


# 随机显示各个版本下的图片，由于原图中有很多无法被制作成masked图，所以以TRAIN_FACE_PATH版本为标准
def show_versions(versions, titles):
    train = load_train_list(os.path.join(DATA_ROOT,TRAIN_FACE_PATH+'.csv')) 
    idx = np.random.choice(len(train))
    plt.figure(figsize=(12,3))
    for i, ver in enumerate(versions):
        path = train['path'][idx].replace(TRAIN_FACE_PATH, ver) #以
        plt.subplot(1,4,i+1)
        plt.imshow(plt.imread(path))
        plt.title(titles[i])
        plt.axis(False)
    plt.suptitle(train['name'][idx],fontsize=13)
    plt.show()


if __name__=='__main__':
    # # 在终端运行模型时，用命令行参数解析设置config，创建戴口罩数据集
    # parser = argparse.ArgumentParser(description='Generate masked datasets in sections')
    # parser.add_argument('--start_p', type=float, default=0.0)
    # parser.add_argument('--end_p', type=float, default=0.1)
    # parser.add_argument('--unmasked', type=bool, default=False)
    # parser.add_argument('--masked', type=bool, default=False)
    # parser.add_argument('--mask', type=bool, default=False)
    # config = parser.parse_args()
    # print(config)
    # create_masked_dataset(os.path.join(DATA_ROOT, TRAIN_PATH), start_p=config.start_p, end_p=config.end_p,
    #                         unmasked=config.unmasked, masked=config.masked, mask=config.mask)


    # # 为新创建的数据集生成data list
    # create_train_list(os.path.join(DATA_ROOT, TRAIN_FACE_PATH))
    # create_train_list(os.path.join(DATA_ROOT, TRAIN_MASKED_FACE_PATH))


    # 查看已生成的数据集
    versions = [TRAIN_PATH, TRAIN_FACE_PATH, TRAIN_MASKED_FACE_PATH, TRAIN_MASK_PATH]
    titles = ['Original', 'Cropped & Aligned', 'Masked, Cropped & Aligned', 'Unmasked Mask']
    show_versions(versions, titles)

