import cv2
import random
import numpy as np
from imgaug import augmenters as iaa


IMAGE_SIZE = 112 

def TTA_crops(image, target_shape=(32, 32, 3), n_patches=36):
    '''
    Parameters
    ----------
    n_patches : int
                每张图片截取的patches数目，包括：
                5: 中间十字5个patch，共1*5=5种patches
                18：中间9宫格，原patch+1种翻转，共2*9=18种patches
                36: 中间9宫格，原patch+3种翻转，共4*9=36种patches
    '''
    
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    width, height, d = image.shape
    target_w, target_h, d = target_shape
    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    #取原图正中间的9方格
    starts = [[start_x, start_y], #中点

              [start_x - target_w, start_y], #十字
              [start_x, start_y - target_h],
              [start_x + target_w, start_y],
              [start_x, start_y + target_h],

              [start_x + target_w, start_y + target_h], #四角
              [start_x - target_w, start_y - target_h],
              [start_x - target_w, start_y + target_h],
              [start_x + target_w, start_y - target_h],
              ]

    if n_patches == 5:
        starts = starts[:5] #只选取原图正中间的十字5方格
    
    images = []
    for start_index in starts:
        x, y = start_index
        
        # 如果超出左/上边，往右/下移
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        
        # 如果超出右/下边，往左/上移
        if x + target_w >= IMAGE_SIZE:
            x = IMAGE_SIZE - target_w-1
        if y + target_h >= IMAGE_SIZE:
            y = IMAGE_SIZE - target_h-1

        # 截取每个patch的形状
        image_ = image.copy() #深拷贝，不改变原图
        image_ = image_[x:x + target_w, y: y+target_h, :]
        if n_patches >= 5:
            images.append(image_[np.newaxis,...]) #(1,h,w,c)
        if n_patches >=18:
            image_flip_lr = np.fliplr(image_.copy())
            images.append(image_flip_lr[np.newaxis,...])
        if n_patches >=36:
            image_flip_up = np.flipud(image_.copy())
            image_flip_lr_up = np.fliplr(np.flipud(image_.copy()))
            images.append(image_flip_up[np.newaxis,...])
            images.append(image_flip_lr_up[np.newaxis,...])

    return images


def random_erasing(img, probability = 0.5, sl = 0.02, sh = 0.5, r1 = 0.5, channel = 3):
    if random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)

            noise = np.random.random((h,w,channel))*255
            noise = noise.astype(np.uint8)

            if img.shape[2] == channel:
                img[x1:x1 + h, y1:y1 + w, :] = noise
            else:
                print('wrong')
                return
            return img

    return img


def random_resize(img, probability = 0.5,  minRatio = 0.2):
    if random.uniform(0, 1) > probability:
        return img #有一半的机会返回原图

    ratio = random.uniform(minRatio, 1.0)
    h, w, c = img.shape
    new_h = int(h*ratio) #随机缩小
    new_w = int(w*ratio)
    img = cv2.resize(img, (new_w,new_h)) #随机缩小后
    img = cv2.resize(img, (w,h)) #再放大，由于插值法不同，像素值会有所改变
    return img


def random_cropping(image, target_shape=(32, 32, 3), is_random = True):
    image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
    target_h, target_w,_ = target_shape
    height, width, _ = image.shape
    
    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    return zeros


def augumentor(image, target_shape=(32, 32, 3), is_infer=False, n_patches=36):
    if is_infer: #通过各种随机变换后，返回多个patches
        augment_img = iaa.Sequential([iaa.Fliplr(0)])
        image =  augment_img.augment_image(image)
        image = TTA_crops(image, target_shape, n_patches) #返回列表[(1,h,w,c),...]
        image = np.concatenate(image, axis=0) #(n,h,w,c)
        return image

    else: #通过各种随机变换后，再随机resize和crop成1个patch
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)
        image = augment_img.augment_image(image)
        image = random_resize(image) 
        image = random_cropping(image, target_shape, is_random=True) #如果需要各模态取同个位置，可以拼接后再crop
        return image
