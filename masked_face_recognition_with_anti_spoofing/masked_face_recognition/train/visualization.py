import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from config import *
import os
import glob
import imageio

def prepare_heatmap(image, feature_map): #(3,h,w) (c,h,w)
    # 对最后一个Conv层可视化时，从channel方向求每个pixel的feature均值，再默认取第一个样本的结果
    h,w = image.shape[1:]  
    heatmap = torch.mean(feature_map, dim=0) #(8,8) 综合图
    heatmap = (heatmap - torch.min(heatmap))/(torch.max(heatmap)-torch.min(heatmap)) #normalize [0,1]
    heatmap = (heatmap*255).to(torch.uint8) #[0,255]数组
    heatmap = cv2.resize(heatmap.detach().cpu().numpy(), (h,w)) #放大到原图大小
    heatmap = np.clip(heatmap, 0, 255) #保证取值范围
    image = (image*255).to(torch.uint8).detach().cpu().numpy() #转为[0,255]数组
    image = np.transpose(image, [1,2,0])[...,::-1] # (h,w,3)的RGB
    return image, heatmap


def plot_heatmap(img1, img2, map1, map2, dist, label, epoch, heatmap_dir=''): #(3,h,w) (c,h,w)
    # 对最后一个Conv层可视化时，从channel方向求每个pixel的feature均值，再默认取第一个样本的结果
    img1, map1 = prepare_heatmap(img1, map1)
    img2, map2 = prepare_heatmap(img2, map2)

    f,ax = plt.subplots(2,1, figsize=(6,5))
    # 图1：原图
    ax[0].imshow(cv2.hconcat([img1, img2]))
    ax[0].set_title(f"Predicted dist: {dist:.3f}, Same: {label} (epoch {epoch:02})",fontsize=13)
    ax[0].axis('off')

    # 图2：混合原图和最后一层atention
    map1 = cv2.applyColorMap(map1, cv2.COLORMAP_JET) #叠加时两张图需要相同维度，因此需要给map分配颜色，变成(h,w,3)
    map2 = cv2.applyColorMap(map2, cv2.COLORMAP_JET) #HOT/JET都可以
    #alpha/beta是两张图片的比例，gamma是在叠合图上额外加的像素值
    superimposed1 = cv2.addWeighted(src1=img1, alpha=0.7, src2=map1, beta=0.2, gamma=0.0)
    superimposed2 = cv2.addWeighted(src1=img2, alpha=0.7, src2=map2, beta=0.2, gamma=0.0)
    ax[1].imshow(cv2.hconcat([superimposed1, superimposed2]))
    ax[1].set_title("Model Attention on the Last Conv Layer", fontsize=13)
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, f'epoch{epoch:02}_heatmap.png'))
    # plt.show()


def generate_heatmap_gif(heatmap_dir=''):
    filenames = glob.glob(os.path.join(heatmap_dir,f'epoch*heatmap.png')) #完整路径，同时避免.DS_stores等隐藏文件
    filenames = sorted(filenames) #确保按顺序
    plots = []
    for filename in filenames:
        image = plt.imread(filename)
        image = (image*255).astype('uint8')
        plots.append(image)
    imageio.mimsave(os.path.join(heatmap_dir, f'heatmap.gif'), plots, 'GIF-FI', fps=2)
    # display(IPyImage(open(os.path.join(heatmap_dir, f'heatmap.gif'), 'rb').read())) #显示动画