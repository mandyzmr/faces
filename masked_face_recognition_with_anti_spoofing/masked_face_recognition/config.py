import dlib
import torch
from torchvision import transforms 
import multiprocessing
import sys
# sys.path.append('/home/aistudio/external_library') #ai studio应用环境
import os
current_dir = os.path.abspath('./masked_face_recognition') #从上一级路径的总项目调用时
# current_dir = os.path.abspath('./') #从当前路径调用时

'''
此处定义常用固定配置，若需要根据情况训练不同模型，直接通过main.py用终端指定参数进行训练和预测。
'''

#------Device------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_cores = multiprocessing.cpu_count()
torch.backends.cudnn.benchmark=True #cuDNN autotuner

#------Original Dataset------
DATA_ROOT = 'datasets' 
TRAIN_PATH = 'vggface2_train' #检测对齐前的原始图片
TRAIN_LIST = 'vggface2_train.csv' 
VAL_PATH = 'lfw_funneled' #检测对齐前的原始图片
VAL_LIST = 'pairs.txt'

#------Masked Dataset------
IMAGE_SIZE = 256 #截图统一尺寸
TRAIN_FACE_PATH = 'vggface2_train_face' #原版正脸对齐截图
TRAIN_MASKED_FACE_PATH = 'vggface2_train_masked_face' #戴口罩版的正脸对齐截图
TRAIN_MASK_PATH = 'vggface2_train_mask' #未覆盖人脸部位的mask
TRAIN_MASK_BBOX_PATH = 'vggface2_train_mask_bbox' #未覆盖人脸部位的(x,y)坐标范围.txt
VAL_FACE_PATH = 'lfw_funneled' #原版正脸对齐截图
VAL_MASKED_FACE_PATH = 'lfw_funneled_masked' #戴口罩版的正脸对齐截图

#------Detection Model------
# 初始化模型
detector = dlib.get_frontal_face_detector() # 人脸检测模型
predictor = dlib.shape_predictor(os.path.join(current_dir, 'model', 'shape_predictor_68_face_landmarks.dat')) #68个关键点检测模型     
recognizer = dlib.face_recognition_model_v1(os.path.join(current_dir, 'model', 'dlib_face_recognition_resnet_model_v1.dat')) #人脸特征提取模型
        
#------Dataset Loader------
BATCH_SIZE = 10
TRAIN_SIZE = 0.8 #在用SingleDataset在softmax下训练时，把戴口罩数据按比例分成train和test set
N_TRIPLETS = 1000 #再用TripletDataset时，三元组的个数
train_transform = transforms.Compose([
    transforms.ToTensor(), #把array/PIL从[0,255]转为[0,1]的FloatTensor [c,h,w]
    # transforms.RandomHorizontalFlip(), # 随机翻转 PIL/tensor，同时输出mask的时候不用random flip
    # transforms.Normalize( #对tensor的channel进行normalization
    #     mean=[0.5, 0.5, 0.5],
    #     std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(), #把array/PIL从[0,255]转为[0,1]的FloatTensor [c,h,w]
    # transforms.Normalize(
    #     mean=[0.5, 0.5, 0.5],
    #     std=[0.5, 0.5, 0.5])
])

if device.__str__() == 'cuda':
    num_workers = 4 #1000 samples - 4, 5k - 2
    pin_memory = True #使用锁页内存加速复制数据到gpu
else: #AIStudio终端查看df -h可以看到在CPU配置下，shm只有64m，需要增加limit防止爆内存docker run--shm-size 8g，若没有权限，需要设置num_workers=0
    num_workers = cpu_cores
    pin_memory = False

#------Model------
EMBEDDING_DIM = 128
N_CLASSES = 500
PRETRAINED = False

#------Train------
OPTIMIZER = 'sgd' #sgd, adagrad, adam
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 5e-4
N_FOLDS = 10 #evaluate metrics
DIST_MARGIN = 0.1


