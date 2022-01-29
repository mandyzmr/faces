import os
import torch
import sys
current_dir = os.path.abspath('./face_anti_spoofing') #从上一级路径的总项目调用时
# current_dir = os.path.abspath('./') #从当前路径调用时
sys.path.append(os.path.join(current_dir,'model')) #增加绝对路径
sys.path.append(os.path.join(current_dir, 'preprocess'))
sys.path.append(os.path.join(current_dir, 'train'))
from model.facebagnet import *
from preprocess.augmentation import *
from train.train_infer import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceAntiSpoofing():
    def __init__(self, modality='color', patch_size=48):
        self.modality = modality
        self.patch_size = patch_size
        
        # 加载预训练模型
        pretrained_models = {'color': {'name': 'SE-ResNeXt18', 'attention':False, 'checkpoint': os.path.join(current_dir,'running_log','SE-ResNeXt18_color_48', 'checkpoint', 'global_min_acer_model.pth')},
                                      #{'name': 'ResNet18', 'attention':False, 'checkpoint': os.path.join(current_dir,'running_log','ResNet18_color_48', 'checkpoint', 'global_min_acer_model.pth')},
                             'fusion': {'name': 'ResNet18', 'attention': True, 'checkpoint': os.path.join(current_dir,'running_log','FaceBagNet_ResNet18_att_48', 'checkpoint', 'global_min_acer_model.pth')}}
        model_name = pretrained_models[self.modality]['name']
        attention = pretrained_models[self.modality]['attention']
        pretrained_path = pretrained_models[self.modality]['checkpoint']
        self.model = get_model(name=model_name, modality=self.modality, num_classes=2, attention=attention)
        self.model = load_model_checkpoint(self.model, pretrained_path, device)
        self.model.to(device)
        
    def classify(self, image): 
        # 预处理图片，得到多个patches
        image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
        image = augumentor(image, target_shape=(self.patch_size, self.patch_size, 3), 
                           is_infer=True, n_patches=36) #(n,h,w,3)
        image = np.transpose(image, (0, 3, 1, 2)) #(n,3,h,w)
        image = image.astype(np.float32)
        image = image / 255.0
        image = torch.FloatTensor(image)
        image = image.unsqueeze(0) #(1,n,3,h,w)
        fake_label = None

        # 预测
        y_pred = predict(self.model, [(image,fake_label)], device) #返回1个概率
        y_pred_label = y_pred[0]>0.5 #True/False
        return y_pred_label, y_pred[0] #label, prob


if __name__== "__main__": 

    anti_spoofing = FaceAntiSpoofing()
    image = cv2.imread(os.path.join('casia_surf_dataset','alive.jpg'))
    label = anti_spoofing.classify(image)
    print(label)