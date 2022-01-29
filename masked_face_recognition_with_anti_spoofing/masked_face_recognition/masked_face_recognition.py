import torch
import os
import sys
current_dir = os.path.abspath('./masked_face_recognition') #从上一级路径的总项目调用时
# current_dir = os.path.abspath('./') #从当前路径调用时
sys.path.append(os.path.join(current_dir, 'model')) #增加绝对路径
sys.path.append(os.path.join(current_dir, 'preprocess'))
sys.path.append(os.path.join(current_dir, 'train'))
from summary import *
from config import *
from train.train_triplet_loss import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MaskedFaceRecognition():
    def __init__(self):
        # 加载预训练模型
        self.model = get_model(name='resnet50', embedding_dim=128, n_classes=500, att_model=True, 
                        cbam=True, softmax_criterion=False, att_criterion=False, pretrained=False)
        pretrained_path = os.path.join(current_dir, 'running_log','resnet50_att_cbam_masked_128', 'checkpoint', 'global_max_auc_model.pth')
        checkpoint = torch.load(pretrained_path, device)
        self.model = load_model_checkpoint(self.model, checkpoint, device) 
        self.thres = 1.26 #根据最优模型定下来的dist threshold
      
    def compute_face_descriptor(self, image): 
        image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
        emb = predict(self.model, image, device) 
        return emb


if __name__== "__main__": 
    recognition = MaskedFaceRecognition()
    masked = cv2.imread(os.path.join('testing','0053_01_2.jpg'))
    unmasked = cv2.imread(os.path.join('testing','0001_01_2.jpg'))
    emb1 = recognition.compute_face_descriptor(masked)
    emb2 = recognition.compute_face_descriptor(unmasked)
    dist = np.linalg.norm(emb1-emb2)
    # plt.subplot(1,2,1)
    # plt.imshow(unmasked[...,::-1])
    # plt.subplot(1,2,2)
    # plt.imshow(masked[...,::-1])
    # plt.show()

    print(f'distances: {dist}')
    print(f'similarity: {1-dist/np.sqrt(2)}')
    if dist < recognition.thres:
        print('Same person')
    else:
        print('Different person')       


