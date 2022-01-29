import cv2
import numpy as np
import matplotlib.pyplot as plt 
import torch
import pickle
import dlib
from tqdm import tqdm
import PySimpleGUI as sg
sg.theme("BlueMono")
from main import *

def photo_emb(name_path):
    image_list = os.listdir(name_path)
    image_list.sort()
    embs = []
    for image in image_list: #每个人的图片路径
        image_path = os.path.join(name_path, image)
        image = cv2.imread(image_path)
        emb, _ = model.get_emb(image)
        embs.append(np.array(emb))
    emb = np.mean(embs, axis=0) #取10张图片预测的emb均值
    return emb


# 多渠道组建人脸库
def create_face_database(model):
    face_database = {}
    finished = False

    print(f'Creating face database ...')        
    while not finished:
        event, values = choose_path()
        if event=='Finish': #完成创建，跳出循环
            break

        filename = f"{values['dep']}_face_database.pkl"
        if values['photo']: 
            if values['image_path']: #单个员工照片
                emb = photo_emb(values['image_path'])
                face_database[values['name']] = emb

            elif values['folder_path']: #批量员工照片
                folder_path  = values['folder_path']
                ppl_list = os.listdir(folder_path) #每个员工提供10张照片
                ppl_list.sort()
                for name in tqdm(ppl_list):
                    name_path = os.path.join(folder_path, name)
                    if os.path.isdir(name_path): #只有在是文件夹的情况下
                        emb = photo_emb(name_path)
                        face_database[name] = emb

        elif values['video']: #视频创建
            emb = model.face_recog(values['video_path']) 
            face_database[values['name']] = emb

        elif values['webcam']: #摄像头
            emb = model.face_recog() 
            face_database[values['name']] = emb
    
    with open(filename, 'wb') as f:
        pickle.dump(face_database, f)
    print(f'Face database for {len(face_database)} staff has been created and saved to {filename}.')
    return face_database


def choose_path(): #GUI可视化窗口，可以根据自定义情况选择员工人脸数据库路径
    window = sg.Window(
        title="创建员工人脸库",
        layout=[
            [sg.Text('部门：'), sg.InputText(key='dep', size=(10,1)),
             sg.Text('姓名：'), sg.InputText(key='name', size=(10,1))],
            [sg.Radio('照片', group_id='R', key='photo', default=True)],
            [sg.Text('单人照片文件夹：'), sg.InputText(key='image_path'), sg.FolderBrowse(initial_folder = os.getcwd())],
            [sg.Text('多人照片文件夹：'), sg.InputText(key='folder_path'), sg.FolderBrowse(initial_folder = os.getcwd())],
            [sg.Radio('视频', group_id='R', key='video')],
            [sg.Text('单人视频文件：'), sg.InputText(key='video_path'), sg.FileBrowse(initial_folder = os.getcwd())],
            [sg.Radio('摄像头', group_id='R', key='webcam')],
            [sg.Button('Start'), sg.Button('Finish')],
        ],
    )
    
    event, values = window.read()
    window.close()
    return event, values


if __name__=='__main__':
    model = MaskedFaceRecognition_AntiSpoofing()
    face_database = create_face_database(model)
    print(face_database.keys())








