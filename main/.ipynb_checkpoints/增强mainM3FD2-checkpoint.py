from ultralytics import YOLO
import torch
import numpy as np
import random
import os

from pathlib import Path
folder_path = Path('待实验/M3FD线程1')  # 替换为你的文件夹路径
filenames=[]

for file_path in folder_path.iterdir():
    if file_path.is_file():
        filenames.append(str(folder_path)+'/'+file_path.name)

    
for cfg_path in filenames:
    
    model_name = os.path.splitext(os.path.basename(cfg_path))[0]
    #print(cfg_path,"######",model_name)
    for seed in [56710,52319]:
        model = YOLO(cfg_path)
        train_args = {
        'name':"数据集种子1234"+str(model_name)+str(seed),
        'data':'ultralytics/cfg/datasets/mydataM3FD1234.yaml',
        'seed':seed,
        'ch':6,
        'epochs':140,
        'hsv_h': 0.0,         # 色相增强强度
        'hsv_s': 0.0,         # 饱和度增强强度
        'hsv_v': 0.0,         # 明度增强强度
        }
        model.train(**train_args)  # 训练
    




