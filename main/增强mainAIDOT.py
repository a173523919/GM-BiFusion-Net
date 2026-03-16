from ultralytics import YOLO
import torch
import numpy as np
import random
import os

from pathlib import Path
folder_path = Path('待实验/线程1')  # 替换为你的文件夹路径
filenames=[]

for file_path in folder_path.iterdir():
    if file_path.is_file():
        filenames.append(str(folder_path)+'/'+file_path.name)

    
for cfg_path in filenames:
    
    model_name = os.path.splitext(os.path.basename(cfg_path))[0]
    #print(cfg_path,"######",model_name)
    for seed in [1234,56710,856397,76014,52319]:
        model = YOLO(cfg_path)
        train_args = {
        'name':str(model_name)+str(seed),
        'data':'ultralytics/cfg/datasets/mydataAIDOT_test.yaml',
        'seed':seed,
        'amp':False,
        'hsv_h':0.0,
        'hsv_s':0.0,
        'hsv_v':0.0,
        }
        model.train(**train_args)  # 训练
    





