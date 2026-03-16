from ultralytics import YOLO
import torch
import numpy as np
import random


seed=int(1234)

from pathlib import Path
folder_path = Path('待实验/线程2')  # 替换为你的文件夹路径
filenames=[]
for file_path in folder_path.iterdir():
    if file_path.is_file():
        filenames.append(str(folder_path)+'/'+file_path.name)

for cfg_path in filenames: 
    model = YOLO(cfg_path)
    train_args = {
    'data':'ultralytics/cfg/datasets/mydatawild.yaml',
    'seed':seed,
    'hsv_h':0,
    'hsv_s':0,
    'hsv_v':0,
    }
    model.train(**train_args)  # 训练



