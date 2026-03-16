from ultralytics import YOLO
import torch
import numpy as np
import random


seed=int(56710)

from pathlib import Path
folder_path = Path('待实验')  # 替换为你的文件夹路径
filenames=[]
for file_path in folder_path.iterdir():
    if file_path.is_file():
        filenames.append(str(folder_path)+'/'+file_path.name)

for cfg_path in filenames: 
    model = YOLO(cfg_path)
    train_args = {
    'name':"上一篇论文L56710",
    'data':'ultralytics/cfg/datasets/mydatavisdrone.yaml',
    'seed':seed,
    }
    model.train(**train_args)  # 训练



