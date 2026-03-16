from ultralytics import YOLO
import torch
import numpy as np
import random


from pathlib import Path
folder_path = Path('待实验/线程9')  # 替换为你的文件夹路径
filenames=[]
for file_path in folder_path.iterdir():
    if file_path.is_file():
        filenames.append(str(folder_path)+'/'+file_path.name)
cfg_path=filenames[0]
for seed in [856397,76014,52319]: 
    model = YOLO(cfg_path)
    train_args = {
    'name':"YOLO11N"+str(seed),
    'data':'ultralytics/cfg/datasets/mydatadrone.yaml',
    'seed':seed,
        
    'augment': False,  # 主开关
    'pretrained': False,
    # 显式关闭各项增强
    'mosaic': 0.0,        # 关闭马赛克增强
    'mixup': 0.0,         # 关闭MixUp
    'copy_paste': 0.0,    # 关闭复制粘贴
    'hsv_h': 0.0,         # 色相增强强度
    'hsv_s': 0.0,         # 饱和度增强强度
    'hsv_v': 0.0,         # 明度增强强度
    'translate': 0.0,     # 平移增强
    'scale': 0.0,         # 缩放增强
    'shear': 0.0,         # 剪切增强
    'perspective': 0.0,   # 透视增强
    'flipud': 0.0,       # 上下翻转概率
    'fliplr': 0.0,       # 左右翻转概率
    'degrees': 0.0,       # 旋转角度范围
    'rect': False         # 禁用矩形训练
    }
    model.train(**train_args)  # 训练



