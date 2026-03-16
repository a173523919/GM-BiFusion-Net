import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# ------------------ 路径与参数 ------------------
rgb_dir = Path("测试用例")   # RGB 图像 

model_path = "runs/detect/train/weights/best.pt" # 模型路径
output_dir = Path("RGB可视化")  # 输出目录
output_dir.mkdir(exist_ok=True)

target_size = (640, 640)  # 目标尺寸

# 固定要处理的图片（不含扩展名）
selected_stems = ["11", "12", "13", "14", "15","16","7"]

# 类别名称（必须和 labels 里的 class_id 顺序一致）
class_names =["antelope","donkey", "person"]

line_thickness = 1   # 线条粗细

# ------------------ 通用函数 ------------------
def preprocess_image(path, target_size):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, target_size)
    return img

def draw_boxes(img, boxes, labels, color=(0, 255, 0), thickness=1):
    """
    画框并在左上角写类别
    boxes: list of [x1, y1, x2, y2]
    labels: list of int (class id)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    for (x1, y1, x2, y2), cls_id in zip(boxes, labels):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 2), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

def draw_stats(img, stats_dict):
    """在图像左上角绘制类别统计信息（按 class_names 顺序，数量为 0 不显示）"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 1
    line_height = 25

    # 按 class_names 的顺序输出，跳过数量为 0 的类别
    draw_idx = 0   # 记录"真正画出来的行号"，避免空行
    for cls_name in class_names:
        count = stats_dict.get(cls_name, 0)
        if count == 0:
            continue
        text = f"{cls_name}: {count}"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        y_pos = 5 + th + 5 + draw_idx * line_height

        # 绘制文本背景
        cv2.rectangle(img, (10, y_pos - th), (20 + tw, y_pos + 5), (0, 0, 0), -1)
        # 绘制文本
        cv2.putText(img, text, (10, y_pos), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        draw_idx += 1

# ------------------ 主流程 ------------------
model = YOLO(model_path).to('cpu')

# 全局统计所有图片的类别数量
global_stats = defaultdict(int)

for i, stem in enumerate(selected_stems, 1):
    # 尝试 jpg / png
    rgb_path = rgb_dir / f"{stem}.jpg"
    if not rgb_path.exists():
        rgb_path = rgb_path.with_suffix(".jpg")

    rgb_img = preprocess_image(rgb_path, target_size)

    # 直接使用RGB图像进行推理
    input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    print("输入张量形状",input_tensor.shape)

    # 推理
    results = model(input_tensor, imgsz=target_size, verbose=False,device='cpu')

    # 提取框 + 类别
    pred_boxes, pred_labels = [], []
    # 当前图片的类别统计
    current_stats = defaultdict(int)
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                cls = int(box.cls.cpu().item())
                pred_boxes.append([x1, y1, x2, y2])
                pred_labels.append(cls)
                
                # 更新统计
                cls_name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
                current_stats[cls_name] += 1
                global_stats[cls_name] += 1

    # 画预测框
    rgb_vis = rgb_img.copy()
    draw_boxes(rgb_vis, pred_boxes, pred_labels, thickness=line_thickness)
    
    # 绘制当前图片的统计信息
    draw_stats(rgb_vis, current_stats)

    # 保存
    cv2.imwrite(str(output_dir / f"{i}_rgb.jpg"), rgb_vis)

    # 打印当前图片统计
    print(f"Image {stem} detection stats:")
    for cls, count in current_stats.items():
        print(f"  {cls}: {count}")
    print(f"Processed {i}/{len(selected_stems)}: {stem}")

# 打印全局统计
print("\nGlobal detection statistics:")
for cls, count in global_stats.items():
    print(f"  {cls}: {count}")

print("✅ 测试完成，结果保存在"+str(output_dir)+"目录")