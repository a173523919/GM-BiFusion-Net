import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

#部分版本会自动翻转图片，尝试切换
#im6 = np.concatenate([rgb_img, ir_img], axis=2)  # HWC, 6
#im6 = np.concatenate([ir_img, rgb_img], axis=2)  # HWC, 6

# ------------------ 路径与参数 ------------------
rgb_dir = Path("/root/autodl-tmp/datasets/drone/mydata/images/test")      # RGB
ir_dir  = Path("/root/autodl-tmp/datasets/drone/mydata/image/test")   # IR

model_path = "runs/obb/H2Fhead+gconv+MambaN1234/weights/last.pt"

output_dir = Path("output可视化")
output_dir.mkdir(exist_ok=True)

#selected_stems = ["00042", "00107","01159","03472"]
selected_stems = ["00042","03472","04748","04799"]
class_names = ["bus","car","van", "feright_car", "truck"]
line_thickness = 1

# ------------------ 通用函数 ------------------
def draw_hbb(img, boxes_xyxy, labels, color=(0, 255, 0), thickness=1):
    """画水平框 xyxy"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    for (x1, y1, x2, y2), cls_id in zip(boxes_xyxy, labels):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 2), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

def draw_obb_poly(img, polys, labels, color=(0, 255, 0), thickness=1):
    """画旋转框（四点多边形），polys: (N,4,2)"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    for poly, cls_id in zip(polys, labels):
        poly = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(img, [poly], isClosed=True, color=color, thickness=thickness)

        # 在第一个点附近写类别
        x, y = int(poly[0, 0]), int(poly[0, 1])
        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x, y - th - 2), (x + tw, y), color, -1)
        cv2.putText(img, label, (x, y - 2), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

def draw_stats(img, stats_dict):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 1
    line_height = 25

    draw_idx = 0
    for cls_name in class_names:
        count = stats_dict.get(cls_name, 0)
        if count == 0:
            continue
        text = f"{cls_name}: {count}"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        y_pos = 5 + th + 5 + draw_idx * line_height
        cv2.rectangle(img, (10, y_pos - th), (20 + tw, y_pos + 5), (0, 0, 0), -1)
        cv2.putText(img, text, (10, y_pos), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        draw_idx += 1

def extract_dets_unified(result):
    """
    统一抽取检测结果，支持 OBB/HBB：
    返回：
      mode: "obb" or "hbb" or "none"
      boxes_xyxy: list[(x1,y1,x2,y2)]   # 对 HBB 或 OBB fallback
      polys: list[np.ndarray shape(4,2)]# 仅 OBB 真正旋转框有
      labels: list[int]
      confs: list[float]
    """
    boxes_xyxy, polys, labels, confs = [], [], [], []

    # 1) 优先 OBB
    if getattr(result, "obb", None) is not None and len(result.obb):
        obb = result.obb
        labels = obb.cls.cpu().numpy().astype(int).tolist()
        confs  = obb.conf.cpu().numpy().astype(float).tolist() if hasattr(obb, "conf") else [0.0] * len(labels)

        # 尝试取旋转四点
        if hasattr(obb, "xyxyxyxy"):
            p = obb.xyxyxyxy.cpu().numpy()
            # 兼容 (N,8) 或 (N,4,2)
            p = p.reshape(-1, 4, 2)
            polys = [pi for pi in p]
            return "obb", boxes_xyxy, polys, labels, confs

        # 否则退化为外接 HBB（画矩形）
        if hasattr(obb, "xyxy"):
            b = obb.xyxy.cpu().numpy()
            boxes_xyxy = [bi.tolist() for bi in b]
            return "obb", boxes_xyxy, polys, labels, confs

        return "obb", boxes_xyxy, polys, labels, confs  # 极端情况

    # 2) HBB
    if result.boxes is not None and len(result.boxes):
        b = result.boxes
        labels = b.cls.cpu().numpy().astype(int).tolist()
        confs  = b.conf.cpu().numpy().astype(float).tolist() if hasattr(b, "conf") else [0.0] * len(labels)
        xyxy   = b.xyxy.cpu().numpy()
        boxes_xyxy = [bi.tolist() for bi in xyxy]
        return "hbb", boxes_xyxy, polys, labels, confs

    return "none", boxes_xyxy, polys, labels, confs

# ------------------ 主流程 ------------------
model = YOLO(model_path)
global_stats = defaultdict(int)

for i, stem in enumerate(selected_stems, 1):
    rgb_path = rgb_dir / f"{stem}.jpg"
    ir_path  = ir_dir  / f"{stem}.jpg"
    if not rgb_path.exists():
        rgb_path = rgb_path.with_suffix(".png")
    if not ir_path.exists():
        ir_path = ir_path.with_suffix(".png")

    rgb_img = cv2.imread(str(rgb_path))
    ir_img  = cv2.imread(str(ir_path))

    # (可选) 保证和训练一致的尺寸
    rgb_img = cv2.resize(rgb_img, (640, 640))
    ir_img  = cv2.resize(ir_img,  (640, 640))

    #im6 = np.concatenate([rgb_img, ir_img], axis=2)  # HWC, 6
    im6 = np.concatenate([ir_img, rgb_img], axis=2)  # HWC, 6
   

    results = model.predict(source=im6, imgsz=640, conf=0.25, iou=0.5, verbose=False)
    r0 = results[0]

    mode, boxes_xyxy, polys, labels, confs = extract_dets_unified(r0)

    print(f"[{stem}] mode={mode}, num det={len(labels)}",
          f", max conf={max(confs) if len(confs) else None}")

    # 当前图片统计
    current_stats = defaultdict(int)
    for c in labels:
        cls_name = class_names[c] if 0 <= c < len(class_names) else str(c)
        current_stats[cls_name] += 1
        global_stats[cls_name] += 1

    # 可视化
    rgb_vis = rgb_img.copy()
    ir_vis  = ir_img.copy()

    if mode == "obb" and len(polys):
        draw_obb_poly(rgb_vis, polys, labels, thickness=line_thickness)
        draw_obb_poly(ir_vis,  polys, labels, thickness=line_thickness)
    else:
        # HBB 或 OBB fallback（只有 xyxy）
        draw_hbb(rgb_vis, boxes_xyxy, labels, thickness=line_thickness)
        draw_hbb(ir_vis,  boxes_xyxy, labels, thickness=line_thickness)

    draw_stats(rgb_vis, current_stats)
    draw_stats(ir_vis, current_stats)

    cv2.imwrite(str(output_dir / f"{i}_rgb.jpg"), rgb_vis)
    cv2.imwrite(str(output_dir / f"{i}_ir.jpg"),  ir_vis)

    print(f"Image {stem} detection stats:")
    for cls, count in current_stats.items():
        print(f"  {cls}: {count}")
    print(f"Processed {i}/{len(selected_stems)}: {stem}")

print("\nGlobal detection statistics:")
for cls, count in global_stats.items():
    print(f"  {cls}: {count}")

print("✅ 测试完成，结果保存在 output可视化/ 目录")
