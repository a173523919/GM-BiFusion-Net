import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import math

# ------------------ 路径与参数 ------------------
rgb_dir = Path("/root/autodl-tmp/datasets/M3FD/dataset_yolo1234/images/test")      # RGB
ir_dir  = Path("/root/autodl-tmp/datasets/M3FD/dataset_yolo1234/image/test")   # IR
label_dir  = Path("/root/autodl-tmp/datasets/M3FD/dataset_yolo1234/labels/test")

output_dir = Path("out_GT_vis")
output_dir.mkdir(exist_ok=True)

target_size = (640, 640)  # (W,H)

#selected_stems = ["00042", "00107","00836","00900","00973","01159","01192","03472"]
#selected_stems = ["00042", "00107","01159","03472"]
selected_stems = ["00214","00061","01487"]
class_names = ["people","car","bus", "lamp", "motorcycle","truck"]
line_thickness = 1

# ------------------ 工具函数 ------------------
def preprocess_image(path, target_size):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, target_size)
    return img

def yolo_hbb_to_xyxy(cls, cx, cy, w, h, img_w, img_h):
    """HBB: (cls,cx,cy,w,h) normalized -> xyxy abs"""
    cx *= img_w
    cy *= img_h
    w  *= img_w
    h  *= img_h
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return int(cls), [x1, y1, x2, y2]

def yolo_obb4pts_to_poly(parts, img_w, img_h):
    """
    OBB 9列: cls x1 y1 x2 y2 x3 y3 x4 y4
    默认认为 x,y 是归一化到[0,1]，这里转绝对坐标
    """
    cls = int(float(parts[0]))
    pts = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(4, 2)
    pts[:, 0] *= img_w
    pts[:, 1] *= img_h
    return cls, pts.astype(np.int32)

def yolo_obb6_to_poly(parts, img_w, img_h):
    """
    OBB 6列: cls cx cy w h angle
    默认：cx,cy,w,h 归一化；angle 是度（deg）
    如果你的 angle 是弧度：把 angle_deg = angle * 180/math.pi
    如果你的 cxcywh 是绝对像素：删掉乘 img_w/img_h 的那几行
    """
    cls = int(float(parts[0]))
    cx, cy, w, h, angle = map(float, parts[1:])

    # 认为是归一化：
    cx *= img_w
    cy *= img_h
    w  *= img_w
    h  *= img_h

    angle_deg = angle  # 默认 angle 已经是 “度”
    # 如果你的 angle 是弧度，改成：
    # angle_deg = angle * 180.0 / math.pi

    rect = ((cx, cy), (w, h), angle_deg)  # OpenCV: angle in degrees
    poly = cv2.boxPoints(rect)  # (4,2) float
    return cls, poly.astype(np.int32)

def poly_to_xyxy(poly):
    xs = poly[:, 0]
    ys = poly[:, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def draw_hbb(img, boxes_xyxy, labels, color=(0,255,0), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    for (x1, y1, x2, y2), cls_id in zip(boxes_xyxy, labels):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 2), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), font, font_scale, (0,0,0), 1, cv2.LINE_AA)

def draw_polys(img, polys, labels, color=(0,255,0), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    for poly, cls_id in zip(polys, labels):
        poly = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(img, [poly], True, color, thickness)

        x, y = int(poly[0,0]), int(poly[0,1])
        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x, y - th - 2), (x + tw, y), color, -1)
        cv2.putText(img, label, (x, y - 2), font, font_scale, (0,0,0), 1, cv2.LINE_AA)

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
        cv2.rectangle(img, (10, y_pos - th), (20 + tw, y_pos + 5), (0,0,0), -1)
        cv2.putText(img, text, (10, y_pos), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        draw_idx += 1

def read_labels_unified(label_path, img_w, img_h):
    """
    返回：
      hbb_boxes: list[xyxy]
      obb_polys: list[(4,2)]
      labels:    list[int]  (与 hbb_boxes/obb_polys 对应)
      modes:     list[str]  ('hbb'/'obb') 方便调试
    画的时候：优先画 obb_polys；没有 obb 才画 hbb_boxes
    """
    hbb_boxes, obb_polys, labels, modes = [], [], [], []

    if not label_path.exists():
        return hbb_boxes, obb_polys, labels, modes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # HBB: 5列
            if len(parts) == 5:
                cls, cx, cy, w, h = map(float, parts)
                cls_i, box = yolo_hbb_to_xyxy(cls, cx, cy, w, h, img_w, img_h)
                hbb_boxes.append(box)
                labels.append(cls_i)
                modes.append("hbb")

            # OBB: 9列四点
            elif len(parts) == 9:
                cls_i, poly = yolo_obb4pts_to_poly(parts, img_w, img_h)
                obb_polys.append(poly)
                labels.append(cls_i)
                modes.append("obb")

            # OBB: 6列(cxcywh+angle)
            elif len(parts) == 6:
                cls_i, poly = yolo_obb6_to_poly(parts, img_w, img_h)
                obb_polys.append(poly)
                labels.append(cls_i)
                modes.append("obb")

            else:
                # 其它格式先跳过（避免 silent fail，建议你打印看看）
                # print("Skip line (unknown format):", label_path.name, "cols=", len(parts), parts[:3])
                continue

    return hbb_boxes, obb_polys, labels, modes

# ------------------ 主流程 ------------------
global_stats = defaultdict(int)

W, H = target_size

for i, stem in enumerate(selected_stems, 1):
    rgb_path = (rgb_dir / f"{stem}.jpg")
    ir_path  = (ir_dir  / f"{stem}.jpg")
    if not rgb_path.exists():
        rgb_path = rgb_path.with_suffix(".png")
    if not ir_path.exists():
        ir_path = ir_path.with_suffix(".png")

    rgb_img = preprocess_image(rgb_path, target_size)
    ir_img  = preprocess_image(ir_path,  target_size)

    label_path = label_dir / f"{stem}.txt"
    hbb_boxes, obb_polys, labels, modes = read_labels_unified(label_path, W, H)

    # 统计
    current_stats = defaultdict(int)
    for cls in labels:
        cls_name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
        current_stats[cls_name] += 1
        global_stats[cls_name] += 1

    # 绘制：OBB 优先（如果有 obb_polys），否则画 HBB
    rgb_vis = rgb_img.copy()
    ir_vis  = ir_img.copy()

    if len(obb_polys) > 0:
        # 画旋转框
        draw_polys(rgb_vis, obb_polys, [labels[j] for j,m in enumerate(modes) if m=="obb"], thickness=line_thickness)
        draw_polys(ir_vis,  obb_polys, [labels[j] for j,m in enumerate(modes) if m=="obb"], thickness=line_thickness)

        # （可选）也画外接 HBB，便于对照：取消注释即可
        # obb_boxes = [poly_to_xyxy(p) for p in obb_polys]
        # obb_labels = [labels[j] for j,m in enumerate(modes) if m=="obb"]
        # draw_hbb(rgb_vis, obb_boxes, obb_labels, color=(0,0,255), thickness=1)
        # draw_hbb(ir_vis,  obb_boxes, obb_labels, color=(0,0,255), thickness=1)

    else:
        # 只画水平框
        draw_hbb(rgb_vis, hbb_boxes, labels, thickness=line_thickness)
        draw_hbb(ir_vis,  hbb_boxes, labels, thickness=line_thickness)

    draw_stats(rgb_vis, current_stats)
    draw_stats(ir_vis,  current_stats)

    cv2.imwrite(str(output_dir / f"{i}_rgb.jpg"), rgb_vis)
    cv2.imwrite(str(output_dir / f"{i}_ir.jpg"),  ir_vis)

    print(f"Image {stem} GT stats:")
    for cls, count in current_stats.items():
        print(f"  {cls}: {count}")
    print(f"Processed {i}/{len(selected_stems)}: {stem}")

print("\nGlobal GT statistics:")
for cls, count in global_stats.items():
    print(f"  {cls}: {count}")

print(f"✅ 可视化完成，结果保存在 {output_dir}/ 目录")
