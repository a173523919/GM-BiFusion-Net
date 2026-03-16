#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import xml.etree.ElementTree as ET

# ===== 只改这三项 =====
images_dir = "/root/autodl-tmp/datasets/M3FD/dataset_yolo1234/images/test"
xml_dir    = "/root/autodl-tmp/datasets/M3FD/xml_labels"
target_class = "people"   # 改成你的类别名
# ======================

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]

def find_image_by_stem(stem: str):
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
    # 兜底：同名任意扩展（再筛一次图片扩展）
    cand = glob.glob(os.path.join(images_dir, stem + ".*"))
    for p in cand:
        if os.path.splitext(p)[1].lower() in IMG_EXTS and os.path.isfile(p):
            return p
    return None

def count_target_in_voc_xml(xml_path: str, target: str) -> int:
    root = ET.parse(xml_path).getroot()
    cnt = 0
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is not None and (name_node.text or "").strip() == target:
            cnt += 1
    return cnt

def main():
    xml_paths = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    if not xml_paths:
        raise RuntimeError(f"No xml files found in: {xml_dir}")

    scored = []  # (count, img_path)
    missing = 0

    for xp in xml_paths:
        stem = os.path.splitext(os.path.basename(xp))[0]
        try:
            cnt = count_target_in_voc_xml(xp, target_class)
        except Exception:
            continue

        if cnt <= 0:
            continue

        img_path = find_image_by_stem(stem)
        if img_path is None:
            missing += 1
            continue

        scored.append((cnt, img_path))

    scored.sort(key=lambda x: x[0], reverse=True)
    top5 = scored[:30]

    if not top5:
        print(f"No images found with class '{target_class}'.")
        return

    print(f"Top {len(top5)} images for class '{target_class}':")
    for cnt, imgp in top5:
        dst = os.path.join(os.getcwd(), os.path.basename(imgp))  # 当前目录，保持原名
        shutil.copy2(imgp, dst)
        print(f"  {cnt}\t{os.path.basename(imgp)} -> {dst}")

    if missing:
        print(f"[Warn] {missing} xmls have no matching image file.")

if __name__ == "__main__":
    main()
