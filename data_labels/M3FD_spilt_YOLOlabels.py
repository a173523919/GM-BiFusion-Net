import shutil
import random
from pathlib import Path

# ------------------------
# Config (按需改路径)
# ------------------------
SRC_RGB_DIR   = Path("o_images")   # RGB
SRC_TIR_DIR   = Path("o_image")    # TIR
SRC_LABEL_DIR = Path("o_labels")   # YOLO labels (.txt)

TRAIN_RATIO = 0.8
SEED = 1234

DST_ROOT = Path( ("dataset_yolo"+str(SEED)) )    # 输出根目录（自己改）

# YOLO-style split folders
DST_RGB_TRAIN = DST_ROOT /"images"/ "train"
DST_RGB_VAL   = DST_ROOT /"images"/ "test"

DST_TIR_TRAIN = DST_ROOT /"image"/ "train"
DST_TIR_VAL   = DST_ROOT /"image"/  "test"

DST_LAB_TRAIN = DST_ROOT / "labels" / "train"
DST_LAB_VAL   = DST_ROOT / "labels" / "test"



# 建议默认 False：不破坏源数据，可重复运行
MOVE_LABELS = False

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def ensure_dirs():
    for d in [DST_RGB_TRAIN, DST_RGB_VAL, DST_TIR_TRAIN, DST_TIR_VAL, DST_LAB_TRAIN, DST_LAB_VAL]:
        d.mkdir(parents=True, exist_ok=True)


def find_image_by_stem(folder: Path, stem: str):
    """在 folder 下找与 stem 同名的图片（常见后缀），找到就返回 Path，否则 None"""
    if not folder.exists():
        return None

    for ext in IMG_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p

    # 兜底：扫描（更慢，但兼容奇怪后缀）
    for p in folder.iterdir():
        if p.is_file() and p.stem == stem:
            return p
    return None


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 目标存在则避免覆盖
    if dst.exists():
        base = dst.stem
        ext = dst.suffix
        parent = dst.parent
        i = 1
        while True:
            cand = parent / f"{base}_dup{i}{ext}"
            if not cand.exists():
                dst = cand
                break
            i += 1
    shutil.move(str(src), str(dst))
    return dst


def main():
    ensure_dirs()

    if not SRC_LABEL_DIR.exists():
        raise FileNotFoundError(f"Label folder not found: {SRC_LABEL_DIR.resolve()}")

    # 以 label 为样本基准
    label_files = sorted([p for p in SRC_LABEL_DIR.iterdir()
                          if p.is_file() and p.suffix.lower() == ".txt"])
    stems = [p.stem for p in label_files]

    rng = random.Random(SEED)
    rng.shuffle(stems)

    n_total = len(stems)
    n_train = int(n_total * TRAIN_RATIO)

    train_stems = set(stems[:n_train])
    val_stems   = set(stems[n_train:])

    print(f"Total samples (by labels): {n_total}")
    print(f"Train: {len(train_stems)} | Val: {len(val_stems)} | Seed: {SEED}")
    print(f"Names kept identical across images/image/labels. MOVE_LABELS={MOVE_LABELS}")

    missing_rgb, missing_tir, missing_lab = [], [], []
    copied_rgb = copied_tir = copied_lab = 0

    def process_split(split_stems: set, dst_rgb: Path, dst_tir: Path, dst_lab: Path):
        nonlocal copied_rgb, copied_tir, copied_lab

        for stem in sorted(split_stems):
            # 1) label
            src_label = SRC_LABEL_DIR / f"{stem}.txt"
            if not src_label.exists():
                missing_lab.append(stem)
            else:
                dst_label = dst_lab / src_label.name
                if MOVE_LABELS:
                    safe_move(src_label, dst_label)
                else:
                    safe_copy(src_label, dst_label)
                copied_lab += 1

            # 2) RGB image (保持文件名不变)
            rgb_img = find_image_by_stem(SRC_RGB_DIR, stem)
            if rgb_img is None:
                missing_rgb.append(stem)
            else:
                safe_copy(rgb_img, dst_rgb / rgb_img.name)
                copied_rgb += 1

            # 3) TIR image (保持文件名不变)
            tir_img = find_image_by_stem(SRC_TIR_DIR, stem)
            if tir_img is None:
                missing_tir.append(stem)
            else:
                safe_copy(tir_img, dst_tir / tir_img.name)
                copied_tir += 1

    process_split(train_stems, DST_RGB_TRAIN, DST_TIR_TRAIN, DST_LAB_TRAIN)
    process_split(val_stems,   DST_RGB_VAL,   DST_TIR_VAL,   DST_LAB_VAL)

    print("\nDone.")
    print(f"Copied labels: {copied_lab}")
    print(f"Copied RGB images: {copied_rgb}")
    print(f"Copied TIR images: {copied_tir}")

    if missing_lab:
        print(f"\n[WARN] Missing labels for {len(missing_lab)} stems. Examples: {missing_lab[:10]}")
    if missing_rgb:
        print(f"\n[WARN] Missing RGB images for {len(missing_rgb)} stems. Examples: {missing_rgb[:10]}")
    if missing_tir:
        print(f"\n[WARN] Missing TIR images for {len(missing_tir)} stems. Examples: {missing_tir[:10]}")

    print("\nOutput folders:")
    print(f"  RGB train:   {DST_RGB_TRAIN.resolve()}")
    print(f"  RGB val:     {DST_RGB_VAL.resolve()}")
    print(f"  TIR train:   {DST_TIR_TRAIN.resolve()}")
    print(f"  TIR val:     {DST_TIR_VAL.resolve()}")
    print(f"  Labels train:{DST_LAB_TRAIN.resolve()}")
    print(f"  Labels val:  {DST_LAB_VAL.resolve()}")


if __name__ == "__main__":
    main()
