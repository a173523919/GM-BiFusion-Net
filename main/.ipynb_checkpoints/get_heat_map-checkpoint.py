import sys
sys.path.insert(0, "/yolo")

from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ================== 你只改这里 ==================
WEIGHTS = "runs/Neck消融obbS/H2Fhead+gconv+MambaS1234/weights/best.pt"
IMAGE_RGB = "../autodl-tmp/datasets/drone/mydata/images/test/05141.jpg"
IMAGE_IR  = "../autodl-tmp/datasets/drone/mydata/image/test/05141.jpg"
OUT_PNG   = "heatmap_only.png"
IMGSZ     = 640
ALPHA     = 0.45
# =================================================


def find_last_conv2d(module):
    import torch.nn as nn
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found in model.")
    return last


def find_conv_before_detect(yolo: YOLO):
    torch_model = yolo.model
    mods = getattr(torch_model, "model", None)

    if mods is None:
        return find_last_conv2d(torch_model)

    mods = list(mods)
    for i in range(len(mods) - 1, -1, -1):
        try:
            return find_last_conv2d(mods[i])
        except Exception:
            continue

    return find_last_conv2d(torch_model)


def load_rgb_ir_to_6ch(rgb_path, ir_path, imgsz):
    bgr = cv2.imread(rgb_path)
    if bgr is None:
        raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
    rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    ir_bgr = cv2.imread(ir_path)
    if ir_bgr is None:
        raise RuntimeError(f"Failed to read IR image: {ir_path}")
    ir_rgb = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2RGB)

    rgb_rs = cv2.resize(rgb_u8, (imgsz, imgsz)).astype(np.float32) / 255.0
    ir_rs  = cv2.resize(ir_rgb, (imgsz, imgsz)).astype(np.float32) / 255.0

    x = np.concatenate([rgb_rs, ir_rs], axis=2)     # (H,W,6)
    x = np.transpose(x, (2, 0, 1))[None, ...]       # (1,6,H,W)
    return rgb_u8, torch.from_numpy(x).float()


def eigencam_from_activation(act_bchw: torch.Tensor):
    act = act_bchw[0].detach().cpu().numpy()   # (C,H,W)
    C, H, W = act.shape

    X = act.reshape(C, -1).T                  # (HW, C)
    X = X - X.mean(axis=0, keepdims=True)

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    cam = (X @ Vt[0]).reshape(H, W)

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-9)
    return cam.astype(np.float32)


def overlay_heatmap(rgb_u8, cam01, alpha):
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    heat = cv2.applyColorMap((cam01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = (1 - alpha) * rgb_f + alpha * heat
    return (out * 255).astype(np.uint8)


def main():
    for p in [WEIGHTS, IMAGE_RGB, IMAGE_IR]:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = YOLO(WEIGHTS)
    model.model.to(device).eval()

    target_layer = find_conv_before_detect(model)
    captured = {}

    def hook_fn(_, __, out):
        captured["act"] = out.detach()

    h = target_layer.register_forward_hook(hook_fn)

    rgb_u8, x6 = load_rgb_ir_to_6ch(IMAGE_RGB, IMAGE_IR, IMGSZ)
    x6 = x6.to(device)

    with torch.no_grad():
        _ = model.model(x6)

    h.remove()

    if "act" not in captured:
        raise RuntimeError("No activation captured.")

    cam_small = eigencam_from_activation(captured["act"])
    H, W = rgb_u8.shape[:2]
    cam = cv2.resize(cam_small, (W, H), interpolation=cv2.INTER_LINEAR)

    vis = overlay_heatmap(rgb_u8, cam, ALPHA)
    cv2.imwrite(OUT_PNG, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
