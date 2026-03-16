# GM-BiFusion-Net

## 0. Installation Requirement

Before running the project successfully, make sure the required environment and dependencies are installed correctly.

Reference repository:  
- [VisionMamba](https://github.com/VisionMamba/)

---

## 1. Training Scripts

Multi-random-seed sequential training is supported with the following scripts:

- `maindrone1.py`  
  Suitable for the **Drone Vehicle** dataset.

- `mainM3FD.py`  
  Suitable for the **M3FD** dataset.

---

## 2. Validation Scripts

The following scripts are used to batch-validate all trained models and obtain performance metrics:

- `val.py`
- `valM3FD.py`

### Notes
- These scripts only accept **RGB-IR paired data** as input.
- When validating or training on a **single-modality dataset**, modify `ch` in `ultralytics/cfg/default.yaml` from `6` to `3`.

Example:
```yaml
ch: 3
```

---

## 3. Visualization Scripts

The following scripts are used for batch visualization to obtain the actual detection results of the models on images:

- `vis_droneVichel.py`
- `vis_M3FD.py`

### Notes
- These scripts only accept **RGB-IR paired data** as input.
- They output **RGB and IR dual-modality detection results**.

For single-modality visualization:

- `vis_AIDOT.py`

### Notes
- This script only accepts **single-modality data** as input.
- It outputs **single-modality visualization results**.

### Troubleshooting for Single-Modality Visualization
If visualization on a single-modality dataset fails, try modifying the following line in:

`ultralytics/engine/predictor.py`

Original:
```python
self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 6, *self.imgsz))
```

Change to:
```python
self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
```

For dual-modality input, change it back from `3` to `6` if necessary.

---

## 4. Dataset Organization

Dataset directory structure:

```text
images/train or val or test
image/train or val or test
```

### Description
- `images/` stores **RGB data**
- `image/` stores **IR data**

### Notes for Single-Modality Data
- For a single-modality dataset, the `image` folder should be disabled.
- Do **not** disable the `images` folder alone.

The main data loading logic is located in:

```text
ultralytics/data/base.py
```

---

## 5. Public Dataset Sources

Original public dataset links:

- [DroneVehicle](https://github.com/VisDrone/DroneVehicle)
- [TarDAL](https://github.com/JinyuanLiu-CV/TarDAL)

---

## 6. Acknowledgements

Special thanks to the following works:

```bibtex
@inproceedings{liu2022target,
  title={Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection},
  author={Liu, Jinyuan and Fan, Xin and Huang, Zhanbo and Wu, Guanyao and Liu, Risheng and Zhong, Wei and Luo, Zhongxuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5802--5811},
  year={2022}
}

@ARTICLE{sun2020drone,
  title={Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning},
  author={Sun, Yiming and Cao, Bing and Zhu, Pengfei and Hu, Qinghua},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3168279}
}

@article{liu2024vmamba,
  title={VMamba: Visual State Space Model},
  author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Liu, Yunfan},
  journal={arXiv preprint arXiv:2401.10166},
  year={2024}
}
```
