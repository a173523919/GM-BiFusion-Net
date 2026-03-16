# Dataset Preprocessing Instructions / 数据集预处理说明

This document describes the purpose and processing workflow of the dataset preprocessing scripts.  
本文说明数据集预处理脚本的作用及处理流程。

---

## 1. `remove_img_blank.py`

### English
`remove_img_blank.py` is used to remove the white borders around the original dataset images.  
Some images in the raw dataset contain blank white margins around the valid image content. These margins do not provide useful information for training, so they should be removed before annotation conversion and model training.

### 中文
`remove_img_blank.py` 用于移除原始数据集图像四周的白边。  
原始数据集中部分图像在有效内容区域外围存在空白白边，这些区域对模型训练没有实际作用，因此需要在标注转换和模型训练之前将其移除。

---

## 2. Annotation Conversion / 标注转换

After removing the white borders, the original XML annotations should be converted into YOLO format.  
移除白边后，需要将原始数据集的 XML 标注转换为 YOLO 格式。

### 2.1 `RBxml2yoloHBB.py`

#### English
`RBxml2yoloHBB.py` is used to convert the original XML annotations into YOLO HBB (Horizontal Bounding Box) format.

#### 中文
`RBxml2yoloHBB.py` 用于将原始数据集的 XML 标注转换为 YOLO HBB（水平边界框）格式。

---

### 2.2 `RBxml2yoloOBB.py`

#### English
`RBxml2yoloOBB.py` is used to convert the original XML annotations into YOLO OBB (Oriented Bounding Box) format.

#### 中文
`RBxml2yoloOBB.py` 用于将原始数据集的 XML 标注转换为 YOLO OBB（旋转边界框）格式。

---

## 3. Recommended Workflow / 推荐流程

### English
1. Run `remove_img_blank.py` to remove the white borders around the original dataset images.  
2. After border removal, use `RBxml2yoloHBB.py` to convert the XML annotations into YOLO HBB format.  
3. After border removal, use `RBxml2yoloOBB.py` to convert the XML annotations into YOLO OBB format.  

### 中文
1. 运行 `remove_img_blank.py`，移除原始数据集图像四周的白边。  
2. 去除白边后，使用 `RBxml2yoloHBB.py` 将原始数据集的 XML 标注转换为 YOLO HBB 格式。  
3. 去除白边后，使用 `RBxml2yoloOBB.py` 将原始数据集的 XML 标注转换为 YOLO OBB 格式。  

---

## 4. Summary / 总结

| Script | English Description | 中文说明 |
|---|---|---|
| `remove_img_blank.py` | Remove white borders around original images | 移除原始图像四周白边 |
| `RBxml2yoloHBB.py` | Convert XML annotations to YOLO HBB format | 将 XML 标注转换为 YOLO HBB 格式 |
| `RBxml2yoloOBB.py` | Convert XML annotations to YOLO OBB format | 将 XML 标注转换为 YOLO OBB 格式 |

---

## 5. Note / 备注

### English
The annotation conversion should be performed after the image border removal step, so that the processed images and the converted labels remain consistent.

### 中文
标注转换应在图像去白边之后进行，以保证处理后的图像与转换后的标注保持一致。
