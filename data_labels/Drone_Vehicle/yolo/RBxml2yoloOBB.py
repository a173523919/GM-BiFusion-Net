import os
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
from collections import defaultdict

# ================= 配置参数 =================
CROP_AREA = (100, 100, 740, 612)  # (x_min, y_min, x_max, y_max)
FINAL_CLASSES = ['bus', 'car', 'van', 'feright_car', 'truck']
LABEL_MAPPING = {
    'bus': 'bus',
    'car': 'car',
    'van': 'van',
    'feright': 'feright_car',
    'feright car': 'feright_car',
    'feright_car': 'feright_car',
    'truck': 'truck',
    'truvk': 'truck',
    '*': None
}
IMAGE_EXTS = ['.jpg', '.jpeg', '.png']

# ================= 路径配置 =================
XML_DIR = 'RGBxmllabels/val'
PROCESSED_IMG_DIR = 'mydata/images/val'
OUTPUT_LABEL_DIR = 'obbRGBlabels/val'

# ================= 核心函数 =================
class LabelProcessor:
    def __init__(self):
        self.label_stats = defaultdict(int)
        self.error_log = []
        
    def find_image_file(self, base_name):
        for ext in IMAGE_EXTS:
            img_path = os.path.join(PROCESSED_IMG_DIR, f"{base_name}{ext}")
            if os.path.exists(img_path):
                return img_path
        self.error_log.append(f"Image not found: {base_name}")
        return None

    def get_normalized_points(self, points, crop_w, crop_h):
        """归一化多边形顶点坐标并确保在[0,1]范围内"""
        normalized = []
        for x, y in points:
            nx = (x - CROP_AREA[0]) / crop_w
            ny = (y - CROP_AREA[1]) / crop_h
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            normalized.extend([nx, ny])
        return normalized

    def process_xml(self, xml_path):
        try:
            base_name = os.path.splitext(os.path.basename(xml_path))[0]
            img_path = self.find_image_file(base_name)
            if not img_path:
                return None
                
            # 获取图像尺寸
            img = cv2.imread(img_path)
            if img is None:
                self.error_log.append(f"Invalid image: {img_path}")
                return None
                
            # 解析XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 计算裁剪区域尺寸
            crop_w = CROP_AREA[2] - CROP_AREA[0]
            crop_h = CROP_AREA[3] - CROP_AREA[1]
            
            valid_annotations = []
            for obj in root.findall('object'):
                # 标签处理
                raw_label = obj.find('name').text.strip().lower()
                mapped_label = LABEL_MAPPING.get(raw_label)
                
                if not mapped_label or mapped_label not in FINAL_CLASSES:
                    continue
                    
                # 获取多边形或边界框的点
                points = []
                polygon = obj.find('polygon')
                if polygon is not None:
                    # 处理多边形标注
                    for i in range(1, 5):
                        x = int(polygon.find(f'x{i}').text)
                        y = int(polygon.find(f'y{i}').text)
                        points.append((x, y))
                else:
                    # 处理矩形框标注 (转换为4个角点)
                    bndbox = obj.find('bndbox')
                    if bndbox is not None:
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)
                        points = [
                            (xmin, ymin),  # 左上
                            (xmax, ymin),  # 右上
                            (xmax, ymax),  # 右下
                            (xmin, ymax)   # 左下
                        ]
                
                if not points:
                    continue
                    
                # 归一化坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
                normalized_points = self.get_normalized_points(points, crop_w, crop_h)
                
                # 记录统计信息
                class_id = FINAL_CLASSES.index(mapped_label)
                self.label_stats[mapped_label] += 1
                
                # 格式化为OBB标签行: class_id x1 y1 x2 y2 x3 y3 x4 y4
                points_str = " ".join([f"{coord:.6f}" for coord in normalized_points])
                valid_annotations.append(f"{class_id} {points_str}")
                    
            return valid_annotations
            
        except Exception as e:
            self.error_log.append(f"{xml_path}: {str(e)}")
            return None

# ================= 主流程 =================
def main():
    processor = LabelProcessor()
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    
    # 处理所有XML文件
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
    print(f"Processing {len(xml_files)} XML files for OBB labels...")
    
    for xml_file in tqdm(xml_files):
        xml_path = os.path.join(XML_DIR, xml_file)
        annotations = processor.process_xml(xml_path)
        
        if annotations:
            output_path = os.path.join(OUTPUT_LABEL_DIR, f"{os.path.splitext(xml_file)[0]}.txt")
            with open(output_path, 'w') as f:
                f.write('\n'.join(annotations))
    
    # 显示统计结果
    print("\n=== 类别分布统计 ===")
    print(f"{'类别名称':<12} | {'编号':<4} | 出现次数")
    print("-"*30)
    for idx, name in enumerate(FINAL_CLASSES):
        count = processor.label_stats.get(name, 0)
        print(f"{name:<12} | {idx:<4} | {count}")
    
    # 显示错误日志
    if processor.error_log:
        print("\n=== 处理异常 ===")
        for error in processor.error_log[:5]:  # 最多显示5条错误
            print(f"• {error}")
        if len(processor.error_log) > 5:
            print(f"（其余 {len(processor.error_log)-5} 条错误未显示）")

if __name__ == '__main__':
    main()