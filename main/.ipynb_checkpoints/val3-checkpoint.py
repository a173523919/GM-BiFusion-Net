from ultralytics import YOLO
import torch
import numpy as np

def validate_model(model_path, data_yaml):
    """验证指定模型并输出详细指标"""
    model = YOLO(model_path)
    val_args = {
        'data': data_yaml,
        'conf': 0.001,
        'batch':16,
        'ch':3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    metrics = model.val(**val_args)
    
    class_map50 = {}
    class_maps = {}
    for class_id, class_name in enumerate(metrics.names.values()):
        class_map50[class_name] = metrics.box.ap50[class_id]
        class_maps[class_name] = metrics.box.maps[class_id]
    
    return metrics, model, class_map50, class_maps

if __name__ == "__main__":
    ############################## 随机种子数 #########################################
    seeds = ["1234","52319","56710","76014","856397"]
    ############################## 数据集路径 #########################################
    data_yaml = "ultralytics/cfg/datasets/mydatadronetest.yaml"
    
    # 存储结果的字典
    results = {
        "overall": {
            "map50": [],
            "map": [],
            "fps": [],
        },
        "per_class_map50": {},
        "per_class_maps": {},
    }
    
    for seed in seeds:
        ############################## 模型路径 #########################################
        model_path = "runs/单模态obb/单模态IRyolov8n" + seed + "/weights/last.pt"
        print(f"正在验证模型: {model_path}")
        
        # 验证
        metrics, model, class_map50, class_maps = validate_model(model_path, data_yaml)
        
        # 1. 提取总体指标 (保持原始float精度以便后续计算)
        map50 = metrics.box.map50
        map_all = metrics.box.map
        fps = 1000 / metrics.speed['inference']

        # 2. 存储总体指标
        results["overall"]["map50"].append(map50)
        results["overall"]["map"].append(map_all)
        results["overall"]["fps"].append(fps)
        
        # 3. 存储每个类别的mAP50
        for class_name, val in class_map50.items():
            if class_name not in results["per_class_map50"]:
                results["per_class_map50"][class_name] = []
            results["per_class_map50"][class_name].append(val)
        
        # 4. 存储每个类别的mAP50-95
        for class_name, val in class_maps.items():
            if class_name not in results["per_class_maps"]:
                results["per_class_maps"][class_name] = []
            results["per_class_maps"][class_name].append(val)
        
        print(f"模型 {model_path} 完成 -> mAP50: {map50:.4f}")

    # ====================== LaTeX 格式输出函数 ====================== #
    def get_latex_format(values, is_percent=True, digits=2):
        """
        输入: list of floats
        如果 is_percent=True，数值会乘以100
        """
        arr = np.array(values, dtype=np.float64)
        
        if is_percent:
            arr = arr * 100  # 转换为百分比 (0.9162 -> 91.62)
            
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
        
        # f-string中 {{ }} 转义为 { }，双反斜杠输出LaTeX命令
        return f"{mean:.{digits}f}{{\\scriptsize$\\pm${std:.{digits}f}}}"

    # ====================== 打印最终 LaTeX 代码 ====================== #
    print("\n" + "="*50)
    print("LaTeX 格式结果 (直接复制):")
    print("="*50)
    
    # 1. 总体指标
    # 注意：FPS 通常不需要乘 100，所以我设为 False
    print(f"FPS:       {get_latex_format(results['overall']['fps'], is_percent=False)}")
    print(f"mAP50:     {get_latex_format(results['overall']['map50'])}")
    print(f"mAP50-95:  {get_latex_format(results['overall']['map'])}")
    
    # 2. 各类别 mAP50
    print("\n--- 各类别 mAP50 ---")
    for class_name, map50_list in results["per_class_map50"].items():
        print(f"{class_name:<20}: {get_latex_format(map50_list)}")
    
    # 3. 各类别 mAP50-95
    print("\n--- 各类别 mAP50-95 ---")
    for class_name, map_list in results["per_class_maps"].items():
        print(f"{class_name:<20}: {get_latex_format(map_list)}")
