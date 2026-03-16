import os
import yaml
import shutil
from pathlib import Path

def process_folder(folder_path):
    try:
        # 转换为Path对象便于路径操作
        folder = Path(folder_path)
        
        # 读取args.yaml文件
        yaml_path = folder / "args.yaml"
        with open(yaml_path, 'r') as f:
            args = yaml.safe_load(f)
        
        # 获取seed和model值
        try:
            seed_value = str(args['seed'])
            model_value = Path(str(args['model'])).stem+seed_value
            
        except KeyError as e:
            print(f"错误：在YAML文件中找不到必要字段 {e}")
            return

        # 重命名CSV文件
        csv_old = folder / "results.csv"
        csv_new = folder / f"{seed_value}.csv"
        
        if not csv_old.exists():
            print(f"错误：找不到需要重命名的文件 {csv_old}")
            return
            
        # 处理目标文件已存在的情况
        if csv_new.exists():
            print(f"警告：目标文件已存在，正在覆盖 {csv_new}")
            csv_new.unlink()
            
        csv_old.rename(csv_new)
        print(f"文件已重命名：{csv_old.name} -> {csv_new.name}")

        # 重命名文件夹
        parent_dir = folder.parent
        print(parent_dir)
        new_folder = parent_dir / model_value
        
        # 检查目标文件夹是否已存在
        if new_folder.exists():
            print(f"错误：目标文件夹已存在 {new_folder}")
            return
            
        folder.rename(new_folder)
        print(f"文件夹已重命名：{folder.name} -> {model_value}")

    except Exception as e:
        print(f"操作失败：{str(e)}")

if __name__ == "__main__":
    
    #处理指定目录，可以修改为：
    ppath="runs/detect/M3FD/train"
    target_dir = ppath
    process_folder(target_dir)
    for i in range(2,7):
        target_dir = ppath+str(i)
        process_folder(target_dir)