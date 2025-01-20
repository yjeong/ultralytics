# basic_stats.py

import os
import yaml
from PIL import Image
import pandas as pd

# YAML 파일 파싱
def parse_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data

# 이미지 크기 통계
def image_size_stats(directory):
    sizes = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        sizes.append(img.size)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    df = pd.DataFrame(sizes, columns=['Width', 'Height'])
    return df.describe()

# 데이터셋 경로 설정
yaml_paths = [
    '/mnt/nas4/jyh/ultralytics/code/work_cd.yaml',
    '/mnt/nas4/jyh/ultralytics/code/work_cn.yaml',
]

for yaml_file in yaml_paths:
    data = parse_yaml(yaml_file)
    train_dir = data['train']
    val_dir = data['val']
    
    print(f"\nDataset: {yaml_file}")
    print("Training Image Size Stats:")
    print(image_size_stats(train_dir))
    
    print("\nValidation Image Size Stats:")
    print(image_size_stats(val_dir))
