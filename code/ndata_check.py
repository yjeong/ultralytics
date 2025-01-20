# data_check.py

import os
import yaml

# YAML 파일 파싱
def parse_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data

# 이미지와 레이블 매칭 수 확인
def check_image_label_matching(yaml_path):
    data = parse_yaml(yaml_path)
    train_dirs = data['train']
    val_dirs = data['val']
    
    # train_dirs와 val_dirs가 리스트인지 확인
    if not isinstance(train_dirs, list):
        train_dirs = [train_dirs]
    if not isinstance(val_dirs, list):
        val_dirs = [val_dirs]
    
    #이미지 카운트
    def count_images(directories, dataset_type):
        total = 0
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        for directory in directories:
            print(f"Processing {dataset_type} directory: {directory}")
            if not os.path.isdir(directory):
                print(f"Warning: Directory {directory} does not exist.")
                continue
            items = os.listdir(directory)
            if not items:
                print(f"Warning: No items found in {directory}.")
                continue
            # Check if the first item is a directory to determine structure
            first_item_path = os.path.join(directory, items[0])
            if os.path.isdir(first_item_path):
                # Assume class subdirectories exist
                for class_name in items:
                    class_dir = os.path.join(directory, class_name)
                    if not os.path.isdir(class_dir):
                        print(f"Skipping {class_dir} as it is not a directory.")
                        continue
                    images = [f for f in os.listdir(class_dir) if os.path.splitext(f)[1].lower() in supported_extensions]
                    image_count = len(images)
                    total += image_count
                    print(f"  Class '{class_name}': {image_count} images")
            else:
                # Assume images are directly inside the directory
                images = [f for f in items if os.path.splitext(f)[1].lower() in supported_extensions]
                image_count = len(images)
                total += image_count
                print(f"  {dataset_type.capitalize()} directory contains {image_count} images directly")
        return total
    
    train_count = count_images(train_dirs, "train")
    val_count = count_images(val_dirs, "validation")
    
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")

# 데이터셋 경로 설정
yaml_paths = [
    '/mnt/nas4/jyh/ultralytics/code/work_cd.yaml',
    '/mnt/nas4/jyh/ultralytics/code/work_cn.yaml'
]

for yaml_file in yaml_paths:
    print(f"\nChecking dataset: {yaml_file}")
    check_image_label_matching(yaml_file)
