# umap_analysis.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
import yaml
import numpy as np
import warnings

# FutureWarning 억제 (선택 사항)
warnings.filterwarnings("ignore", category=FutureWarning)

# 데이터셋 YAML 파일 목록 정의 (절대 경로 사용)
yaml_files = [
    '/mnt/nas4/jyh/ultralytics/code/work_cd.yaml',
    '/mnt/nas4/jyh/ultralytics/code/work_cn.yaml',
    '/mnt/nas4/jyh/ultralytics/code/work_td.yaml',
    '/mnt/nas4/jyh/ultralytics/code/work_tn.yaml'  # work_te.yaml로 가정
]

# 결과를 저장할 디렉토리 생성
output_dir = 'result'
os.makedirs(output_dir, exist_ok=True)

# YAML 파일 파싱 함수
def parse_yaml(yaml_file):
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        print(f"Error parsing YAML file {yaml_file}: {e}")
        return None

# 이미지 전처리 함수
def preprocess_image(image_path, size=(224, 224)):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(size)
        image = np.array(image) / 255.0  # 정규화
        image = image.flatten()
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# 데이터 로드 함수 (단일 클래스 처리)
def load_images(image_dirs, split_name='train'):
    images = []
    labels = []
    label = 0  # 모든 이미지를 단일 클래스로 처리
    for image_dir in image_dirs:
        if not os.path.isdir(image_dir):
            print(f"[{split_name}] Directory does not exist: {image_dir}")
            continue
        image_files = os.listdir(image_dir)
        print(f"[{split_name}] Processing directory: {image_dir}, found {len(image_files)} files.")
        for img_name in image_files:
            img_path = os.path.join(image_dir, img_name)
            if not os.path.isfile(img_path):
                print(f"[{split_name}] Skipping non-file: {img_path}")
                continue
            img = preprocess_image(img_path)
            if img is not None:
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# UMAP 시각화 및 저장 함수
def plot_umap(umap_embeddings, labels, title, save_path):
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("hsv", len(np.unique(labels)))
    sns.scatterplot(
        x=umap_embeddings[:,0], y=umap_embeddings[:,1],
        hue=labels,
        palette=palette,
        legend='full',
        alpha=0.3
    )
    plt.title(title)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

# 각 YAML 파일에 대해 처리 수행
for yaml_file in yaml_files:
    print(f"\nProcessing dataset: {yaml_file}")
    
    # YAML 파일 존재 여부 확인
    if not os.path.isfile(yaml_file):
        print(f"YAML file does not exist: {yaml_file}. Skipping.")
        continue
    
    # YAML 파일 파싱
    data = parse_yaml(yaml_file)
    if data is None:
        print(f"Failed to parse YAML file: {yaml_file}. Skipping.")
        continue
    
    train_dirs = data.get('train', [])
    val_dirs = data.get('val', [])
    
    # train_dirs와 val_dirs가 리스트인지 확인하고, 아니면 리스트로 변환
    if not isinstance(train_dirs, list):
        train_dirs = [train_dirs]
    if not isinstance(val_dirs, list):
        val_dirs = [val_dirs]
    
    print(f"Train directories: {train_dirs}")
    print(f"Validation directories: {val_dirs}")
    
    # 훈련 이미지 로드
    print("Loading training images...")
    train_images, train_labels = load_images(train_dirs, split_name='train')
    print(f"Loaded {len(train_images)} training images.")
    
    # 검증 이미지 로드
    print("Loading validation images...")
    val_images, val_labels = load_images(val_dirs, split_name='val')
    print(f"Loaded {len(val_images)} validation images.")
    
    # 클래스 매핑 출력 (단일 클래스이므로 고정)
    print("Class to label mapping:")
    print("  single_class: 0")
    
    # PCA로 차원 축소 (UMAP 전에)
    if len(train_images) == 0:
        print(f"No training images loaded for dataset {yaml_file}. Skipping PCA and UMAP.")
        continue
    
    # 동적으로 n_components 설정
    desired_n_components = 50
    actual_n_components = min(desired_n_components, len(train_images)-1)
    if actual_n_components < 1:
        actual_n_components = 1  # 최소 1로 설정
    
    print(f"Performing PCA with n_components={actual_n_components}...")
    pca = PCA(n_components=actual_n_components)
    try:
        train_pca = pca.fit_transform(train_images)
        val_pca = pca.transform(val_images)
    except Exception as e:
        print(f"PCA failed for dataset {yaml_file}: {e}. Skipping PCA and UMAP.")
        continue
    
    # UMAP으로 차원 축소
    print("Performing UMAP...")
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    try:
        train_umap = umap_model.fit_transform(train_pca)
        val_umap = umap_model.transform(val_pca)
    except Exception as e:
        print(f"UMAP failed for dataset {yaml_file}: {e}. Skipping UMAP.")
        continue
    
    # 결과를 저장할 디렉토리 생성
    output_dir = 'results/umap'
    os.makedirs(output_dir, exist_ok=True)

    # UMAP 시각화 및 저장
    dataset_name = os.path.splitext(os.path.basename(yaml_file))[0]  # 'work_cd', etc.
    
    print("Plotting UMAP for training data...")
    train_plot_path = os.path.join(output_dir, f"umap_training_{dataset_name}.png")
    plot_umap(train_umap, train_labels, f"UMAP Projection - Training Data ({dataset_name})", train_plot_path)
    
    print("Plotting UMAP for validation data...")
    val_plot_path = os.path.join(output_dir, f"umap_validation_{dataset_name}.png")
    plot_umap(val_umap, val_labels, f"UMAP Projection - Validation Data ({dataset_name})", val_plot_path)

print("\nAll datasets processed. UMAP plots are saved in the 'result' directory.")