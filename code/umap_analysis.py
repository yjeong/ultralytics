# umap_analysis.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
import numpy as np
import warnings

# FutureWarning 억제 (선택 사항)
warnings.filterwarnings("ignore", category=FutureWarning)

# 절대 경로 설정
base_dirs = [
    '/mnt/nas4/jyh/ultralytics/datasets/testset/cctv_day_cls',
    '/mnt/nas4/jyh/ultralytics/datasets/testset/cctv_night_cls',
    '/mnt/nas4/jyh/ultralytics/datasets/testset/tod_day_cls',
    '/mnt/nas4/jyh/ultralytics/datasets/testset/tod_night_cls'
]

output_dir = '/mnt/nas4/jyh/ultralytics/results/umap'
os.makedirs(output_dir, exist_ok=True)

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

# 데이터 로드 함수
def load_images(base_dir, num_classes=10, splits=['train', 'val']):
    images = []
    class_labels = []
    split_labels = []

    for split in splits:
        for class_idx in tqdm(range(num_classes), desc=f"Processing {base_dir}/{split}"):
            class_dir = os.path.join(base_dir, split, str(class_idx))
            if not os.path.isdir(class_dir):
                print(f"Directory does not exist: {class_dir}. Skipping.")
                continue
            image_files = os.listdir(class_dir)
            print(f"Processing {class_dir}, found {len(image_files)} files.")
            for img_name in image_files:
                img_path = os.path.join(class_dir, img_name)
                if not os.path.isfile(img_path):
                    print(f"Skipping non-file: {img_path}")
                    continue
                img = preprocess_image(img_path)
                if img is not None:
                    images.append(img)
                    class_labels.append(str(class_idx))
                    split_labels.append(split)

    return np.array(images), np.array(class_labels), np.array(split_labels)

# UMAP 시각화 및 저장 함수
def plot_umap(umap_embeddings, class_labels, split_labels, category_name, save_path):
    plt.figure(figsize=(12, 8))
    
    # 고유 클래스 및 스플릿 추출
    unique_classes = sorted(np.unique(class_labels))
    unique_splits = sorted(np.unique(split_labels))
    
    # 클래스별 색상 팔레트 설정
    palette = sns.color_palette("hsv", len(unique_classes))
    class_palette = {cls: palette[i] for i, cls in enumerate(unique_classes)}
    
    # 스플릿별 마커 설정
    markers = {'train': 'o', 'val': 's'}
    
    # Seaborn scatterplot 설정
    sns.scatterplot(
        x=umap_embeddings[:,0], y=umap_embeddings[:,1],
        hue=class_labels,
        palette=class_palette,
        style=split_labels,
        markers=markers,
        alpha=0.6,
        edgecolor='w',
        linewidth=0.5
    )
    
    plt.title(f"UMAP Projection - {category_name.replace('_cls', '').capitalize()} Colored by Class and Split")
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # 범례 조정
    handles, labels = plt.gca().get_legend_handles_labels()
    # 클래스와 스플릿을 분리하여 범례 생성
    class_handles = handles[:len(unique_classes)]
    class_labels_legend = labels[:len(unique_classes)]
    split_handles = handles[len(unique_classes):]
    split_labels_legend = labels[len(unique_classes):]
    
    first_legend = plt.legend(class_handles, class_labels_legend, title='Class (0-9)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.legend(split_handles, split_labels_legend, title='Split', bbox_to_anchor=(1.05, 0.6), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")

def main():
    for base_dir in base_dirs:
        category_name = os.path.basename(base_dir)  # 예: cctv_day_cls

        # 각 기본 디렉토리가 존재하는지 확인
        if not os.path.isdir(base_dir):
            print(f"Base directory does not exist: {base_dir}. Skipping.")
            continue

        # 이미지 로드
        print(f"\nLoading images for category: {category_name}")
        images, class_labels, split_labels = load_images(base_dir)
        print(f"Loaded {len(images)} images for category: {category_name}")

        if len(images) == 0:
            print(f"No images loaded for category: {category_name}. Skipping.")
            continue

        # PCA로 차원 축소 (UMAP 전에)
        desired_n_components = 50
        actual_n_components = min(desired_n_components, len(images)-1)
        if actual_n_components < 1:
            actual_n_components = 1  # 최소 1로 설정

        print(f"Performing PCA with n_components={actual_n_components} for category: {category_name}")
        pca = PCA(n_components=actual_n_components)
        try:
            images_pca = pca.fit_transform(images)
        except Exception as e:
            print(f"PCA failed for category {category_name}: {e}. Skipping UMAP.")
            continue

        # UMAP으로 차원 축소
        print(f"Performing UMAP for category: {category_name}")
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
        try:
            umap_embeddings = umap_model.fit_transform(images_pca)
        except Exception as e:
            print(f"UMAP failed for category {category_name}: {e}. Skipping plotting.")
            continue

        # UMAP 시각화 및 저장
        plot_filename = f"umap_{category_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plot_umap(
            umap_embeddings, 
            class_labels, 
            split_labels, 
            category_name, 
            plot_path
        )

    print("\nAll UMAP plots have been saved in the '/mnt/nas4/jyh/ultralytics/results/umap' directory.")

if __name__ == "__main__":
    main()
