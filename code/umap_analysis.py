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
from matplotlib.patches import Ellipse

# FutureWarning 억제 (선택 사항)
warnings.filterwarnings("ignore", category=FutureWarning)

# 절대 경로 설정
base_dirs = [
    '/mnt/nas4/jyh/ultralytics/datasets/coco8',
    '/mnt/nas4/jyh/ultralytics/datasets/coco8_cls'
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
    category_labels = []

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
                    category_labels.append(os.path.basename(base_dir))  # 카테고리 이름 저장

    return np.array(images), np.array(class_labels), np.array(split_labels), np.array(category_labels)

# 타원을 그리는 헬퍼 함수
def plot_cov_ellipse(mean, cov, ax, n_std=2.0, edgecolor='black', linewidth=2, linestyle='-', **kwargs):
    """
    mean: 중심 좌표 (x, y)
    cov: 2x2 공분산 행렬
    ax: matplotlib axes 객체
    n_std: 표준 편차의 배수 (타원의 크기 조절)
    edgecolor: 타원의 테두리 색상
    linewidth: 타원의 테두리 두께
    linestyle: 타원의 테두리 선 스타일
    kwargs: 추가적인 Ellipse 속성
    """
    # 고유값과 고유 벡터 계산
    eigenvals, eigenvecs = np.linalg.eigh(cov)

    # 고유값이 큰 순서대로 정렬
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    # 고유 벡터의 각도 계산
    angle = np.degrees(np.arctan2(*eigenvecs[:,0][::-1]))

    # 타원의 크기 계산 (표준 편차의 배수)
    width, height = 2 * n_std * np.sqrt(eigenvals)

    # 타원 생성
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, facecolor='none', **kwargs)

    ax.add_patch(ellipse)

# UMAP 시각화 및 저장 함수 (개별 카테고리용)
def plot_umap_individual(umap_embeddings, class_labels, split_labels, category_name, save_path):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 고유 클래스 및 스플릿 추출
    unique_classes = sorted(np.unique(class_labels))
    unique_splits = sorted(np.unique(split_labels))

    # 클래스별 색상 팔레트 설정
    palette = sns.color_palette("hsv", len(unique_classes))
    class_palette = {cls: palette[i] for i, cls in enumerate(unique_classes)}

    # 스플릿별 마커 설정
    markers = {'train': 'o', 'val': 's'}

    # Seaborn scatterplot 설정
    scatter = sns.scatterplot(
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
    handles, labels = scatter.get_legend_handles_labels()
    # 클래스와 스플릿을 분리하여 범례 생성
    n_classes = len(unique_classes)
    class_handles = handles[:n_classes]
    class_labels_legend = labels[:n_classes]
    split_handles = handles[n_classes:]
    split_labels_legend = labels[n_classes:]

    first_legend = plt.legend(class_handles, class_labels_legend, title='Class (0-9)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.legend(split_handles, split_labels_legend, title='Split', bbox_to_anchor=(1.05, 0.6), loc='upper left')

    # 클래스별 타원 그리기
    for cls in unique_classes:
        # 해당 클래스의 인덱스
        idx = class_labels == cls
        if np.sum(idx) < 2:
            continue  # 최소 2개의 점 필요
        points = umap_embeddings[idx]
        mean = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        plot_cov_ellipse(mean, cov, ax, n_std=2.0, edgecolor=class_palette[cls], linewidth=2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved individual plot to {save_path}")

# UMAP 시각화 및 저장 함수 (통합 카테고리용)
def plot_umap_combined(umap_embeddings, category_labels, category_names, save_path):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # 고유 카테고리 추출
    unique_categories = sorted(np.unique(category_labels))

    # 카테고리별 색상 팔레트 설정
    palette = sns.color_palette("Set1", len(unique_categories))
    category_palette = {category: palette[i] for i, category in enumerate(unique_categories)}

    # Seaborn scatterplot 설정
    scatter = sns.scatterplot(
        x=umap_embeddings[:,0], y=umap_embeddings[:,1],
        hue=category_labels,
        palette=category_palette,
        alpha=0.6,
        edgecolor='w',
        linewidth=0.5
    )

    plt.title("UMAP Projection - Combined CCTV Day & Night Colored by Category")
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # 범례 조정
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles, labels, title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 카테고리별 타원 그리기
    for category in unique_categories:
        # 해당 카테고리의 인덱스
        idx = category_labels == category
        if np.sum(idx) < 2:
            continue  # 최소 2개의 점 필요
        points = umap_embeddings[idx]
        mean = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        plot_cov_ellipse(mean, cov, ax, n_std=2.0, edgecolor=category_palette[category], linewidth=2)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined plot to {save_path}")

def main():
    # 개별 카테고리 처리
    for base_dir in base_dirs:
        category_name = os.path.basename(base_dir)  # 예: cctv_day_cls

        # 각 기본 디렉토리가 존재하는지 확인
        if not os.path.isdir(base_dir):
            print(f"Base directory does not exist: {base_dir}. Skipping.")
            continue

        # 이미지 로드
        print(f"\nLoading images for category: {category_name}")
        images, class_labels, split_labels, category_labels = load_images(base_dir)
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

        # UMAP 시각화 및 저장 (개별 카테고리)
        plot_filename = f"umap_{category_name}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plot_umap_individual(
            umap_embeddings, 
            class_labels, 
            split_labels, 
            category_name, 
            plot_path
        )

    # 통합 카테고리 처리 (cctv_day_cls와 cctv_night_cls를 함께)
    combined_images = []
    combined_category_labels = []

    for base_dir in base_dirs:
        category_name = os.path.basename(base_dir)  # 예: cctv_day_cls

        # 각 기본 디렉토리가 존재하는지 확인
        if not os.path.isdir(base_dir):
            print(f"Base directory does not exist: {base_dir}. Skipping.")
            continue

        # 이미지 로드
        print(f"\nLoading images for combined category: {category_name}")
        images, _, _, category_labels = load_images(base_dir)
        print(f"Loaded {len(images)} images for combined category: {category_name}")

        if len(images) == 0:
            print(f"No images loaded for combined category: {category_name}. Skipping.")
            continue

        combined_images.append(images)
        combined_category_labels.append(category_labels)

    if combined_images:
        combined_images = np.concatenate(combined_images, axis=0)
        combined_category_labels = np.concatenate(combined_category_labels, axis=0)

        print(f"\nPerforming PCA for combined categories")
        desired_n_components = 50
        actual_n_components = min(desired_n_components, len(combined_images)-1)
        if actual_n_components < 1:
            actual_n_components = 1  # 최소 1로 설정

        pca = PCA(n_components=actual_n_components)
        try:
            combined_pca = pca.fit_transform(combined_images)
        except Exception as e:
            print(f"PCA failed for combined categories: {e}. Skipping UMAP.")
            combined_pca = None

        if combined_pca is not None:
            print(f"Performing UMAP for combined categories")
            umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
            try:
                combined_umap = umap_model.fit_transform(combined_pca)
            except Exception as e:
                print(f"UMAP failed for combined categories: {e}. Skipping plotting.")
                combined_umap = None

            if combined_umap is not None:
                # UMAP 시각화 및 저장 (통합 카테고리)
                combined_plot_filename = "umap_combined_cctv_day_night_cls.png"
                combined_plot_path = os.path.join(output_dir, combined_plot_filename)
                plot_umap_combined(
                    combined_umap, 
                    combined_category_labels, 
                    base_dirs, 
                    combined_plot_path
                )

    print("\nAll UMAP plots have been saved in the '/mnt/nas4/jyh/ultralytics/results/umap' directory.")

if __name__ == "__main__":
    main()
