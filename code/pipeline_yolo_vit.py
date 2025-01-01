# pipeline_yolo_vit.py

import os
import cv2
from ultralytics import YOLO
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import pandas as pd

def load_vit_model(model_name='google/vit-base-patch16-224', device='cpu'):
    """
    사전 학습된 ViT 모델과 feature extractor를 로드합니다.
    
    Parameters:
    - model_name: Hugging Face의 ViT 모델 이름
    - device: 모델을 실행할 디바이스 ('cpu' 또는 'cuda')
    
    Returns:
    - model: ViT 이미지 분류 모델
    - feature_extractor: 이미지 전처리 도구
    """
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    model.eval()  # 평가 모드로 전환
    return model, feature_extractor

def classify_with_vit(model, feature_extractor, image_path, device='cpu'):
    """
    ViT 모델을 사용하여 패칭된 객체 이미지를 분류합니다.
    
    Parameters:
    - model: ViT 이미지 분류 모델
    - feature_extractor: 이미지 전처리 도구
    - image_path: 분류할 객체 이미지 경로
    - device: 모델을 실행할 디바이스 ('cpu' 또는 'cuda')
    
    Returns:
    - prediction: 예측된 클래스 라벨
    - score: 해당 클래스의 확률 점수
    """
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    score = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
    
    return model.config.id2label[predicted_class_idx], score

def batch_classify_vit(model, feature_extractor, crop_dir, output_csv, device='cpu'):
    """
    패칭된 객체 이미지를 순회하며 ViT 모델로 분류하고, 결과를 CSV 파일로 저장합니다.
    
    Parameters:
    - model: ViT 이미지 분류 모델
    - feature_extractor: 이미지 전처리 도구
    - crop_dir: 패칭된 객체 이미지가 저장된 클래스별 디렉토리
    - output_csv: 분류 결과를 저장할 CSV 파일 경로
    - device: 모델을 실행할 디바이스 ('cpu' 또는 'cuda')
    
    Returns:
    - y_true: 실제 클래스 리스트
    - y_pred: 예측 클래스 리스트
    """
    class_folders = [f for f in os.listdir(crop_dir) if os.path.isdir(os.path.join(crop_dir, f))]
    results = []
    y_true = []
    y_pred = []
    
    for class_folder in class_folders:
        class_path = os.path.join(crop_dir, class_folder)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            prediction, score = classify_with_vit(model, feature_extractor, img_path, device)
            results.append({
                'image_path': img_path,
                'true_class': class_folder,
                'predicted_class': prediction,
                'score': score
            })
            y_true.append(class_folder)
            y_pred.append(prediction)
            print(f"[INFO] Classified {img_path}: {prediction} ({score:.4f})")
    
    # CSV 저장
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_path', 'true_class', 'predicted_class', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"[INFO] Classification results saved to {output_csv}")
    
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, classes, output_path, normalize=False):
    """
    혼동 행렬을 계산하고 시각화합니다.
    
    Parameters:
    - y_true: 실제 클래스 리스트
    - y_pred: 예측 클래스 리스트
    - classes: 클래스 이름 리스트
    - output_path: 혼동 행렬 이미지를 저장할 경로
    - normalize: 정규화 여부 (기본 False)
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    title = 'Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] {title} saved to {output_path}")

def plot_precision_recall(y_true, y_pred, classes, output_dir):
    """
    각 클래스별 Precision-Recall 곡선을 시각화하고 저장합니다.
    
    Parameters:
    - y_true: 실제 클래스 리스트
    - y_pred: 예측 클래스 리스트
    - classes: 클래스 이름 리스트
    - output_dir: PR 곡선을 저장할 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    y_true_bin = label_binarize(y_true, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)
    n_classes = len(classes)
    
    for i, cls in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
        pr_auc = auc(recall, precision)
        
        plt.figure()
        plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {cls}')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'PR_curve_{cls}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Precision-Recall curve for {cls} saved to {plot_path}")

def plot_roc_curve(y_true, y_pred, classes, output_dir):
    """
    각 클래스별 ROC 곡선을 시각화하고 저장합니다.
    
    Parameters:
    - y_true: 실제 클래스 리스트
    - y_pred: 예측 클래스 리스트
    - classes: 클래스 이름 리스트
    - output_dir: ROC 곡선을 저장할 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    y_true_bin = label_binarize(y_true, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)
    n_classes = len(classes)
    
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # 랜덤 추측선
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {cls}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'ROC_curve_{cls}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] ROC curve for {cls} saved to {plot_path}")

def plot_labels_distribution(y_true, classes, output_path):
    """
    클래스별 레이블 분포를 시각화하고 저장합니다.
    
    Parameters:
    - y_true: 실제 클래스 리스트
    - classes: 클래스 이름 리스트
    - output_path: 분포 그래프를 저장할 경로
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_true, order=classes)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Labels Distribution')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Labels distribution plot saved to {output_path}")

def plot_labels_correlogram(y_true, classes, output_path):
    """
    레이블 간 상관 관계를 시각화하고 저장합니다.
    
    Parameters:
    - y_true: 실제 클래스 리스트
    - classes: 클래스 이름 리스트
    - output_path: 상관 관계 그래프를 저장할 경로
    """
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=classes)
    # Compute correlation matrix
    corr = np.corrcoef(y_true_bin, rowvar=False)
    # Create a dataframe for seaborn
    corr_df = pd.DataFrame(corr, index=classes, columns=classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Labels Correlation Matrix (Correlogram)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Labels correlation matrix saved to {output_path}")

def plot_classification_report(y_true, y_pred, classes, output_path):
    """
    분류 보고서를 텍스트 파일로 저장하고, 시각적으로도 표시합니다.
    
    Parameters:
    - y_true: 실제 클래스 리스트
    - y_pred: 예측 클래스 리스트
    - classes: 클래스 이름 리스트
    - output_path: 분류 보고서를 저장할 경로
    """
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    # Save as CSV
    report_csv = output_path.replace('.png', '.csv')
    report_df.to_csv(report_csv)
    print(f"[INFO] Classification report saved to {report_csv}")
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(report_df.iloc[:-3, :-1].astype(float), annot=True, cmap='viridis')
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Classification report heatmap saved to {output_path}")

def plot_all_metrics(y_true, y_pred, classes, output_dir):
    """
    모든 성능 척도를 계산하고 시각화합니다.
    
    Parameters:
    - y_true: 실제 클래스 리스트
    - y_pred: 예측 클래스 리스트
    - classes: 클래스 이름 리스트
    - output_dir: 모든 그래프를 저장할 기본 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion Matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, classes, cm_path, normalize=False)
    
    cm_norm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(y_true, y_pred, classes, cm_norm_path, normalize=True)
    
    # Precision-Recall Curve
    pr_dir = os.path.join(output_dir, 'precision_recall_curves')
    plot_precision_recall(y_true, y_pred, classes, pr_dir)
    
    # ROC Curve
    roc_dir = os.path.join(output_dir, 'roc_curves')
    plot_roc_curve(y_true, y_pred, classes, roc_dir)
    
    # Labels Distribution
    dist_path = os.path.join(output_dir, 'labels_distribution.png')
    plot_labels_distribution(y_true, classes, dist_path)
    
    # Labels Correlation Matrix (Correlogram)
    corr_path = os.path.join(output_dir, 'labels_correlogram.png')
    plot_labels_correlogram(y_true, classes, corr_path)
    
    # Classification Report
    report_path = os.path.join(output_dir, 'classification_report.png')
    plot_classification_report(y_true, y_pred, classes, report_path)

def detect_and_crop_objects(yolo_model, image_path, output_dir, min_size=10, device='cpu'):
    """
    YOLO 모델을 사용하여 이미지에서 객체를 감지하고, 패칭하여 저장합니다.
    
    Parameters:
    - yolo_model: Ultralytics YOLO 모델 객체
    - image_path: 원본 이미지 경로
    - output_dir: 패칭된 객체 이미지를 저장할 디렉토리
    - min_size: 패칭할 객체의 최소 픽셀 크기 (가로 또는 세로)
    - device: 모델을 실행할 디바이스 ('cpu' 또는 'cuda')
    
    Returns:
    - List of paths to the cropped object images
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return []
    
    results = yolo_model.predict(source=img, conf=0.25, device=device)
    
    cropped_image_paths = []
    for result in results:
        for box in result.boxes:
            class_idx = int(box.cls)
            xc, yc, bw, bh = box.xywh[0]
            x1 = int((xc - bw / 2) * img.shape[1])
            y1 = int((yc - bh / 2) * img.shape[0])
            x2 = int((xc + bw / 2) * img.shape[1])
            y2 = int((yc + bh / 2) * img.shape[0])
            
            # 유효성 검사
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                continue
            
            cropped = img[y1:y2, x1:x2]
            # 저장 파일명: 원본명_객체번호.jpg
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cropped_filename = f"{base_name}_{box.id}.jpg"
            cropped_path = os.path.join(output_dir, cropped_filename)
            cv2.imwrite(cropped_path, cropped)
            cropped_image_paths.append(cropped_path)
    
    return cropped_image_paths

def batch_detect_and_crop(yolo_model, src_image_dir, dst_crop_dir, min_size=10, device='cpu'):
    """
    원본 이미지 디렉토리를 순회하며 객체를 감지하고 패칭하여 저장합니다.
    
    Parameters:
    - yolo_model: Ultralytics YOLO 모델 객체
    - src_image_dir: 원본 이미지가 있는 디렉토리
    - dst_crop_dir: 패칭된 객체 이미지를 저장할 기본 디렉토리
    - min_size: 패칭할 객체의 최소 픽셀 크기 (가로 또는 세로)
    - device: 모델을 실행할 디바이스 ('cpu' 또는 'cuda')
    """
    os.makedirs(dst_crop_dir, exist_ok=True)
    image_files = [f for f in os.listdir(src_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        image_path = os.path.join(src_image_dir, img_file)
        # YOLO 결과
        cropped_image_paths = detect_and_crop_objects(
            yolo_model, image_path, dst_crop_dir, min_size, device
        )
        if not cropped_image_paths:
            print(f"[INFO] No valid objects found in image: {image_path}")

def main():
    # 설정: 학습된 YOLO 가중치 파일 경로
    trained_weights_path = "results/cls_head/cctv_day_exp/weights/best.pt"  # 실제 경로로 변경
    
    # 설정: 원본 이미지 디렉토리 목록
    source_directories = [
        "datasets/testset/army_add/cctv_day/images/train",
        "datasets/testset/army_add/cctv_day/images/val",
        "datasets/testset/cctv_day/images/train",
        "datasets/testset/cctv_day/images/val",
        "datasets/testset/cctv_night/images/train",
        "datasets/testset/cctv_night/images/val",
        "datasets/testset/tod_day/images/train",
        "datasets/testset/tod_day/images/val",
        "datasets/testset/tod_night/images/train",
        "datasets/testset/tod_night/images/val",
    ]
    
    # 설정: 패칭된 이미지 저장 디렉토리 목록 --> 패칭된 이미지 저장할 경로. (source 디렉토리와 매칭)
    cropped_directories = [
        "results/army_add_vit/cctv_day_cls/train",
        "results/army_add_vit/cctv_day_cls/val",
        "results/cctv_day_vit/train",
        "results/cctv_day_vit/val",
        "results/cctv_night_vit/train",
        "results/cctv_night_vit/val",
        "results/tod_day_vit/train",
        "results/tod_day_vit/val",
        "results/tod_night_vit/train",
        "results/tod_night_vit/val",
    ]
    
    # 설정: ViT 분류 결과 저장 CSV 파일
    output_csv = "results/pipeline/classification_results.csv"
    
    # 설정: 성능 척도 및 그래프 저장 디렉토리
    performance_metrics_dir = "results/pipeline/performance_metrics"
    
    # 디바이스 설정: GPU 사용 가능 시 GPU 사용, 아니면 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("[INFO] Using GPU for inference.")
    else:
        print("[INFO] Using CPU for inference.")
    
    # YOLO 모델 로드 (학습된 가중치 사용)
    print("[INFO] Loading fine-tuned YOLO model...")
    yolo_model = YOLO(trained_weights_path).to(device)
    print("[INFO] YOLO model loaded.")
    
    # 모든 원본 이미지 디렉토리에 대해 객체 감지 및 패칭 수행
    print("[INFO] Starting object detection and cropping with YOLO...")
    for src_dir, crop_dir in zip(source_directories, cropped_directories):
        print(f"[INFO] Processing directory: {src_dir}")
        batch_detect_and_crop(yolo_model, src_dir, crop_dir, min_size=10, device=device)
    print("[INFO] Object detection and cropping completed.")
    
    # ViT 모델 로드
    print("[INFO] Loading ViT model...")
    vit_model, vit_feature_extractor = load_vit_model(device=device)
    print("[INFO] ViT model loaded.")
    
    # ViT를 사용한 분류 수행
    print("[INFO] Starting classification with ViT...")
    y_true_all = []
    y_pred_all = []
    
    for crop_dir in cropped_directories:
        if not os.path.exists(crop_dir):
            print(f"[WARNING] Cropped directory does not exist: {crop_dir}")
            continue
        print(f"[INFO] Classifying images in directory: {crop_dir}")
        y_true, y_pred = batch_classify_vit(vit_model, vit_feature_extractor, crop_dir, output_csv, device=device)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
    
    print("[INFO] Classification with ViT completed.")
    
    # 성능 척도 및 그래프 생성
    print("[INFO] Starting performance metrics calculation and plotting...")
    classes = sorted(list(set(y_true_all)))  # 실제 클래스 목록 정렬
    plot_all_metrics(y_true_all, y_pred_all, classes, performance_metrics_dir)
    print("[INFO] Performance metrics calculation and plotting completed.")

if __name__ == "__main__":
    main()
