# analyze_vit_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from PIL import Image
import json

# ================================================
# 설정 섹션
# ================================================

# 이미지 디렉토리 및 CSV 파일 경로 설정
IMAGE_DIRECTORY = "/mnt/nas4/jyh/ultralytics/datasets/VisDrone/VisDrone2019-DET-test-challenge/images"
METADATA_CSV = os.path.join(IMAGE_DIRECTORY, 'images_with_vit.csv')

# 출력 디렉토리 설정
OUTPUT_DIR = "/mnt/nas4/jyh/ultralytics/results_vit_analyzer/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================
# 데이터 로드 및 전처리
# ================================================

# CSV 파일 로드
if not os.path.exists(METADATA_CSV):
    raise FileNotFoundError(f"메타데이터 CSV 파일을 찾을 수 없습니다: {METADATA_CSV}")

df = pd.read_csv(METADATA_CSV)

# 'Unknown' 클래스를 제외하거나 별도로 처리할 수 있습니다.
df = df[df['pred_class_vit'] != 'Unknown']

# 전체 정확도 계산
overall_accuracy = (df['label'] == df['pred_class_vit']).mean()
print(f"전체 정확도: {overall_accuracy:.2f}")

# 클래스 레이블 목록
classes = sorted(df['label'].unique())
print(f"클래스 목록: {classes}")

# ================================================
# 혼동 행렬 생성 및 시각화
# ================================================

# 혼동 행렬 계산
cm = confusion_matrix(df['label'], df['pred_class_vit'], labels=classes)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# 혼동 행렬 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_heatmap.png'))
plt.show()

# ================================================
# 클래스별 정확도 계산 및 시각화
# ================================================

# 클래스별 정확도 계산
class_accuracy = df.groupby('label').apply(lambda x: (x['label'] == x['pred_class_vit']).mean()).sort_values(ascending=False)

# 클래스별 정확도 막대 그래프 시각화
plt.figure(figsize=(12, 6))
sns.barplot(x=class_accuracy.index, y=class_accuracy.values, palette='viridis')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_accuracy.png'))
plt.show()

# ================================================
# 정밀도, 재현율, F1-Score 계산 및 출력
# ================================================

# 분류 보고서 생성
report = classification_report(df['label'], df['pred_class_vit'], target_names=classes, digits=4)
print("Classification Report:")
print(report)

# 분류 보고서를 텍스트 파일로 저장
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write(report)

# ================================================
# 신뢰도 분포 히스토그램 생성
# ================================================

plt.figure(figsize=(8, 6))
sns.histplot(df['confidence_vit'], bins=50, kde=True, color='skyblue')
plt.xlabel('Confidence')
plt.ylabel('Frequency')
plt.title('Confidence Distribution')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_distribution.png'))
plt.show()

# ================================================
# ROC 곡선 및 AUC 계산 및 시각화 (다중 클래스)
# ================================================

# 레이블 이진화
y_true = label_binarize(df['label'], classes=classes)
y_pred = label_binarize(df['pred_class_vit'], classes=classes)
n_classes = y_true.shape[1]

# ROC AUC 계산
roc_auc = dict()
fpr = dict()
tpr = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC 곡선 시각화
plt.figure(figsize=(12, 8))
colors = sns.color_palette("hls", n_classes)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right", fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'))
plt.show()

# ================================================
# 올바르게 예측된 이미지와 잘못 예측된 이미지 시각화
# ================================================

def plot_image_with_predictions(image_path, true_label, pred_label, confidence, save_path=None):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})", fontsize=12)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 올바르게 예측된 상위 5개 이미지 시각화
correct_predictions = df[df['label'] == df['pred_class_vit']].sample(n=5, random_state=42)
for idx, row in correct_predictions.iterrows():
    img_path = os.path.join(IMAGE_DIRECTORY, row['image_filename'])
    save_path = os.path.join(OUTPUT_DIR, f'correct_prediction_{idx}.png')
    plot_image_with_predictions(img_path, row['label'], row['pred_class_vit'], row['confidence_vit'], save_path)

# 잘못 예측된 상위 5개 이미지 시각화
incorrect_predictions = df[df['label'] != df['pred_class_vit']].sample(n=5, random_state=42)
for idx, row in incorrect_predictions.iterrows():
    img_path = os.path.join(IMAGE_DIRECTORY, row['image_filename'])
    save_path = os.path.join(OUTPUT_DIR, f'incorrect_prediction_{idx}.png')
    plot_image_with_predictions(img_path, row['label'], row['pred_class_vit'], row['confidence_vit'], save_path)

# ================================================
# 학습 곡선 (손실 및 정확도) 시각화
# ================================================

# Trainer 로그 파일 로드
# 'logs' 폴더 내에 'trainer_state.json' 파일이 있다고 가정
trainer_state_path = os.path.join("/mnt/nas4/jyh/ultralytics/results_vit/finetuned_model/logs", 'trainer_state.json')

if os.path.exists(trainer_state_path):
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    
    # 손실과 정확도 추출
    training_history = trainer_state.get('log_history', [])
    epochs = []
    train_loss = []
    eval_loss = []
    eval_accuracy = []

    for entry in training_history:
        if 'epoch' in entry and 'loss' in entry:
            epochs.append(entry['epoch'])
            train_loss.append(entry['loss'])
        if 'epoch' in entry and 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
        if 'epoch' in entry and 'eval_accuracy' in entry:
            eval_accuracy.append(entry['eval_accuracy'])
    
    if epochs and train_loss and eval_loss and eval_accuracy:
        # 손실 그래프 시각화
        plt.figure(figsize=(12, 5))
        
        # 손실
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label='Train Loss')
        plt.plot(epochs, eval_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        
        # 정확도
        plt.subplot(1, 2, 2)
        plt.plot(epochs, eval_accuracy, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Over Epochs')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
        plt.show()
    else:
        print("손실 및 정확도 데이터를 추출할 수 없습니다. Trainer 로그를 확인하세요.")
else:
    print(f"Trainer state 파일을 찾을 수 없습니다: {trainer_state_path}")
    print("학습 곡선을 시각화하려면 Trainer의 로그 파일을 확인하세요.")

# ================================================
# 스크립트 종료 메시지
# ================================================

print(f"\n모든 분석 결과가 {OUTPUT_DIR} 디렉토리에 저장되었습니다.")
