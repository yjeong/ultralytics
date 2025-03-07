import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
from tqdm import tqdm
import torch
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 시각화 스타일 설정
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# ================================================
# 설정 섹션
# ================================================

name = "VisDrone"

# 입력 디렉토리 설정 (사용자 수정)
INPUT_DIR = f"/mnt/nas4/jyh/ultralytics/ViT_results/ViT_{name}/{name}"

# 출력 디렉토리 설정 (사용자 수정)
OUTPUT_DIR = f"/mnt/nas4/jyh/ultralytics/ViT_results/ViT_{name}/{name}/swin_plots"  # 예: "/home/user/ViT_plots"

# swin_finetuned 디렉토리 경로
SWIN_FINETUNED_DIR = os.path.join(INPUT_DIR, "swin_finetuned")

# 로그 디렉토리 (TensorBoard 이벤트 파일이 저장된 곳)
LOG_DIR = os.path.join(SWIN_FINETUNED_DIR, "logs")  # 실제 로그 위치에 따라 수정

# 패치 CSV 파일 경로
PATCHES_CSV = os.path.join(INPUT_DIR, "patches_with_swin.csv")

# ================================================
# 함수 정의
# ================================================

def ensure_output_dir(directory):
    """
    출력 디렉토리가 존재하지 않으면 생성합니다.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"출력 디렉토리를 생성했습니다: {directory}")
    else:
        print(f"출력 디렉토리가 이미 존재합니다: {directory}")

def plot_training_metrics(log_dir, output_path):
    """
    TensorBoard 이벤트 파일을 로드하여 훈련 및 검증 정확도와 손실을 시각화하고 저장합니다.
    """
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"TensorBoard 이벤트 파일을 찾을 수 없습니다: {log_dir}. 훈련 지표 그래프 생성을 생략합니다.")
        return

    event_file = sorted(event_files)[-1]
    print(f"가장 최근의 TensorBoard 이벤트 파일을 사용합니다: {event_file}")

    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    available_tags = event_acc.Tags().get('scalars', [])
    print(f"사용 가능한 스칼라 태그: {available_tags}")

    train_acc = []
    train_acc_steps = []
    if 'train_accuracy' in available_tags:
        for scalar in event_acc.Scalars('train_accuracy'):
            train_acc_steps.append(scalar.step)
            train_acc.append(scalar.value)
    else:
        print("태그 'train_accuracy'가 존재하지 않습니다.")

    eval_acc = []
    eval_acc_steps = []
    if 'eval_accuracy' in available_tags:
        for scalar in event_acc.Scalars('eval_accuracy'):
            eval_acc_steps.append(scalar.step)
            eval_acc.append(scalar.value)
    else:
        print("태그 'eval_accuracy'가 존재하지 않습니다.")

    train_loss = []
    train_loss_steps = []
    if 'train_loss' in available_tags:
        for scalar in event_acc.Scalars('train_loss'):
            train_loss_steps.append(scalar.step)
            train_loss.append(scalar.value)
    else:
        print("태그 'train_loss'가 존재하지 않습니다.")

    eval_loss = []
    eval_loss_steps = []
    if 'eval_loss' in available_tags:
        for scalar in event_acc.Scalars('eval_loss'):
            eval_loss_steps.append(scalar.step)
            eval_loss.append(scalar.value)
    else:
        print("태그 'eval_loss'가 존재하지 않습니다.")

    if not (train_acc or eval_acc or train_loss or eval_loss):
        print("훈련 지표를 찾을 수 없습니다. 훈련 지표 그래프 생성을 생략합니다.")
        return

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    if train_acc:
        plt.plot(train_acc_steps, train_acc, label='Train Accuracy')
    if eval_acc:
        plt.plot(eval_acc_steps, eval_acc, label='Validation Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    if train_loss:
        plt.plot(train_loss_steps, train_loss, label='Train Loss')
    if eval_loss:
        plt.plot(eval_loss_steps, eval_loss, label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"훈련 및 검증 지표 그래프가 저장되었습니다: {output_path}")

def plot_confusion_matrix(csv_path, class_names, output_path):
    """
    혼동 행렬(비정규화)을 생성하고 시각화하여 저장합니다.
    """
    df = pd.read_csv(csv_path)
    y_true = df['pred_class_yolo']
    y_pred = df['pred_class_swin']

    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Non-Normalized)')

    # 0인 값을 표시하지 않는 로직
    for text_obj in ax.texts:
        if text_obj.get_text() == '0':
            text_obj.set_text('')  # 0 대신 빈 문자열로 설정

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"혼동 행렬이 저장되었습니다: {output_path}")

def plot_confusion_matrix_normalized(csv_path, class_names, output_path):
    """
    정규화된 혼동 행렬을 생성하고 시각화하여 저장합니다.
    """
    df = pd.read_csv(csv_path)
    y_true = df['pred_class_yolo']
    y_pred = df['pred_class_swin']

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Normalized)')

    # 0.00인 값을 표시하지 않는 로직
    # (ex. "0.00", "0.0", "0" 등 패턴에 맞춰 처리)
    for text_obj in ax.texts:
        # 숫자가 0 또는 0.00으로 표시된 경우 숨김
        if text_obj.get_text() in ('0', '0.0', '0.00'):
            text_obj.set_text('')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"정규화된 혼동 행렬이 저장되었습니다: {output_path}")

def plot_classification_report(csv_path, class_names, output_path):
    """
    각 클래스별 정밀도, 재현율, F1-점수를 시각화하여 저장합니다.
    """
    df = pd.read_csv(csv_path)
    y_true = df['pred_class_yolo']
    y_pred = df['pred_class_swin']

    report = classification_report(y_true, y_pred, labels=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df = report_df[report_df['index'].isin(class_names)]

    plt.figure(figsize=(14, 8))
    width = 0.2
    x = range(len(class_names))

    plt.bar([p - width for p in x], report_df['precision'], width=width, label='Precision', color='skyblue')
    plt.bar(x, report_df['recall'], width=width, label='Recall', color='lightgreen')
    plt.bar([p + width for p in x], report_df['f1-score'], width=width, label='F1-Score', color='salmon')

    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Precision, Recall, F1-Score per Class')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"정밀도, 재현율, F1-점수 그래프가 저장되었습니다: {output_path}")

def plot_confidence_distribution(csv_path, output_path):
    """
    올바른 예측과 오분류된 예측의 신뢰도 분포를 시각화하여 저장합니다.
    """
    df = pd.read_csv(csv_path)
    correct = df['pred_class_swin'] == df['pred_class_yolo']
    incorrect = ~correct

    plt.figure(figsize=(12, 6))
    sns.histplot(df[correct]['confidence_swin'], color='green', label='Correct', kde=True, stat="density", bins=30, alpha=0.6)
    sns.histplot(df[incorrect]['confidence_swin'], color='red', label='Incorrect', kde=True, stat="density", bins=30, alpha=0.6)
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Score Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"예측 신뢰도 분포 그래프가 저장되었습니다: {output_path}")

def visualize_sample_predictions(csv_path, output_path, num_samples=5):
    """
    올바르게 분류된 샘플과 오분류된 샘플의 이미지를 시각화하여 저장합니다.
    """
    df = pd.read_csv(csv_path)
    correct_df = df[df['pred_class_swin'] == df['pred_class_yolo']]
    incorrect_df = df[df['pred_class_swin'] != df['pred_class_yolo']]

    num_correct = min(num_samples, len(correct_df))
    num_incorrect = min(num_samples, len(incorrect_df))

    correct = correct_df.sample(n=num_correct, random_state=42) if num_correct > 0 else pd.DataFrame()
    incorrect = incorrect_df.sample(n=num_incorrect, random_state=42) if num_incorrect > 0 else pd.DataFrame()

    max_samples = max(num_correct, num_incorrect)
    if max_samples == 0:
        print("시각화할 샘플이 없습니다.")
        return

    fig, axes = plt.subplots(2, max_samples, figsize=(20, 8))

    # 올바른 예측
    for i in range(max_samples):
        if i < num_correct:
            row = correct.iloc[i]
            try:
                img = Image.open(row['patch_path']).convert("RGB")
                axes[0, i].imshow(img)
                axes[0, i].set_title(f"True: {row['pred_class_yolo']}\nPred: {row['pred_class_swin']}")
            except Exception as e:
                print(f"이미지 로딩 오류: {row['patch_path']}. 오류: {e}")
                axes[0, i].text(0.5, 0.5, 'Image Error', horizontalalignment='center', verticalalignment='center')
            axes[0, i].axis('off')
        else:
            axes[0, i].axis('off')

    # 오분류된 예측
    for i in range(max_samples):
        if i < num_incorrect:
            row = incorrect.iloc[i]
            try:
                img = Image.open(row['patch_path']).convert("RGB")
                axes[1, i].imshow(img)
                axes[1, i].set_title(f"True: {row['pred_class_yolo']}\nPred: {row['pred_class_swin']}")
            except Exception as e:
                print(f"이미지 로딩 오류: {row['patch_path']}. 오류: {e}")
                axes[1, i].text(0.5, 0.5, 'Image Error', horizontalalignment='center', verticalalignment='center')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')

    if max_samples > 0:
        axes[0, 0].set_ylabel('Correct', fontsize=12)
        axes[1, 0].set_ylabel('Incorrect', fontsize=12)

    plt.suptitle('Sample Predictions', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"샘플 예측 결과 시각화가 저장되었습니다: {output_path}")

def plot_class_distribution(csv_path, output_path):
    """
    데이터셋 내 각 클래스의 샘플 수를 시각화하여 저장합니다.
    """
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(14, 7))
    sns.countplot(x='pred_class_yolo', data=df, order=sorted(df['pred_class_yolo'].unique()))
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"클래스 분포 그래프가 저장되었습니다: {output_path}")

def plot_confidence_boxplot(csv_path, output_path):
    """
    각 클래스별 예측 신뢰도의 분포를 박스 플롯으로 시각화하여 저장합니다.
    """
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(16, 10))
    sns.boxplot(x='pred_class_swin', y='confidence_swin', data=df, order=sorted(df['pred_class_swin'].unique()))
    plt.xlabel('Predicted Class')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Scores per Predicted Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"클래스별 신뢰도 박스 플롯이 저장되었습니다: {output_path}")

def plot_checkpoint_performance(output_path):
    """
    체크포인트 별 성능 비교 기능을 제거합니다.
    """
    print("체크포인트 별 성능 비교 기능이 제거되었습니다.")

# ================================================
# 메인 함수 실행
# ================================================

def run_all_plots(input_dir, output_dir, swin_finetuned_dir, patches_csv, log_dir, device):
    """
    모든 분석 및 시각화를 실행하는 함수입니다.
    """
    ensure_output_dir(output_dir)
    if not os.path.exists(patches_csv):
        print(f"CSV 파일을 찾을 수 없습니다: {patches_csv}. 모든 플롯을 생성할 수 없습니다.")
        return

    df = pd.read_csv(patches_csv)
    class_names = sorted(df['pred_class_yolo'].unique())

    # 1. 훈련 및 검증 지표의 변화 추이
    training_metrics_path = os.path.join(output_dir, "training_validation_metrics.png")
    print("1. 훈련 및 검증 지표의 변화 추이")
    plot_training_metrics(log_dir, training_metrics_path)

    # 2. 혼동 행렬 (Non-Normalized)
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    print("2. 혼동 행렬 (Non-Normalized)")
    plot_confusion_matrix(patches_csv, class_names, confusion_matrix_path)

    # 2-1. 혼동 행렬 (Normalized)
    confusion_matrix_normalized_path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    print("2-1. 혼동 행렬 (Normalized)")
    plot_confusion_matrix_normalized(patches_csv, class_names, confusion_matrix_normalized_path)

    # 3. 정밀도, 재현율, F1-점수
    classification_report_path = os.path.join(output_dir, "classification_report.png")
    print("3. 정밀도, 재현율, F1-점수")
    plot_classification_report(patches_csv, class_names, classification_report_path)

    # 4. 예측 신뢰도 분포
    confidence_distribution_path = os.path.join(output_dir, "confidence_distribution.png")
    print("4. 예측 신뢰도 분포")
    plot_confidence_distribution(patches_csv, confidence_distribution_path)

    # 5. 체크포인트 별 성능 비교 (제거)
    checkpoint_performance_path = os.path.join(output_dir, "checkpoint_performance.png")
    print("5. 체크포인트 별 성능 비교 (제거)")
    plot_checkpoint_performance(checkpoint_performance_path)

    # 6. 샘플 예측 결과 시각화
    sample_predictions_path = os.path.join(output_dir, "sample_predictions.png")
    print("6. 샘플 예측 결과 시각화")
    visualize_sample_predictions(patches_csv, sample_predictions_path, num_samples=5)

    # 7. 클래스 불균형 분석
    class_distribution_path = os.path.join(output_dir, "class_distribution.png")
    print("7. 클래스 불균형 분석")
    plot_class_distribution(patches_csv, class_distribution_path)

    # 8. 클래스별 신뢰도 박스 플롯
    confidence_boxplot_path = os.path.join(output_dir, "confidence_boxplot.png")
    print("8. 클래스별 신뢰도 박스 플롯")
    plot_confidence_boxplot(patches_csv, confidence_boxplot_path)

    print("\n모든 그래프가 성공적으로 생성되고 저장되었습니다.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_all_plots(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        swin_finetuned_dir=SWIN_FINETUNED_DIR,
        patches_csv=PATCHES_CSV,
        log_dir=LOG_DIR,
        device=device
    )
