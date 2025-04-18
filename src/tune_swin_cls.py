# swin_tune_cls.py

import os
import pandas as pd
from transformers import SwinForImageClassification, AutoImageProcessor, TrainingArguments, Trainer
from PIL import Image
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ================================================
# 설정 섹션
# ================================================

name = "VisDrone"

# 여러 모델 디렉토리 리스트
MODEL_DIRECTORIES = [
    f"/mnt/nas4/jyh/ultralytics/ViT_results/ViT_{name}/{name}"
    #"/mnt/nas4/jyh/ultralytics/results_vit/frozen_transfer_cd5"
]

# Swin Transformer 모델 설정
MODEL_SWIN_NAME = "microsoft/swin-large-patch4-window7-224"  # 사용할 Swin Transformer 모델 이름 또는 경로
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4  # 배치 크기 설정
NUM_EPOCHS = 5  # 미세 조정 에포크 수
LEARNING_RATE = 5e-5  # 학습률

# ================================================
# 패치 데이터셋 정의
# ================================================

class PatchDataset(Dataset):
    def __init__(self, dataframe, feature_extractor, label_mapping=None, transform=None):
        self.df = dataframe
        self.feature_extractor = feature_extractor
        self.label_mapping = label_mapping  # 클래스 이름을 인덱스로 매핑하는 딕셔너리
        self.transform = transform  # 이미지 증강 추가 (필요시)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patch_path = self.df.iloc[idx]['patch_path']
        label = self.df.iloc[idx]['pred_class_yolo']  # 레이블 컬럼 변경
        if self.label_mapping:
            label = self.label_mapping.get(label, -1)  # 매핑되지 않은 레이블은 -1로 설정
        image = Image.open(patch_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # [3, H, W]
        return {
            'pixel_values': pixel_values,
            'labels': label
        }

# ================================================
# 함수 정의
# ================================================

def load_swin_model(model_name, num_labels):
    print(f"Swin Transformer 모델 로드 중: {model_name}...")
    feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    model = SwinForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # 분류기 크기 불일치 무시
    )
    return feature_extractor, model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def classify_and_finetune(model_dir, metadata_csv, feature_extractor, swin_model, class_to_idx):
    """
    주어진 모델 디렉토리에서 패치 메타데이터를 로드하고, Swin 모델을 미세 조정한 후,
    패치를 분류하여 결과를 모델 디렉토리에 저장합니다.
    """
    # 모델 이름 추출 (예: frozen_transfer_cn3)
    model_name = os.path.basename(model_dir)
    print(f"\n모델: {model_name}")

    # 모델별 출력 디렉토리 설정 (미세 조정된 모델 및 로그 저장)
    swin_finetuned_dir = os.path.join(model_dir, "swin_finetuned")
    os.makedirs(swin_finetuned_dir, exist_ok=True)
    swin_best_model_dir = os.path.join(swin_finetuned_dir, "best_model")
    swin_logs_dir = os.path.join(swin_finetuned_dir, "logs")

    # 패치 메타데이터 로드
    df = pd.read_csv(metadata_csv)
    print(f"총 패치된 이미지 수: {len(df)}")

    # 패치가 저장되지 않은 경우 스킵
    df = df[df['patch_path'].notna() & (df['patch_path'] != "")]
    if df.empty:
        print("패치 이미지가 없습니다. 분류를 스킵합니다.")
        return

    # 클래스 이름을 인덱스로 매핑
    df['labels'] = df['pred_class_yolo'].map(class_to_idx)
    # 매핑되지 않은 레이블이 있는지 확인
    if df['labels'].isnull().any():
        print("경고: 일부 레이블이 클래스 매핑에 존재하지 않습니다. 해당 샘플을 제거합니다.")
        df = df.dropna(subset=['labels'])
        df['labels'] = df['labels'].astype(int)

    # 데이터셋 분할 (학습: 80%, 검증: 20%)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)

    # 커스텀 데이터셋 정의 (필요시 transform 추가)
    train_dataset = PatchDataset(train_df.reset_index(drop=True), feature_extractor, class_to_idx)
    val_dataset = PatchDataset(val_df.reset_index(drop=True), feature_extractor, class_to_idx)

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir=swin_finetuned_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=swin_logs_dir,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Trainer 초기화
    trainer = Trainer(
        model=swin_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 모델 미세 조정
    print(f"{model_name}에 대한 Swin 모델 미세 조정 시작...")
    trainer.train()

    # 학습된 모델 저장
    trainer.save_model(swin_best_model_dir)
    print(f"{model_name}에 대한 Swin 모델이 성공적으로 미세 조정되었습니다.")

    # Feature Extractor 저장
    feature_extractor.save_pretrained(swin_best_model_dir)
    print(f"{model_name}에 대한 Feature Extractor가 저장되었습니다.")

    # 미세 조정된 모델 로드
    swin_model_finetuned = SwinForImageClassification.from_pretrained(swin_best_model_dir)
    swin_model_finetuned.to(DEVICE)
    swin_model_finetuned.eval()

    # 패치 메타데이터 다시 불러오기
    df = pd.read_csv(metadata_csv)
    df = df[df['patch_path'].notna() & (df['patch_path'] != "")]
    df['labels'] = df['pred_class_yolo'].map(class_to_idx)
    df = df.dropna(subset=['labels'])
    df['labels'] = df['labels'].astype(int)

    # 데이터셋 정의 (레이블은 필요 없으므로)
    classification_dataset = PatchDataset(df.reset_index(drop=True), feature_extractor, label_mapping=None)
    classification_dataloader = DataLoader(classification_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Swin 분류 수행
    print(f"{model_name}에 대한 Swin 분류 시작...")
    swin_predictions = []
    swin_confidences = []

    with torch.no_grad():
        for batch in tqdm(classification_dataloader, desc="Swin 분류 중"):
            pixel_values = batch['pixel_values'].to(DEVICE)  # [batch_size, 3, H, W]
            outputs = swin_model_finetuned(pixel_values=pixel_values)
            logits = outputs.logits  # [batch_size, num_labels]
            probs = torch.softmax(logits, dim=-1)  # [batch_size, num_labels]
            preds = torch.argmax(probs, dim=-1).cpu().numpy()  # [batch_size]
            confidences = torch.max(probs, dim=-1).values.cpu().numpy()  # [batch_size]
            swin_predictions.extend(preds)
            swin_confidences.extend(confidences)

    # 클래스 이름 매핑
    class_names = list(class_to_idx.keys())
    swin_pred_classes = [class_names[pred] if pred < len(class_names) else "Unknown" for pred in swin_predictions]

    # 예측 결과를 DataFrame에 추가
    df['pred_class_swin'] = swin_pred_classes
    df['confidence_swin'] = swin_confidences

    # 분류 결과 CSV 파일 경로 설정 (model_dir에 저장)
    output_csv = os.path.join(model_dir, "patches_with_swin.csv")

    # 분류 결과 저장
    df.to_csv(output_csv, index=False)
    print(f"Swin 분류 결과가 {output_csv}에 저장되었습니다.")

# ================================================
# 메인 함수 정의
# ================================================

def main():
    # Swin Transformer 모델 로드 (미세 조정을 위해)
    print(f"Swin Transformer 모델 로드 중: {MODEL_SWIN_NAME}...")
    feature_extractor, swin_model = load_swin_model(MODEL_SWIN_NAME, num_labels=1000)  # 초기 클래스 수는 나중에 수정 가능

    # 각 모델 디렉토리별로 분류 및 미세 조정 수행
    for model_dir in MODEL_DIRECTORIES:
        metadata_csv = os.path.join(model_dir, 'patches_metadata.csv')

        if not os.path.exists(metadata_csv):
            print(f"경고: 메타데이터 CSV 파일이 존재하지 않습니다: {metadata_csv}. 스킵합니다.")
            continue

        # 클래스 매핑을 위한 고유 레이블 수집
        df_labels = pd.read_csv(metadata_csv)
        unique_labels = df_labels['pred_class_yolo'].unique()  # 'true_label' 대신 'pred_class_yolo' 사용
        class_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print(f"{model_dir}의 클래스 수: {len(class_to_idx)}")

        # Swin 모델의 num_labels 업데이트
        feature_extractor, swin_model = load_swin_model(MODEL_SWIN_NAME, num_labels=len(class_to_idx))
        swin_model.to(DEVICE)

        # classify_and_finetune 함수 호출
        classify_and_finetune(model_dir, metadata_csv, feature_extractor, swin_model, class_to_idx)

    print("\n모든 모델 디렉토리에 대한 Swin 분류 및 미세 조정이 완료되었습니다.")

# ================================================
# 메인 함수 실행
# ================================================

if __name__ == "__main__":
    main()
