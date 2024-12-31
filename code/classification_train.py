import os
from ultralytics import YOLO

def train_classification_head(data, model_path, epochs, imgsz, batch, output):
    """
    YOLO 분류 모델의 classification head만 학습하고 백본을 저장합니다.

    Args:
        data (str): 데이터셋 YAML 경로 (예: 'imagenet').
        model_path (str): YOLO 모델 YAML 경로 (예: 'yolo11n-cls.yaml').
        epochs (int): 학습 반복 수.
        imgsz (int): 입력 이미지 크기.
        batch (int): 배치 크기.
        output (str): 학습 완료된 백본 모델 저장 경로.
    """
    # 모델 초기화
    model = YOLO(model_path)

    # 특정 레이어 고정 (classification head만 훈련)
    for name, module in model.model.named_children():
        if name != "head":  # 'head'가 classification head라고 가정
            for param in module.parameters():
                param.requires_grad = False

    # 모델 학습
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch
    )

    # 학습된 백본 저장
    model.save(output)

if __name__ == "__main__":
    # 설정
    data = "imagenet"
    model_path = "yolo11n-cls.yaml"
    epochs = 100
    imgsz = 64
    batch = 16
    output = "yolo11n-cls-backbone.pt"

    # 학습 실행
    train_classification_head(data, model_path, epochs, imgsz, batch, output)
