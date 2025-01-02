import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np

def ensure_10class_folders(data_dir, num_classes=10):
    """
    data_dir 내부에 train/0..9, val/0..9 폴더가 없으면 생성합니다.
    또한, 빈 폴더가 있을 경우 더미 이미지를 하나 생성하여 클래스 수 불일치 문제를 방지합니다.
    """
    for split in ["train", "val"]:
        split_path = os.path.join(data_dir, split)
        for c in range(num_classes):
            class_folder = os.path.join(split_path, str(c))
            if not os.path.exists(class_folder):
                os.makedirs(class_folder, exist_ok=True)
                print(f"[INFO] Created missing class folder: {class_folder}")
            # 폴더가 비어 있는지 확인
            if not any(os.scandir(class_folder)):
                # 더미 이미지 생성 (10x10 픽셀의 검은색 이미지)
                dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
                dummy_filename = "dummy.jpg"
                dummy_path = os.path.join(class_folder, dummy_filename)
                cv2.imwrite(dummy_path, dummy_image)
                print(f"[INFO] Created dummy image in empty class folder: {dummy_path}")

def train_classification_head(
    data_dir,
    pretrained_weights="yolo11x-cls.pt",
    epochs=3,
    batch_size=16,
    img_size=224,
    lr=1e-4,
    project_dir="results/cls_head",
    experiment_name="exp"
):
    """
    분류(Classify) 태스크로 모델을 학습합니다.
    - Backbone, Neck, Reg Head는 동결(freeze)되고, Classification Head만 학습됩니다.
    - 결과(가중치, 그래프, 지표)는 지정된 결과 폴더에 저장됩니다.
    
    Parameters:
    - data_dir (str): 분류용 데이터 폴더 경로 (train/0..9, val/0..9 구조)
    - pretrained_weights (str): 사전 학습된 가중치 파일 경로
    - epochs (int): 학습 에폭 수
    - batch_size (int): 배치 사이즈
    - img_size (int): 이미지 크기
    - lr (float): 학습률
    - project_dir (str): 결과 저장 상위 디렉토리
    - experiment_name (str): 실험 이름 (결과 폴더명)
    """
    
    # (A) 10개 클래스 폴더 보장 (빈 폴더 자동 생성 및 더미 이미지 추가)
    ensure_10class_folders(data_dir, num_classes=10)

    # (B) 모델 로드: 분류 모드 + 사전 학습 가중치
    model_ = YOLO(pretrained_weights, task="classify")

    # (C) Manual Freeze: 모든 레이어 동결 후, 분류 헤드만 풀기
    # 모델 구조 확인을 위해 모델 출력 (디버깅 용도)
    print("\n[DEBUG] Model structure:")
    print(model_.model)

    # 1. 모든 파라미터 동결
    for param in model_.model.parameters():
        param.requires_grad = False

    # 2. 분류 헤드만 다시 학습 가능하도록 설정
    # 분류 헤드의 이름을 확인하고 해당 파라미터만 requires_grad=True로 설정
    # 일반적으로 'head' 또는 'classifier'로 명명됨. 없으면 마지막 모듈을 분류 헤드로 간주
    if hasattr(model_.model, 'head'):
        head = model_.model.head
        print("[INFO] Found 'head' attribute. Unfreezing classification head.")
    elif hasattr(model_.model, 'classifier'):
        head = model_.model.classifier
        print("[INFO] Found 'classifier' attribute. Unfreezing classification head.")
    else:
        # 마지막 서브모듈을 분류 헤드로 간주
        head = list(model_.model.children())[-1]
        print("[INFO] Using last module as classification head. Unfreezing.")

    for param in head.parameters():
        param.requires_grad = True

    # (D) 학습 호출: freeze 파라미터는 사용하지 않음 (이미 manual freeze 설정)
    results = model_.train(
        data=data_dir,
        epochs=epochs,
        batch=batch_size,
        # freeze=0,  # 이미 파라미터를 동결/해제했으므로 주석 처리
        imgsz=img_size,
        lr0=lr,
        project=project_dir,
        name=experiment_name,
        verbose=True
    )

    # (E) 결과 파일 정리
    experiment_path = os.path.join(project_dir, experiment_name)
    output_result_path = os.path.join(experiment_path, "results")
    os.makedirs(output_result_path, exist_ok=True)

    # 1) plot() -> results.png, confusion_matrix.png 등 생성
    results.plot()
    default_plot_path = os.path.join(experiment_path, "results.png")
    if os.path.exists(default_plot_path):
        os.rename(
            default_plot_path,
            os.path.join(output_result_path, "results.png")
        )

    # 추가 그래프 파일 이동 (confusion_matrix.png, F1_curve.png 등)
    for f in ["confusion_matrix.png", "F1_curve.png", "P_curve.png", "R_curve.png"]:
        src = os.path.join(experiment_path, f)
        if os.path.exists(src):
            os.rename(src, os.path.join(output_result_path, f))

    # 2) results.json, results.csv 이동
    json_path = os.path.join(experiment_path, "results.json")
    csv_path = os.path.join(experiment_path, "results.csv")
    if os.path.exists(json_path):
        os.rename(json_path, os.path.join(output_result_path, "results.json"))
    if os.path.exists(csv_path):
        os.rename(csv_path, os.path.join(output_result_path, "results.csv"))

    print("[INFO] Training finished! (Classification mode, manual freeze)")
    print(f"[INFO] Weights/Logs     : {experiment_path}")
    print(f"[INFO] Plots & metrics  : {output_result_path}")

    # (선택) 추가 지표 출력
    if 'metrics/accuracy_top1' in results.metrics:
        print(f"[INFO] final accuracy_top1 : {results.metrics['metrics/accuracy_top1']}")
    else:
        print("[INFO] final accuracy_top1 : N/A")

    return results

# -------------------------------------------------------
# (메인) 여러 폴더 반복 처리 예시
# -------------------------------------------------------
if __name__ == "__main__":
    data_dirs = [
        ("datasets/testset/cctv_day_cls", "cctv_day_exp"),
        #("datasets/testset/cctv_night_cls", "cctv_night_exp"),
        ("datasets/testset/tod_day_cls", "tod_day_exp")
        #("datasets/testset/tod_night_cls", "tod_night_exp")
    ]

    for folder_path, exp_name in data_dirs:
        print(f"\n[INFO] Training folder: {folder_path}")
        train_classification_head(
            data_dir=folder_path,
            pretrained_weights="yolo11x-cls.pt",
            epochs=3,
            batch_size=16,
            img_size=224,
            lr=1e-4,
            project_dir="results/cls_head",
            experiment_name=exp_name
        )