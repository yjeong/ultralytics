import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

def train_classification_head():
    """
    예시:
    1) yolo11x-cls.pt 가중치 로드
    2) 분류 모드 (task='classify')
    3) 분류용 YAML (work_cd_cls.yaml)을 사용
    4) Backbone, Neck 동결(Freeze), Classification Head만 학습
    5) 결과(가중치, 그래프, 지표)를 별도 폴더에 저장
    """

    # ---------------------------------------------------------
    # (1) 주요 파라미터 고정
    # ---------------------------------------------------------
    data_yaml = "/mnt/nas4/jyh/ultralytics/code/work_cd.yaml"   # 분류용 YAML, 예: path: ... train: train, val: val ...
    pretrained_weights = "yolo11x-cls.pt" # 분류 사전학습 가중치
    freeze_layers = 20                    # 동결할 레이어 개수
    epochs = 30
    batch_size = 16
    img_size = 224  # 분류 모델은 224, 256 등 원하는 사이즈
    lr = 1e-4

    # 결과 저장 디렉토리
    project_dir = "results/cls_head"         # 상위 폴더
    experiment_name = "cctv_day_exp"      # 실험 폴더명
    result_dir = "results"                # 그래프/지표 파일 저장 폴더

    # ---------------------------------------------------------
    # (2) 모델 로드 - task='classify'로 지정
    # ---------------------------------------------------------
    model = YOLO(pretrained_weights, task="classify")

    # ---------------------------------------------------------
    # (3) 학습 수행
    # ---------------------------------------------------------
    #  - freeze=freeze_layers : 앞부분 레이어 동결
    #  - Ultralytics 분류 모드에서 freeze=N은
    #    '모델 구조'에 따라 N개 레이어 동결.
    results = model.train(
        data=data_yaml,         # "code/work_cd_cls.yaml"
        epochs=epochs,
        batch=batch_size,
        freeze=freeze_layers,
        imgsz=img_size,
        lr0=lr,
        project=project_dir,
        name=experiment_name,
        verbose=True
    )

    # ---------------------------------------------------------
    # (4) 학습 결과 정리 (그래프, json/csv 파일 등)
    # ---------------------------------------------------------
    experiment_path = os.path.join(project_dir, experiment_name)
    output_result_path = os.path.join(experiment_path, result_dir)
    os.makedirs(output_result_path, exist_ok=True)

    # 4-1) Ultralytics 내장 plot() → results.png, confusion_matrix.png 등 생성
    results.plot()
    default_plot_path = os.path.join(experiment_path, "results.png")
    if os.path.exists(default_plot_path):
        os.rename(
            default_plot_path,
            os.path.join(output_result_path, "results.png")
        )

    # 나머지 파일(confusion_matrix.png 등)도 이동
    for f in ["confusion_matrix.png", "F1_curve.png", "P_curve.png", "R_curve.png"]:
        src = os.path.join(experiment_path, f)
        if os.path.exists(src):
            dst = os.path.join(output_result_path, f)
            os.rename(src, dst)

    # 4-2) 정량 지표 파일 (json, csv) 이동
    json_path = os.path.join(experiment_path, "results.json")
    csv_path = os.path.join(experiment_path, "results.csv")
    if os.path.exists(json_path):
        os.rename(json_path, os.path.join(output_result_path, "results.json"))
    if os.path.exists(csv_path):
        os.rename(csv_path, os.path.join(output_result_path, "results.csv"))

    print("[INFO] Training finished (classification mode)!")
    print(f"[INFO] Best weights, logs : {experiment_path}")
    print(f"[INFO] Results/plots      : {output_result_path}")

    return results


# -------------------------------------------------------
# (메인 구동부)
# -------------------------------------------------------
if __name__ == "__main__":
    train_classification_head()
