import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np

def ensure_10class_folders(data_dir, num_classes=10, min_size=(10, 10)):
    """
    data_dir 내부에 train/0..9, val/0..9 폴더가 없으면 생성합니다.
    또한, 빈 폴더가 있을 경우 더미 이미지를 하나 생성하여 클래스 수 불일치 문제를 방지합니다.
    작은 이미지는 업스케일링하여 최소 크기를 충족시킵니다.
    """
    for split in ["train", "val"]:
        split_path = os.path.join(data_dir, split)
        for c in range(num_classes):
            class_folder = os.path.join(split_path, str(c))
            if not os.path.exists(class_folder):
                os.makedirs(class_folder, exist_ok=True)
                print(f"[INFO] Created missing class folder: {class_folder}")
            
            for file in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file)
                try:
                    img = cv2.imread(file_path)
                    if img is None:
                        continue
                    h, w, _ = img.shape
                    if h < min_size[0] or w < min_size[1]:
                        # 업스케일링
                        img_resized = cv2.resize(img, min_size, interpolation=cv2.INTER_LINEAR)
                        cv2.imwrite(file_path, img_resized)
                        print(f"[INFO] Resized small image: {file_path} to {min_size}")
                except Exception as e:
                    print(f"[WARNING] Failed to process image {file_path}: {e}")
            
            # 폴더가 비어 있는지 확인
            if not any(os.scandir(class_folder)):
                # 더미 이미지 생성
                dummy_image = np.zeros((*min_size, 3), dtype=np.uint8)
                dummy_filename = "dummy.jpg"
                dummy_path = os.path.join(class_folder, dummy_filename)
                cv2.imwrite(dummy_path, dummy_image)
                print(f"[INFO] Created dummy image in empty class folder: {dummy_path}")

def train_classification_head(
    data_dir,
    pretrained_weights="yolo11x-cls.pt",
    epochs=10,
    batch_size=16,
    img_size=224,
    lr=1e-4,
    project_dir="results/cls_head",
    experiment_name="exp"
):
    """
    분류(Classify) 태스크로 모델을 학습합니다.
    """
    ensure_10class_folders(data_dir, num_classes=10, min_size=(10, 10))

    model_ = YOLO(pretrained_weights, task="classify")
    print("\n[DEBUG] Model structure:")
    print(model_.model)

    for param in model_.model.parameters():
        param.requires_grad = False

    if hasattr(model_.model, 'head'):
        head = model_.model.head
        print("[INFO] Found 'head' attribute. Unfreezing classification head.")
    elif hasattr(model_.model, 'classifier'):
        head = model_.model.classifier
        print("[INFO] Found 'classifier' attribute. Unfreezing classification head.")
    else:
        head = list(model_.model.children())[-1]
        print("[INFO] Using last module as classification head. Unfreezing.")

    for param in head.parameters():
        param.requires_grad = True

    results = model_.train(
        data=data_dir,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr,
        project=project_dir,
        name=experiment_name,
        verbose=True
    )

    experiment_path = os.path.join(project_dir, experiment_name)
    output_result_path = os.path.join(experiment_path, "results")
    os.makedirs(output_result_path, exist_ok=True)

    metrics = results.results_dict
    if "top1" in metrics and "top5" in metrics:
        fig, ax = plt.subplots()
        ax.bar(["Top-1", "Top-5"], [metrics["top1"], metrics["top5"]], color=["blue", "green"])
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Accuracy")
        plt.savefig(os.path.join(output_result_path, "classification_accuracy.png"))
        plt.close(fig)

    for f in ["confusion_matrix.png", "F1_curve.png", "P_curve.png", "R_curve.png"]:
        src = os.path.join(experiment_path, f)
        if os.path.exists(src):
            os.rename(src, os.path.join(output_result_path, f))

    json_path = os.path.join(experiment_path, "results.json")
    csv_path = os.path.join(experiment_path, "results.csv")
    if os.path.exists(json_path):
        os.rename(json_path, os.path.join(output_result_path, "results.json"))
    if os.path.exists(csv_path):
        os.rename(csv_path, os.path.join(output_result_path, "results.csv"))

    print("[INFO] Training finished! (Classification mode, manual freeze)")
    print(f"[INFO] Weights/Logs     : {experiment_path}")
    print(f"[INFO] Plots & metrics  : {output_result_path}")

    print(f"[INFO] final top-1 accuracy : {metrics.get('top1', 'N/A')}")
    print(f"[INFO] final top-5 accuracy : {metrics.get('top5', 'N/A')}")

    return results

if __name__ == "__main__":
    data_dirs = [
        ("datasets/testset/cctv_day_cls", "cctv_day_exp"),
        ("datasets/testset/cctv_night_cls", "cctv_night_exp"),
        ("datasets/testset/tod_day_cls", "tod_day_exp"),
        ("datasets/testset/tod_night_cls", "tod_night_exp")
    ]

    for folder_path, exp_name in data_dirs:
        print(f"\n[INFO] Training folder: {folder_path}")
        train_classification_head(
            data_dir=folder_path,
            pretrained_weights="yolo11x-cls.pt",
            epochs=10,
            batch_size=16,
            img_size=224,
            lr=1e-4,
            project_dir="results/cls_head",
            experiment_name=exp_name
        )
