import os
import torch
import cv2  # OpenCV (patch 저장용)
from ultralytics import YOLO
from transformers import ViTForImageClassification, ViTFeatureExtractor
import yaml

def load_yaml(yaml_path):
    """
    Load a YAML file and return the parsed data.
    """
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def freeze_yolo_except_classification_head(yolo_model):
    """
    분류 헤드만 학습하고, 백본(Backbone), 넥(Neck), 회귀(Detection) 헤드는 동결(freeze)하도록 설정합니다.

    - yolo_model: ultralytics.YOLO(...) 로드된 모델 객체
    - 실제로 'model.model[-1]'이 Classification Head인지 확인해야 합니다.
    - 모델 구조가 다를 경우, 맞춰서 수정하세요.
    """
    # 전체 파라미터 동결
    for param in yolo_model.model.parameters():
        param.requires_grad = False

    # 마지막 레이어가 Classification Head라 가정
    if hasattr(yolo_model.model, 'model'):
        if len(yolo_model.model.model) > 0:
            classification_head = yolo_model.model.model[-1]
            for param in classification_head.parameters():
                param.requires_grad = True

def process_yolo_with_vit(
    yaml_file,
    yolo_model_path,
    vit_model_name,
    output_dir,
    patch_output_dir
):
    """
    1) YAML 파일로부터 데이터셋 경로 및 클래스명 정보 로드
    2) YOLO 모델로 추론 -> Bounding Box 획득
    3) Box 영역을 잘라낸 patch 이미지를 별도 디렉토리에 저장
    4) patch 이미지를 ViT로 분류
    5) 백본/넥/회귀헤드는 동결하고 분류헤드만 학습 가능하도록 설정(freeze_yolo_except_classification_head)

    Args:
        yaml_file (str): Path to the dataset YAML file.
        yolo_model_path (str): YOLOv11 기반의 커스텀 모델(.pt)
        vit_model_name (str): Hugging Face ViT 모델 이름(예: 'google/vit-base-patch16-224')
        output_dir (str): YOLO 추론 결과(라벨, bbox 시각화 등) 저장 디렉토리
        patch_output_dir (str): 잘라낸 객체 패치 이미지를 저장할 별도 디렉토리
    """
    # 1) YAML 로드
    dataset_config = load_yaml(yaml_file)
    train_dir = dataset_config['train'][0]
    val_dir   = dataset_config['val'][0]
    class_names = dataset_config['names']

    # 2) YOLO 모델 초기화
    yolo_model = YOLO(yolo_model_path)

    # [추가] 분류 헤드만 학습 -> Backbone, Neck, Detection Head 동결
    freeze_yolo_except_classification_head(yolo_model)

    # 3) ViT 모델 초기화
    vit_model = ViTForImageClassification.from_pretrained(vit_model_name)
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(patch_output_dir, exist_ok=True)

    def run_inference_and_save(split, data_dir):
        """
        - YOLO 모델로 예측
        - box가 있으면 각 box마다 patch를 잘라내고 저장
        - ViT로 분류
        """
        results = yolo_model.predict(
            source=data_dir,
            conf=0.25,
            save=True,         # detect된 이미지/결과 저장
            save_txt=True,     # 라벨(txt) 저장
            project=output_dir,
            name=split
        )
        # results: 각 이미지마다의 추론 결과 리스트
        for img_idx, result in enumerate(results):
            if not result.boxes or len(result.boxes) == 0:
                continue

            # 일부 YOLO 버전은 result.image / 일부는 result.orig_img 사용
            # yolov11 커스텀 모델에서 result.image에 원본 이미지가 들어있다고 가정
            # 혹은 필요 시 print(dir(result)) 등을 통해 실제 속성 확인
            if result.image is None:
                continue

            # numpy array라 가정
            img_array = result.image  # shape 예: (H, W, 3)

            # box 좌표 [xmin, ymin, xmax, ymax]가 float or int 형태라 가정
            for box_idx, box in enumerate(result.boxes):
                # box 자체가 [xmin, ymin, xmax, ymax] 형태의 리스트/array라고 가정
                # 필요시 print(box)로 확인
                x1, y1, x2, y2 = map(int, box)  # 정수 변환

                # index 범위가 이미지 범위를 넘어갈 경우 대비
                h, w = img_array.shape[:2]
                x1 = max(0, min(x1, w))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue

                # 객체 영역(패치) 추출
                patch = img_array[y1:y2, x1:x2]

                # (추가) patch 저장
                split_patch_dir = os.path.join(patch_output_dir, split)
                os.makedirs(split_patch_dir, exist_ok=True)
                patch_filename = f"patch_{img_idx}_{box_idx}.png"
                patch_path = os.path.join(split_patch_dir, patch_filename)
                cv2.imwrite(patch_path, patch)

                # ViT 분류
                inputs = vit_feature_extractor(images=patch, return_tensors="pt")
                outputs = vit_model(**inputs)
                predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()
                predicted_class_name = class_names[predicted_class_idx] \
                    if predicted_class_idx < len(class_names) else "Unknown"

                print(f"[{split.upper()}] Detected: {predicted_class_name}"
                      f" at [x1={x1},y1={y1},x2={x2},y2={y2}] -> saved {patch_path}")

    # 4) train / val 데이터 처리
    for split, data_dir in zip(['train', 'val'], [train_dir, val_dir]):
        run_inference_and_save(split, data_dir)

    # 5) unlabeled 데이터(있을 경우) 처리
    unlabeled_dir = dataset_config.get('unlabeled', None)
    if unlabeled_dir:
        split = 'unlabeled'
        results = yolo_model.predict(
            source=unlabeled_dir,
            conf=0.05,
            save=True,
            project=output_dir,
            name=split
        )
        for img_idx, result in enumerate(results):
            if not result.boxes or len(result.boxes) == 0:
                continue

            if result.image is None:
                continue

            img_array = result.image

            for box_idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box)
                h, w = img_array.shape[:2]
                x1 = max(0, min(x1, w))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue

                patch = img_array[y1:y2, x1:x2]

                split_patch_dir = os.path.join(patch_output_dir, split)
                os.makedirs(split_patch_dir, exist_ok=True)
                patch_filename = f"patch_{img_idx}_{box_idx}.png"
                patch_path = os.path.join(split_patch_dir, patch_filename)
                cv2.imwrite(patch_path, patch)

                inputs = vit_feature_extractor(images=patch, return_tensors="pt")
                outputs = vit_model(**inputs)
                predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()
                predicted_class_name = class_names[predicted_class_idx] \
                    if predicted_class_idx < len(class_names) else "Unknown"

                print(f"[UNLABELED] Potential {predicted_class_name}"
                      f" at [x1={x1},y1={y1},x2={x2},y2={y2}] -> saved {patch_path}")

if __name__ == "__main__":
    # 경로 및 파라미터 설정
    yaml_file        = "code/work_cd.yaml"                 # dataset YAML
    yolo_model_path  = "yolo11x-cls.pt"                    # YOLOv11 커스텀 모델
    vit_model_name   = "google/vit-base-patch16-224"       # ViT 모델
    output_dir       = "results"                           # YOLO 추론 결과 저장 경로
    patch_output_dir = "patch_results"                     # 잘라낸 이미지 저장 경로

    process_yolo_with_vit(
        yaml_file,
        yolo_model_path,
        vit_model_name,
        output_dir,
        patch_output_dir
    )