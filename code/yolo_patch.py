# yolo_patching.py

import os
import yaml
import cv2
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd
import logging

# ================================================
# 설정 섹션
# ================================================

name = "VOC"  # 데이터셋 이름

MODELS_CONFIG = [
    {
        "model_path": "/mnt/nas4/jyh/ultralytics/results/"+name+"/weights/best.pt",
        "data_yaml": "/mnt/nas4/jyh/ultralytics/ultralytics/cfg/datasets/"+name+".yaml"
    }
]

BASE_OUTPUT_DIR = "/mnt/nas4/jyh/ultralytics/ViT_results/ViT_"+name  # 기본 결과를 저장할 디렉토리 경로

# 파라미터 설정
MIN_IMAGE_SIZE = 10  # 처리할 최소 이미지 크기 (픽셀)
CONF_THRESHOLD = 0.25  # YOLO 검출 신뢰도 임계값
SAVE_PATCHES = True  # 검출된 객체의 패치를 저장할지 여부

# ================================================
# 로깅 설정
# ================================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ================================================
# 함수 정의
# ================================================

def load_yolo_model(model_path):
    logging.info("YOLO 모델 로드 중: %s", model_path)
    model = YOLO(model_path)
    return model

def load_data_yaml(data_yaml_path):
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data

def get_image_paths(images_dirs):
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    image_paths = []
    for images_dir in images_dirs:
        logging.debug("탐색 중인 이미지 디렉토리: %s", images_dir)
        if not os.path.exists(images_dir):
            logging.warning("이미지 디렉토리가 존재하지 않습니다: %s", images_dir)
            continue
        for ext in image_extensions:
            for root, _, files in os.walk(images_dir):
                for file in files:
                    if file.lower().endswith(ext):
                        image_paths.append(os.path.join(root, file))
    logging.debug("발견된 총 이미지 수: %d", len(image_paths))
    return image_paths

def crop_patch(image, bbox):
    """
    이미지에서 바운딩 박스에 해당하는 패치를 잘라냅니다.
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = map(int, bbox)
    patch = image[y1:y2, x1:x2]
    return patch

def main():
    # ================================================
    # 사용자 정의: 데이터셋 루트 디렉토리 수동 설정
    # ================================================
    # 여기에 실제 데이터셋 루트 디렉토리를 설정하세요.
    DATASET_ROOT = "/mnt/nas4/jyh/ultralytics/datasets/"+name

    # ================================================
    # 각 모델별로 처리
    # ================================================
    for config in MODELS_CONFIG:
        model_path = config["model_path"]
        data_yaml_path = config["data_yaml"]

        # 모델 이름 추출
        model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        logging.info("=== 모델: %s ===", model_name)

        # YAML 파일 로드
        data_yaml = load_data_yaml(data_yaml_path)
        class_names = data_yaml.get('names', {})
        nc = data_yaml.get('nc', len(class_names))

        # 'names'가 딕셔너리인지 확인
        if not isinstance(class_names, dict):
            logging.error("'names'은 딕셔너리여야 합니다.")
            continue  # 다음 모델로 넘어감

        # 'names'의 키를 정수로 변환하고, 새로운 딕셔너리 생성
        int_class_names = {}
        for k, v in class_names.items():
            if isinstance(k, str) and k.isdigit():
                int_class_names[int(k)] = v
            elif isinstance(k, int):
                int_class_names[k] = v
            else:
                logging.warning("클래스 키가 숫자 문자열이나 정수가 아닙니다: %s", k)
        class_names = int_class_names

        # 'Unknown' 클래스가 존재하는지 확인
        if 'Unknown' not in class_names.values():
            try:
                # 현재 최대 클래스 인덱스를 찾기
                if class_names:
                    new_class_index = max(class_names.keys()) + 1
                else:
                    new_class_index = 0
                class_names[new_class_index] = 'Unknown'
                nc += 1
                logging.info("'Unknown' 클래스가 추가되었습니다. 클래스 인덱스: %d", new_class_index)
            except ValueError:
                logging.error("'names' 딕셔너리의 키가 정수로 변환할 수 없습니다.")
                continue

        # 'path' 키를 무시하고, 직접 dataset_root 설정
        dataset_root = DATASET_ROOT
        logging.info("Dataset root directory: %s", dataset_root)

        # 테스트 데이터셋 경로 확인
        test_image_dirs = []
        if 'test' in data_yaml:
            if isinstance(data_yaml['test'], dict):
                if 'images' in data_yaml['test']:
                    test_image_dirs = data_yaml['test']['images']
                else:
                    logging.error("'test' 딕셔너리에 'images' 키가 없습니다.")
                    continue
            elif isinstance(data_yaml['test'], str):
                test_image_dirs = [data_yaml['test']]
            elif isinstance(data_yaml['test'], list):
                test_image_dirs = data_yaml['test']
            else:
                logging.error("'test' 키의 타입이 지원되지 않습니다.")
                continue
        else:
            # 'test' 섹션이 없으면 'val' 섹션 사용
            logging.warning("'test' 섹션이 YAML 파일에 없습니다. 'val' 섹션을 테스트 데이터로 사용합니다.")
            if 'val' in data_yaml:
                if isinstance(data_yaml['val'], dict):
                    if 'images' in data_yaml['val']:
                        test_image_dirs = data_yaml['val']['images']
                    else:
                        logging.error("'val' 딕셔너리에 'images' 키가 없습니다.")
                        continue
                elif isinstance(data_yaml['val'], str):
                    test_image_dirs = [data_yaml['val']]
                elif isinstance(data_yaml['val'], list):
                    test_image_dirs = data_yaml['val']
                else:
                    logging.error("'val' 키의 타입이 지원되지 않습니다.")
                    continue
            else:
                logging.error("테스트 이미지 디렉토리가 정의되지 않았습니다.")
                continue  # 다음 모델로 넘어감

        # 클래스 수와 클래스 이름 확인
        if nc != len(class_names):
            logging.warning("'nc' (%d)와 클래스 이름의 개수 (%d)가 일치하지 않습니다.", nc, len(class_names))

        # 모든 테스트 이미지 경로 수집
        test_image_paths = []
        for img_dir in test_image_dirs:
            img_dir_full = os.path.join(dataset_root, img_dir)
            test_image_paths.extend(get_image_paths([img_dir_full]))

        logging.info("총 테스트 이미지 수: %d", len(test_image_paths))

        # 이미지 디렉토리가 존재하지 않으면 스킵
        if not test_image_paths:
            logging.warning("테스트 이미지 디렉토리에 이미지가 없습니다: %s", img_dir_full)
            continue

        # 각 모델별 출력 디렉토리 설정
        model_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        patches_dir = os.path.join(model_output_dir, 'patches')
        os.makedirs(patches_dir, exist_ok=True)
        results_csv = os.path.join(model_output_dir, 'patches_metadata.csv')

        # YOLO 모델 로드
        yolo_model = load_yolo_model(model_path)

        # 결과 저장을 위한 리스트 (모델별)
        model_results = []

        # 메인 루프: 각 이미지에 대해 객체 검출 및 패치 저장
        for idx, img_path in enumerate(tqdm(test_image_paths, desc=f"{model_name} - 이미지 처리 중")):
            try:
                # 이미지 읽기
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning("이미지를 읽을 수 없습니다: %s. 스킵합니다.", img_path)
                    continue
                height, width, _ = img.shape
                if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                    logging.info("스킵: 이미지 크기 (%dx%d)가 최소 크기 %d보다 작습니다. %s", width, height, MIN_IMAGE_SIZE, img_path)
                    continue

                # YOLO로 객체 검출
                detections = yolo_model(img_path, conf=CONF_THRESHOLD)[0]  # 배치 크기 1로 가정

                if len(detections.boxes) == 0:
                    continue  # 검출된 객체가 없으면 건너뜀

                # 검출된 객체 처리
                for det in detections.boxes:
                    # 바운딩 박스 추출
                    bbox = det.xyxy.tolist()
                    if isinstance(bbox[0], list):
                        bbox = bbox[0]
                    x1, y1, x2, y2 = map(int, bbox)
                    confidence = det.conf.item()
                    pred_cls_idx = int(det.cls.item())
                    pred_cls_name = class_names.get(pred_cls_idx, "Unknown")

                    # 패치 자르기
                    patch = crop_patch(img, [x1, y1, x2, y2])

                    # 패치 저장 (옵션)
                    if SAVE_PATCHES:
                        base_name = os.path.splitext(os.path.basename(img_path))[0]
                        patch_filename = f"{base_name}_{x1}_{y1}_{x2}_{y2}.jpg"
                        patch_save_path = os.path.join(patches_dir, patch_filename)
                        cv2.imwrite(patch_save_path, patch)

                    # 결과 리스트에 추가
                    model_results.append({
                        'patch_path': patch_save_path if SAVE_PATCHES else "",
                        'original_image': img_path,
                        'bbox': f"{x1},{y1},{x2},{y2}",
                        'pred_class_yolo': pred_cls_name,
                        'confidence_yolo': confidence
                    })
            except Exception as e:
                logging.error("이미지 처리 중 오류 발생: %s. 스킵합니다.", img_path)
                continue

        # ================================================
        # 결과 저장
        # ================================================

        if model_results:
            # 결과를 DataFrame으로 변환하고 CSV로 저장
            results_df = pd.DataFrame(model_results)
            results_df.to_csv(results_csv, index=False)
            logging.info("패치 메타데이터가 %s에 저장되었습니다.", results_csv)
        else:
            logging.warning("%s에 대한 패치 결과가 없습니다.", model_name)

        logging.info("%s 패칭이 완료되었습니다.", model_name)

# ================================================
# 메인 함수 실행
# ================================================

if __name__ == "__main__":
    main()
