# yolo_patching.py

import os
import yaml
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import pandas as pd

# ================================================
# 설정 섹션
# ================================================

# 모델과 해당하는 YAML 파일의 매핑 리스트
MODELS_CONFIG = [
    {
        "model_path": "/mnt/nas4/jyh/ultralytics/results/frozen_transfer_cd5/weights/best.pt",
        "data_yaml": "/mnt/nas4/jyh/ultralytics/code/work_cd.yaml"
    },
    {
        "model_path": "/mnt/nas4/jyh/ultralytics/results/frozen_transfer_cn5/weights/best.pt",
        "data_yaml": "/mnt/nas4/jyh/ultralytics/code/work_cn.yaml"
    }
]

BASE_OUTPUT_DIR = "/mnt/nas4/jyh/ultralytics/results_vit"  # 기본 결과를 저장할 디렉토리 경로

# 파라미터 설정
MIN_IMAGE_SIZE = 10  # 처리할 최소 이미지 크기 (픽셀)
CONF_THRESHOLD = 0.25  # YOLO 검출 신뢰도 임계값
SAVE_PATCHES = True  # 검출된 객체의 패치를 저장할지 여부

# ================================================
# 함수 정의
# ================================================

def load_yolo_model(model_path):
    print(f"YOLO 모델 로드 중: {model_path}...")
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
        print(f"[DEBUG] 탐색 중인 이미지 디렉토리: {images_dir}")  # 디버깅 출력
        if not os.path.exists(images_dir):
            print(f"[WARNING] 이미지 디렉토리가 존재하지 않습니다: {images_dir}")
            continue
        for ext in image_extensions:
            for root, _, files in os.walk(images_dir):
                for file in files:
                    if file.lower().endswith(ext):
                        image_paths.append(os.path.join(root, file))
    print(f"[DEBUG] 발견된 총 이미지 수: {len(image_paths)}")  # 디버깅 출력
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
    # 각 모델별로 처리
    for config in MODELS_CONFIG:
        model_path = config["model_path"]
        data_yaml_path = config["data_yaml"]
        
        # 모델 이름 추출 (예: frozen_transfer_cd3)
        model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        print(f"\n=== 모델: {model_name} ===")
        
        # YAML 파일 로드
        data_yaml = load_data_yaml(data_yaml_path)
        class_names = data_yaml.get('names', [])
        nc = data_yaml.get('nc', len(class_names))
        
        # 'Unknown' 클래스를 미리 추가 (한 번만)
        if 'Unknown' not in class_names:
            class_names.append('Unknown')
            nc += 1
        
        # 테스트 데이터셋 경로 확인
        test_image_dirs = []
        if 'test' in data_yaml:
            test_image_dirs = data_yaml['test']['images']
        else:
            # 'test' 섹션이 없으면 'val' 섹션 사용
            print("Warning: 'test' 섹션이 YAML 파일에 없습니다. 'val' 섹션을 테스트 데이터로 사용합니다.")
            test_image_dirs = data_yaml.get('val', [])
        
        if not test_image_dirs:
            print("Error: 테스트 이미지 디렉토리가 정의되지 않았습니다.")
            continue  # 다음 모델로 넘어감
        
        # 클래스 수와 클래스 이름 확인
        if nc != len(class_names):
            print(f"Warning: 'nc' ({nc})와 클래스 이름의 개수 ({len(class_names)})가 일치하지 않습니다.")
        
        # 모든 테스트 이미지 경로 수집
        test_image_paths = []
        for img_dir in test_image_dirs:
            img_dir_full = os.path.join(os.path.dirname(data_yaml_path), img_dir)
            test_image_paths.extend(get_image_paths([img_dir_full]))
        
        print(f"총 테스트 이미지 수: {len(test_image_paths)}")
        
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
        for img_path in tqdm(test_image_paths, desc=f"{model_name} - 이미지 처리 중"):
            # 이미지 읽기
            img = cv2.imread(img_path)
            if img is None:
                print(f"경고: 이미지를 읽을 수 없습니다. {img_path}. 스킵합니다.")
                continue
            height, width, _ = img.shape
            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                print(f"스킵: 이미지 크기 ({width}x{height})가 최소 크기 {MIN_IMAGE_SIZE}보다 작습니다. {img_path}")
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
                pred_cls_name = class_names[pred_cls_idx] if pred_cls_idx < len(class_names) else "Unknown"
                
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
        
        # ================================================
        # 결과 저장
        # ================================================
        
        if model_results:
            # 결과를 DataFrame으로 변환하고 CSV로 저장
            results_df = pd.DataFrame(model_results)
            results_df.to_csv(results_csv, index=False)
            print(f"패치 메타데이터가 {results_csv}에 저장되었습니다.")
        else:
            print(f"[WARNING] {model_name}에 대한 패치 결과가 없습니다.")
        
        print(f"{model_name} 패칭이 완료되었습니다.")

# ================================================
# 메인 함수 실행
# ================================================

if __name__ == "__main__":
    main()
