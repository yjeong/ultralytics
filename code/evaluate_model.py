# evaluate_model.py

from ultralytics import YOLO
import yaml

# YAML 파일 파싱
def parse_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data

# 모델 평가 함수
def evaluate_model(yaml_file, model_path, imgsz=64, batch=16, device=0):
    # YOLO 모델 로드 (Classification Task로 설정)
    model = YOLO(model_path, task='classify')
    
    # 평가 수행
    results = model.val(data=yaml_file, imgsz=imgsz, batch=batch, device=device, workers=8)
    
    print(f"Validation Loss: {results['loss']:.4f}")
    print(f"Validation Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate YOLO Classification Model")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained YOLO model file')
    parser.add_argument('--imgsz', type=int, default=64, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=int, default=0, help='GPU device number')

    args = parser.parse_args()

    evaluate_model(
        yaml_file=args.dataset,
        model_path=args.model_path,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )

    '''
    python evaluate_model.py --dataset /mnt/nas4/jyh/ultralytics/code/work_cd.yaml --model_path yolov8n-vit-classifier-epoch10.pt --imgsz 64 --batch 16 --device 0

    '''
