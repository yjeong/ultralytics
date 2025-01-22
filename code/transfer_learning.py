import subprocess
import os

def run_training(data_yaml, model_path, epochs, imgsz, freeze, batch, cache, project, name, device, hyperparams):
    """
    주어진 설정으로 YOLOv11 학습을 실행하는 함수.
    """
    command = [
        "yolo", "detect", "train",
        f"data={data_yaml}",
        f"model={model_path}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"freeze={freeze}",
        f"batch={batch}",
        f"cache={cache}",
        f"project={project}",
        f"name={name}",
        f"device={device}"
    ]
    
    # 하이퍼파라미터를 key=value 형식으로 명령어에 추가
    for key, value in hyperparams.items():
        command.append(f"{key}={value}")
    
    print(f"Starting training for {data_yaml}...")
    print("Executing command:", ' '.join(command))
    
    try:
        # 명령어 실행
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # STDOUT 출력
        print(f"STDOUT for {name}:\n{result.stdout}")
        
        # STDERR 출력 (경고나 에러 메시지)
        if result.stderr:
            print(f"STDERR for {name}:\n{result.stderr}")
        
        print(f"Training completed for {data_yaml}. Results saved to {os.path.join(project, name)}\n")
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during training for {data_yaml}:")
        print(e.stdout)
        print(e.stderr)
        print("Skipping to next training.\n")

def main():
    # 학습할 데이터셋 목록
    data_files = [
        "xView.yaml",
        "VisDrone.yaml"
    ]
    
    # 공통 학습 설정
    model_path = "yolo11x.pt"
    epochs = 100
    imgsz = 640
    freeze = 10
    batch = 4
    cache = "True"
    project = "/mnt/nas4/jyh/ultralytics/results"
    device = 1
    
    # 각 데이터셋에 대해 고유한 이름 지정
    names = [
        "xView",
        "VisDrone"
    ]
    
    # 하이퍼파라미터 설정
    hyperparams = {
        # 학습률 관련
        "lr0": 0.01,
        "lrf": 0.01,

        # 옵티마이저 관련
        "momentum": 0.937,
        "weight_decay": 0.0005,

        # 워밍업 단계
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,

        # 손실 함수 가중치
        "box": 0.05,
        "cls": 0.5,
        "kobj": 1.0,

        # IoU 및 앵커 관련
        "iou": 0.15,

        # 데이터 증강
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,

        # 이미지 뒤집기
        "flipud": 0.0,
        "fliplr": 0.5,

        # 기타 증강
        "mosaic": 0.1,
        "mixup": 0.0,
        "copy_paste": 0.0
    }
    
    # 데이터셋과 이름을 매칭하여 학습 실행
    for data_yaml, name in zip(data_files, names):
        run_training(
            data_yaml=data_yaml,
            model_path=model_path,
            epochs=epochs,
            imgsz=imgsz,
            freeze=freeze,
            batch=batch,
            cache=cache,
            project=project,
            name=name,
            device=device,
            hyperparams=hyperparams
        )

if __name__ == "__main__":
    main()
