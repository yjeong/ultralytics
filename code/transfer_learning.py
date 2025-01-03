import subprocess
import os

def run_training(data_yaml, model_path, epochs, imgsz, freeze, batch, cache, project, name):
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
        f"name={name}"
    ]
    
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
        "work_cd.yaml",
        "work_cn.yaml",
        "work_td.yaml",
        "work_tn.yaml"
    ]
    
    # 공통 학습 설정
    model_path = "yolo11x.pt"
    epochs = 1
    imgsz = 640
    freeze = 10
    batch = 16
    cache = "True"
    project = "/mnt/nas4/jyh/ultralytics/results"
    
    # 각 데이터셋에 대해 고유한 이름 지정
    names = ["frozen_transfer_cd", "frozen_transfer_cn", "frozen_transfer_td", "frozen_transfer_tn"]
    
    # 데이터셋과 이름을 매칭하여 학습 실행
    for data_yaml, name in zip(data_files, names):
        run_training(data_yaml, model_path, epochs, imgsz, freeze, batch, cache, project, name)

if __name__ == "__main__":
    main()
