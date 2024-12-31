import os

def remap_labels(labels_dir, mapping):
    """
    라벨 파일의 클래스 번호를 매핑에 따라 수정합니다.
    
    :param labels_dir: 라벨 파일들이 있는 디렉토리 경로
    :param mapping: 기존 클래스 번호를 새로운 클래스 번호로 매핑한 딕셔너리
    """
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(labels_dir, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"잘못된 라벨 형식: {filepath}")
                    continue
                cls = int(parts[0])
                if cls in mapping:
                    new_cls = mapping[cls]
                    new_line = f"{new_cls} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
                else:
                    print(f"매핑되지 않은 클래스 번호 {cls} in {filepath}")
                    # 필요한 경우, 해당 클래스를 무시하거나 다른 처리 수행
                    continue
            
            with open(filepath, 'w') as file:
                file.writelines(new_lines)

def main():
    # Base directory (Army 폴더)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 각 데이터셋별 매핑 정의
    dataset_mappings = {
        'army_add': {4: 0, 8: 1},        # 기존 클래스 번호 4 → 0, 8 → 1
        'cctv_day': {0: 0, 1: 1, 2: 2}, # 기존 클래스 번호 0 → 0, 1 → 1, 2 → 2
        'cctv_night': {0: 0, 1: 1, 2: 2},
        'tod_day': {0: 0, 1: 1},
        'tod_night': {0: 0, 1: 1}
    }

    for dataset, mapping in dataset_mappings.items():
        labels_train_dir = os.path.join(base_dir, dataset, 'labels', 'train')
        labels_val_dir = os.path.join(base_dir, dataset, 'labels', 'val')
        
        if os.path.isdir(labels_train_dir):
            print(f"Processing {labels_train_dir}")
            remap_labels(labels_train_dir, mapping)
        
        if os.path.isdir(labels_val_dir):
            print(f"Processing {labels_val_dir}")
            remap_labels(labels_val_dir, mapping)
    
    print("라벨 클래스 번호 매핑 완료")

if __name__ == "__main__":
    main()
