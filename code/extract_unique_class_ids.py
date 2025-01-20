# extract_unique_classes.py

import os

def extract_unique_class_ids(label_dirs):
    class_ids = set()
    for lbl_dir in label_dirs:
        if not os.path.isdir(lbl_dir):
            print(f"디렉토리가 존재하지 않습니다: {lbl_dir}")
            continue

        for lbl_file in os.listdir(lbl_dir):
            if lbl_file.endswith('.txt'):
                lbl_path = os.path.join(lbl_dir, lbl_file)
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 1:
                            continue
                        try:
                            class_id = int(parts[0])
                            class_ids.add(class_id)
                        except ValueError:
                            print(f"잘못된 클래스 ID: {parts[0]} in {lbl_path}")
    return sorted(list(class_ids))

def main():
    label_dirs = [
        '/mnt/nas4/jyh/ultralytics/ultralytics/examples/datasets/coco8/labels/train',
        '/mnt/nas4/jyh/ultralytics/ultralytics/examples/datasets/coco8/labels/val'
    ]

    unique_class_ids = extract_unique_class_ids(label_dirs)
    print("고유 클래스 ID:", unique_class_ids)
    print("클래스 이름을 아래 리스트에 추가하세요:")
    for class_id in unique_class_ids:
        print(f"{class_id}: class{class_id}")

if __name__ == "__main__":
    main()
