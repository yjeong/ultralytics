# check_corrupt_files.py

import os
from PIL import Image

def check_images(image_dirs):
    corrupt_images = []
    for img_dir in image_dirs:
        if not os.path.isdir(img_dir):
            print(f"디렉토리가 존재하지 않습니다: {img_dir}")
            continue

        for img_file in os.listdir(img_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(img_dir, img_file)
                try:
                    img = Image.open(img_path)
                    img.verify()  # 이미지 파일이 정상인지 확인
                except Exception as e:
                    print(f"손상된 이미지 파일: {img_path} - 오류: {e}")
                    corrupt_images.append(img_path)
    return corrupt_images

def check_labels(label_dirs):
    corrupt_labels = []
    for lbl_dir in label_dirs:
        if not os.path.isdir(lbl_dir):
            print(f"디렉토리가 존재하지 않습니다: {lbl_dir}")
            continue

        for lbl_file in os.listdir(lbl_dir):
            if lbl_file.endswith('.txt'):
                lbl_path = os.path.join(lbl_dir, lbl_file)
                try:
                    with open(lbl_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                raise ValueError("라벨 파일 형식 오류")
                            class_id, x, y, w, h = parts
                            float(x)
                            float(y)
                            float(w)
                            float(h)
                            int(class_id)
                except Exception as e:
                    print(f"손상된 라벨 파일: {lbl_path} - 오류: {e}")
                    corrupt_labels.append(lbl_path)
    return corrupt_labels

def delete_files(file_list, file_type="파일"):
    for file in file_list:
        try:
            os.remove(file)
            print(f"삭제 완료: {file}")
        except Exception as e:
            print(f"{file_type} 삭제 실패: {file} - 오류: {e}")

def main():
    image_dirs = [
        'datasets/coco8/images/train',
        'datasets/coco8/images/val',
        'datasets/coco8/images'
    ]

    label_dirs = [
        'datasets/coco8/labels/train',
        'datasets/coco8/labels/val'
    ]

    print("이미지 파일 검증 중...")
    corrupt_images = check_images(image_dirs)
    print(f"\n총 손상된 이미지 파일: {len(corrupt_images)}")
    
    print("\n라벨 파일 검증 중...")
    corrupt_labels = check_labels(label_dirs)
    print(f"\n총 손상된 라벨 파일: {len(corrupt_labels)}")

    # 손상된 파일 삭제
    if corrupt_images:
        print("\n손상된 이미지 파일 삭제 중...")
        delete_files(corrupt_images, file_type="이미지")

    if corrupt_labels:
        print("\n손상된 라벨 파일 삭제 중...")
        delete_files(corrupt_labels, file_type="라벨")

if __name__ == "__main__":
    main()
