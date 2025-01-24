#testset 하위폴더에서 실행.

import os
import shutil
import random

def rename_files(images_dir, labels_dir):
    """
    이미지 및 라벨 파일을 번호화된 이름으로 변경합니다.
    """
    # 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # 정렬하여 일관성 유지

    total_files = len(image_files)
    if total_files == 0:
        print(f"경고: {images_dir}에 이미지 파일이 없습니다.")
        return

    num_digits = len(str(total_files))  # 파일 번호에 필요한 자릿수 결정 (예: 1000개면 4자리)

    for idx, filename in enumerate(image_files, start=1):
        # 파일 확장자 추출
        extension = os.path.splitext(filename)[1].lower()
        
        # 새로운 파일 이름 생성 (예: 0001.jpg, 0002.png 등)
        new_base = str(idx).zfill(num_digits)
        new_image_name = f"{new_base}{extension}"
        new_label_name = f"{new_base}.txt"

        # 원본 파일 경로
        src_image_path = os.path.join(images_dir, filename)
        # 원본 라벨 파일 경로
        label_filename = os.path.splitext(filename)[0] + '.txt'
        src_label_path = os.path.join(labels_dir, label_filename)

        # 새로운 파일 경로
        dst_image_path = os.path.join(images_dir, new_image_name)
        dst_label_path = os.path.join(labels_dir, new_label_name)

        # 파일 이름 변경 (이미지)
        if not os.path.exists(dst_image_path):
            os.rename(src_image_path, dst_image_path)
        else:
            print(f"파일 이름 충돌: {dst_image_path} 이미 존재합니다.")

        # 파일 이름 변경 (라벨)
        if os.path.exists(src_label_path):
            if not os.path.exists(dst_label_path):
                os.rename(src_label_path, dst_label_path)
            else:
                print(f"라벨 파일 이름 충돌: {dst_label_path} 이미 존재합니다.")
        else:
            print(f"라벨 파일이 존재하지 않음: {src_label_path}")

    print(f"[{images_dir}] 파일 이름 번호화 완료 ({total_files}개)")

def split_data(images_dir, labels_dir, train_ratio=0.8):
    """
    이미지 및 라벨 파일을 train과 val 폴더로 분할합니다.
    """
    # train 및 val 디렉토리 생성
    train_images_dir = os.path.join(images_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')
    train_labels_dir = os.path.join(labels_dir, 'train')
    val_labels_dir = os.path.join(labels_dir, 'val')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # 모든 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # 일관된 순서 유지

    if len(image_files) == 0:
        print(f"경고: {images_dir}에 이미지 파일이 없습니다.")
        return

    # 랜덤 셔플
    random.shuffle(image_files)

    # 분할 인덱스 계산
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # 파일 이동 함수
    def move_files(file_list, src_dir, dst_images_dir, dst_labels_dir):
        for filename in file_list:
            src_image_path = os.path.join(src_dir, filename)
            dst_image_path = os.path.join(dst_images_dir, filename)
            shutil.move(src_image_path, dst_image_path)

            # 라벨 파일 이동
            label_filename = os.path.splitext(filename)[0] + '.txt'
            src_label_path = os.path.join(labels_dir, label_filename)
            if os.path.exists(src_label_path):
                dst_label_path = os.path.join(dst_labels_dir, label_filename)
                shutil.move(src_label_path, dst_label_path)
            else:
                print(f"라벨 파일이 존재하지 않음: {src_label_path}")

    # train으로 이동
    move_files(train_files, images_dir, train_images_dir, train_labels_dir)

    # val으로 이동
    move_files(val_files, images_dir, val_images_dir, val_labels_dir)

    print(f"[{images_dir}] 데이터 분할 완료 (Train: {len(train_files)}, Val: {len(val_files)})")

def process_folder(base_dir, folder_name):
    """
    지정된 폴더 내의 images 및 labels 폴더를 처리합니다.
    """
    images_dir = os.path.join(base_dir, folder_name, 'images')
    labels_dir = os.path.join(base_dir, folder_name, 'labels')

    if not os.path.isdir(images_dir):
        print(f"에러: {images_dir} 디렉토리가 존재하지 않습니다.")
        return
    if not os.path.isdir(labels_dir):
        print(f"에러: {labels_dir} 디렉토리가 존재하지 않습니다.")
        return

    print(f"처리 중: {folder_name}")
    rename_files(images_dir, labels_dir)
    split_data(images_dir, labels_dir)
    print(f"완료: {folder_name}\n")

def main():
    # 현재 스크립트의 디렉토리 경로 (Army 폴더)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 처리할 폴더 목록 (Army 하위의 다섯 개 폴더)
    target_folders = [
        'folder1/subforder',
        'folder2',
        'folder3'
    ]

    for folder in target_folders:
        process_folder(base_dir, folder)

if __name__ == "__main__":
    main()