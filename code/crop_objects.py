import os
import cv2

# 10개 클래스 이름 (인덱스 0~9)
CLASS_NAMES = [
    "Civilian ship",
    "Warship",
    "Submarine",
    "Inflatable boat",
    "Person",
    "Aircraft",
    "Rotary-wing",
    "Drone",
    "Standard buoy",
    "Vehicle"
]

def make_dirs_for_classes(base_dir, class_names):
    """
    base_dir/train/, base_dir/val/ 내에
    각 클래스별 폴더(0,1,2,... 혹은 실제 이름)를 생성한다.
    """
    for split in ["train", "val"]:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i, cname in enumerate(class_names):
            # 여기서는 인덱스 이름을 폴더로 사용 (i)
            # 만약 실제 클래스명으로 폴더를 만들고 싶다면 cname 사용
            class_folder = str(i)
            class_dir = os.path.join(split_dir, class_folder)
            os.makedirs(class_dir, exist_ok=True)

def convert_yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """
    YOLO 포맷(cx, cy, w, h) [정규화 0~1] -> 절대 픽셀 좌표 x1,y1,x2,y2
    """
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    return x1, y1, x2, y2

def crop_and_save_objects(
    src_img_dir, src_label_dir,
    dst_base_dir, split,
    class_names,
    min_size=5
):
    """
    src_img_dir : 원본 이미지 폴더 (ex: images/train)
    src_label_dir : YOLO 라벨 폴더 (ex: labels/train)
    dst_base_dir : 결과 저장할 폴더 (ex: my_data_cls)
    split : "train" or "val"
    class_names : CLASS_NAMES 리스트 (총 10개)
    min_size : bbox가 너무 작을 경우 스킵하는 임계값(픽셀)
    """
    dst_split_dir = os.path.join(dst_base_dir, split)

    img_files = sorted(os.listdir(src_img_dir))
    for img_file in img_files:
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        base_name = os.path.splitext(img_file)[0]

        img_path = os.path.join(src_img_dir, img_file)
        label_path = os.path.join(src_label_dir, base_name + ".txt")

        if not os.path.exists(label_path):
            # 라벨 파일이 없으면 스킵
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W, _ = img.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue

            class_idx = int(parts[0])
            # YOLO 포맷 cx, cy, w, h
            xc, yc, bw, bh = map(float, parts[1:])

            if class_idx < 0 or class_idx >= len(class_names):
                # 범위 밖 클래스
                continue

            # BBox 좌표 변환
            x1, y1, x2, y2 = convert_yolo_to_xyxy(xc, yc, bw, bh, W, H)
            # 유효성 체크
            if x2 <= x1 or y2 <= y1:
                continue
            # 너무 작은 bbox는 스킵 (옵션)
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                continue

            cropped = img[y1:y2, x1:x2]

            # 저장 예: img001_0.jpg
            out_filename = f"{base_name}_{i}.jpg"
            # class 폴더 (여기서는 숫자 인덱스 이용)
            class_dir = os.path.join(dst_split_dir, str(class_idx))
            out_path = os.path.join(class_dir, out_filename)

            cv2.imwrite(out_path, cropped)

def build_classification_dataset(
    src_root="datasets/testset/cctv_day",
    dst_root="datasets/testset/cctv_day_cls",
    class_names=None
):
    """
    src_root 구조:
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt

    dst_root 최종 구조:
      train/0/*.jpg
      ...
      val/0/*.jpg
      ...
    """
    if class_names is None:
        raise ValueError("class_names list is required!")
    
    # 1) 분류용 폴더 생성
    make_dirs_for_classes(dst_root, class_names)

    # 2) train 크롭
    src_img_train = os.path.join(src_root, "images/train")
    src_lbl_train = os.path.join(src_root, "labels/train")
    crop_and_save_objects(
        src_img_dir=src_img_train,
        src_label_dir=src_lbl_train,
        dst_base_dir=dst_root,
        split="train",
        class_names=class_names
    )

    # 3) val 크롭
    src_img_val = os.path.join(src_root, "images/val")
    src_lbl_val = os.path.join(src_root, "labels/val")
    crop_and_save_objects(
        src_img_dir=src_img_val,
        src_label_dir=src_lbl_val,
        dst_base_dir=dst_root,
        split="val",
        class_names=class_names
    )

if __name__ == "__main__":
    # 사용자 환경에 맞춰 경로/클래스 리스트 설정
    CLASSES_10 = [
        "Civilian ship",
        "Warship",
        "Submarine",
        "Inflatable boat",
        "Person",
        "Aircraft",
        "Rotary-wing",
        "Drone",
        "Standard buoy",
        "Vehicle"
    ]

    build_classification_dataset(
        src_root="datasets/testset/cctv_day",      # 원본 (images/ + labels/)
        dst_root="datasets/testset/cctv_day_cls",  # 결과 (train/0,1..9 + val/0,1..9)
        class_names=CLASSES_10
    )

    print("[INFO] Cropping done! Check 'datasets/my_data_cls/'.")
