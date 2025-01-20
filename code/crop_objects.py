import os
import cv2

##############################################################################
# 1) 유틸리티 함수들
##############################################################################

def convert_yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """
    YOLO 포맷(cx, cy, w, h) [정규화 0~1] -> 절대 픽셀 좌표 (x1, y1, x2, y2)
    xc, yc, w, h: float, [0~1]
    img_w, img_h: int, 실제 이미지 폭, 높이
    """
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    return x1, y1, x2, y2

def make_dirs_for_classes(base_dir, class_names):
    """
    base_dir/train/, base_dir/val/ 내에
    각 클래스별 폴더(0,1,2,... 또는 직접 문자열)를 생성한다.
    예: base_dir/train/0, base_dir/train/1, ...
    """
    for split in ["train", "val"]:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        # 여기서는 인덱스(0~9)로 폴더 생성
        for i, cname in enumerate(class_names):
            class_dir = os.path.join(split_dir, str(i))
            os.makedirs(class_dir, exist_ok=True)

def crop_and_save_objects(
    src_img_dir, 
    src_label_dir,
    dst_base_dir, 
    split,
    class_names,
    min_size=10  # 최소 픽셀 크기를 10으로 설정
):
    """
    이미지/라벨( YOLO 형식 )을 읽어, 객체를 Crop -> 클래스별 폴더에 저장.

    Parameters
    ----------
    src_img_dir : str
        예: 'datasets/testset/cctv_day/images/train'
    src_label_dir : str
        예: 'datasets/testset/cctv_day/labels/train'
    dst_base_dir : str
        최종 크롭 이미지가 저장될 상위 폴더
        예: 'datasets/testset/cctv_day_cls'
    split : str
        "train" or "val"
    class_names : list
        클래스 이름 리스트(10개 등)
    min_size : int
        bbox가 너무 작은 경우(가로or세로 픽셀 < min_size) 스킵
    """
    dst_split_dir = os.path.join(dst_base_dir, split)

    img_files = sorted(os.listdir(src_img_dir))
    for img_file in img_files:
        # 확장자 체크
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 이미지 경로
        img_path = os.path.join(src_img_dir, img_file)
        # 동일 이름 .txt 라벨
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(src_label_dir, base_name + ".txt")

        # 라벨 파일이 없으면 스킵
        if not os.path.exists(label_path):
            continue

        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W, _ = img.shape

        # 라벨(.txt) 읽기
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError as e:
            print(f"[WARNING] Skipping {label_path} due to decoding error: {e}")
            continue

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue

            class_idx = int(parts[0])  # YOLO class index
            xc, yc, bw, bh = map(float, parts[1:])

            # 클래스 인덱스 유효성
            if class_idx < 0 or class_idx >= len(class_names):
                continue

            # 좌표 변환
            x1, y1, x2, y2 = convert_yolo_to_xyxy(xc, yc, bw, bh, W, H)

            # 유효 bbox 체크
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                # 너무 작으면 스킵
                continue

            # Crop
            cropped = img[y1:y2, x1:x2]

            # 저장 파일명 : base_name_i.jpg
            out_filename = f"{base_name}_{i}.jpg"
            class_dir = os.path.join(dst_split_dir, str(class_idx))
            out_path = os.path.join(class_dir, out_filename)

            cv2.imwrite(out_path, cropped)


##############################################################################
# 2) 주요 빌드 함수
##############################################################################

def build_classification_dataset(
    src_root="datasets/testset/cctv_day",
    dst_root="datasets/testset/cctv_day_cls",
    class_names=None
):
    """
    src_root 예:
       'datasets/testset/cctv_day'
       내부에 images/train, images/val, labels/train, labels/val 구조
    dst_root 예:
       'datasets/testset/cctv_day_cls'
       최종적으로 train/0..9, val/0..9 폴더가 생성될 경로
    class_names : 10개 클래스 이름 리스트
    """
    if class_names is None:
        raise ValueError("class_names list is required!")
    
    # 1) 분류용 폴더 생성 (train/0..9, val/0..9)
    make_dirs_for_classes(dst_root, class_names)

    # 2) train 세트 크롭
    src_img_train = os.path.join(src_root, "images/train")
    src_lbl_train = os.path.join(src_root, "labels/train")
    crop_and_save_objects(
        src_img_dir=src_img_train,
        src_label_dir=src_lbl_train,
        dst_base_dir=dst_root,
        split="train",
        class_names=class_names,
        min_size=10  # 최소 픽셀 크기 설정
    )

    # 3) val 세트 크롭
    src_img_val = os.path.join(src_root, "images/val")
    src_lbl_val = os.path.join(src_root, "labels/val")
    crop_and_save_objects(
        src_img_dir=src_img_val,
        src_label_dir=src_lbl_val,
        dst_base_dir=dst_root,
        split="val",
        class_names=class_names,
        min_size=10  # 최소 픽셀 크기 설정
    )

    print(f"[INFO] Done building classification dataset from {src_root} -> {dst_root}")


##############################################################################
# 3) 메인 실행부
##############################################################################

if __name__ == "__main__":
    # 10개 클래스명
    CLASSES_10 = [
        "class1",
        "class2",
        "class3",
        "class4",
        "class5",
        "class6",
        "class7",
        "class8",
        "class9",
        "class10"
    ]

    # 여러 디렉토리 반복 처리
    jobs = [
        {
            "src": "datasets/coco8",
            "dst": "datasets/coco8_cls"
        },
        {
            "src": "datasets/coco8",
            "dst": "datasets/coco8_cls"
        }
    ]

    for job in jobs:
        print(f"\n[INFO] Processing: {job['src']} -> {job['dst']}")
        build_classification_dataset(
            src_root=job["src"],
            dst_root=job["dst"],
            class_names=CLASSES_10
        )
