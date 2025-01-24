import os
import glob
import yaml
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dataset Validation Script for YOLOv11")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--image_dir', type=str, default=None, help='Override image directory in data.yaml')
    parser.add_argument('--label_dir', type=str, default=None, help='Override label directory in data.yaml')
    parser.add_argument('--show', action='store_true', help='Show sample images with labels')
    parser.add_argument('--samples', type=int, default=5, help='Number of sample images to display')
    return parser.parse_args()

def load_data_yaml(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def get_file_pairs(image_dir, label_dir):
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, f'**/*.{ext}'), recursive=True))
    
    file_pairs = []
    for img_path in image_files:
        rel_path = os.path.relpath(img_path, image_dir)
        label_path = os.path.join(label_dir, os.path.splitext(rel_path)[0] + '.txt')
        file_pairs.append((img_path, label_path))
    
    return file_pairs

def validate_labels(file_pairs, nc):
    missing_labels = []
    incorrect_format = []
    class_counts = defaultdict(int)
    for img_path, label_path in file_pairs:
        if not os.path.exists(label_path):
            missing_labels.append(label_path)
            continue
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    incorrect_format.append(label_path)
                    break
                cls = int(parts[0])
                if cls < 0 or cls >= nc:
                    incorrect_format.append(label_path)
                    break
                class_counts[cls] += 1
        except Exception as e:
            incorrect_format.append(label_path)
    
    return missing_labels, incorrect_format, class_counts

def visualize_samples(file_pairs, class_names, samples=5):
    import random
    selected = random.sample(file_pairs, min(samples, len(file_pairs)))
    for img_path, label_path in selected:
        img = Image.open(img_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        ax = plt.gca()
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                cls, x, y, w, h = map(float, parts)
                cls = int(cls)
                # YOLO format to pixel coordinates
                img_width, img_height = img.size
                box_w = w * img_width
                box_h = h * img_height
                box_x = (x * img_width) - (box_w / 2)
                box_y = (y * img_height) - (box_h / 2)
                rect = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(box_x, box_y, class_names[cls], fontsize=12, color='yellow', backgroundcolor='black')
        plt.title(f"Image: {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()

def main():
    args = parse_arguments()
    data = load_data_yaml(args.data)
    
    # Override directories if provided
    train_image_dir = args.image_dir if args.image_dir else data.get('train', '')
    train_label_dir = args.label_dir if args.label_dir else data.get('labels', {}).get('train', '')
    
    val_image_dir = data.get('val', '')
    val_label_dir = data.get('labels', {}).get('val', '')
    
    nc = data.get('nc', 0)
    class_names = data.get('names', [])
    
    if nc != len(class_names):
        print(f"Warning: 'nc' ({nc}) does not match number of class names ({len(class_names)}).")
    
    print("=== Training Set Validation ===")
    train_file_pairs = get_file_pairs(train_image_dir, train_label_dir)
    train_missing, train_incorrect, train_class_counts = validate_labels(train_file_pairs, nc)
    print(f"Total training images: {len(train_file_pairs)}")
    print(f"Missing labels: {len(train_missing)}")
    if len(train_missing) > 0:
        print("Sample missing label files:")
        for path in train_missing[:5]:
            print(f"  {path}")
    print(f"Incorrectly formatted labels: {len(train_incorrect)}")
    if len(train_incorrect) > 0:
        print("Sample incorrectly formatted label files:")
        for path in train_incorrect[:5]:
            print(f"  {path}")
    print("Class distribution in training set:")
    for cls, count in sorted(train_class_counts.items()):
        print(f"  {class_names[cls]} ({cls}): {count}")
    print("")
    
    print("=== Validation Set Validation ===")
    val_file_pairs = get_file_pairs(val_image_dir, val_label_dir)
    val_missing, val_incorrect, val_class_counts = validate_labels(val_file_pairs, nc)
    print(f"Total validation images: {len(val_file_pairs)}")
    print(f"Missing labels: {len(val_missing)}")
    if len(val_missing) > 0:
        print("Sample missing label files:")
        for path in val_missing[:5]:
            print(f"  {path}")
    print(f"Incorrectly formatted labels: {len(val_incorrect)}")
    if len(val_incorrect) > 0:
        print("Sample incorrectly formatted label files:")
        for path in val_incorrect[:5]:
            print(f"  {path}")
    print("Class distribution in validation set:")
    for cls, count in sorted(val_class_counts.items()):
        print(f"  {class_names[cls]} ({cls}): {count}")
    print("")
    
    # Check if there are any issues
    if len(train_missing) > 0 or len(train_incorrect) > 0 or len(val_missing) > 0 or len(val_incorrect) > 0:
        print("⚠️ Dataset has issues. Please fix the missing or incorrectly formatted label files before training.")
    else:
        print("✅ Dataset validation passed. No missing or incorrectly formatted label files found.")
    
    # Visualize samples if requested
    if args.show:
        print("=== Visualizing Sample Training Images ===")
        visualize_samples(train_file_pairs, class_names, samples=args.samples)
        print("=== Visualizing Sample Validation Images ===")
        visualize_samples(val_file_pairs, class_names, samples=args.samples)

if __name__ == "__main__":
    main()
