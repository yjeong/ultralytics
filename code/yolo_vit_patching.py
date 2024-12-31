import os
import torch
from ultralytics import YOLO
from transformers import ViTForImageClassification, ViTFeatureExtractor
import yaml

def load_yaml(yaml_path):
    """
    Load a YAML file and return the parsed data.
    """
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def process_yolo_with_vit(yaml_file, yolo_model_path, vit_model_name, output_dir):
    """
    Process images with YOLO for object detection and classify detected patches with ViT.

    Args:
        yaml_file (str): Path to the dataset YAML file.
        yolo_model_path (str): Path to the pre-trained YOLO model.
        vit_model_name (str): Name of the pre-trained ViT model (Hugging Face).
        output_dir (str): Directory to save results.
    """
    # Load dataset configuration from YAML
    dataset_config = load_yaml(yaml_file)
    train_dir = dataset_config['train'][0]
    val_dir = dataset_config['val'][0]
    class_names = dataset_config['names']

    # Initialize YOLO and ViT models
    yolo_model = YOLO(yolo_model_path)
    vit_model = ViTForImageClassification.from_pretrained(vit_model_name)
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    # Process labeled data (train and val)
    for split, split_dir in zip(['train', 'val'], [train_dir, val_dir]):
        results = yolo_model.predict(source=split_dir, conf=0.25, save=True, save_txt=True, project=output_dir, name=split)
        for result in results:
            for box in result.boxes:
                patch = result.image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                inputs = vit_feature_extractor(images=patch, return_tensors="pt")
                outputs = vit_model(**inputs)
                predicted_class = torch.argmax(outputs.logits, dim=-1).item()
                print(f"Detected: {class_names[predicted_class]} at {box}")

    # Process unlabeled data
    unlabeled_dir = dataset_config.get('unlabeled', None)
    if unlabeled_dir:
        results = yolo_model.predict(source=unlabeled_dir, conf=0.05, save=True, project=output_dir, name="unlabeled")
        for result in results:
            for box in result.boxes:
                patch = result.image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                inputs = vit_feature_extractor(images=patch, return_tensors="pt")
                outputs = vit_model(**inputs)
                predicted_class = torch.argmax(outputs.logits, dim=-1).item()
                print(f"Unlabeled: Potential {class_names[predicted_class]} at {box}")

if __name__ == "__main__":
    # Arguments
    yaml_file = "code/work_cd.yaml"  # Path to YAML configuration
    yolo_model_path = "yolo11x-cls.pt"  # YOLO pre-trained model
    vit_model_name = "google/vit-base-patch16-224"  # Hugging Face ViT model
    output_dir = "results"  # Output directory

    # Process YOLO + ViT
    process_yolo_with_vit(yaml_file, yolo_model_path, vit_model_name, output_dir)
