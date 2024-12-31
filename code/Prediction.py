from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define path to the image file
source = "Seoul_Gyeongbok-gung-3.jpg"

# Run inference on the source
results = model(source)  # list of Results objects