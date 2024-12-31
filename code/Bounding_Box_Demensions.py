from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')  # Load pretrain or fine-tune model

# Process the image
source = cv2.imread('Seoul_Gyeongbok-gung-3.jpg')
results = model(source)

# Extract results
annotator = Annotator(source, example=model.names)

for box in results[0].boxes.xyxy.cpu():
    width, height, area = annotator.get_bbox_dimension(box)
    print("Bounding Box Width {}, Height {}, Area {}".format(
        width.item(), height.item(), area.item()))