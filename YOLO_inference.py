# Import libraries
from ultralytics import YOLO

# Load model
model = YOLO('models/yolov5_last.pt')

# result = model.predict(
#     source='input/01-screenshot.png',
#     save=True,
#     project='runs/detect')

result2 = model.predict(
    source='input/01-tennis_video.mp4',
    save=True,
    conf=0.2,
    project='runs/detect')