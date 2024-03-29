from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.predict(
    source='input/01-screenshot.png',
    save=True,
    project='runs/detect')

result2 = model.predict(
    source='input/01-tennis_video.mp4',
    save=True,
    project='runs/detect')
