from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 license plate detection model (you can train your own later)
model = YOLO('yolov8m.pt')  # You can replace with a custom-trained model

def detect_plates_yolo(image):
    results = model(image)
    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = image[y1:y2, x1:x2]
            detections.append((cropped, (x1, y1, x2 - x1, y2 - y1)))
    return detections
