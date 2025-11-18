# detector.py (modified)
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple

class PlateDetector:
    def __init__(self, model_path: str = "license_plate_detector.pt"):
        self.model = YOLO(model_path)
        self.conf_threshold = 0.5
        self.min_plate_ratio = 1.2
        self.min_plate_area = 2000

    def detect(
        self,
        frame: np.ndarray,
        padding_ratio: float = 0.05,
        debug: bool = False,
        target_size: Tuple[int, int] = (300, 100)   # <-- normalize plate size
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        results = self.model(frame, verbose=False)[0]
        plates = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            area = w * h
            aspect_ratio = w / h if h != 0 else 0

            if aspect_ratio < self.min_plate_ratio or area < self.min_plate_area:
                continue

            # Apply padding safely
            pad_x = int(w * padding_ratio)
            pad_y = int(h * padding_ratio)
            x1 = max(x1 - pad_x, 0)
            y1 = max(y1 - pad_y, 0)
            x2 = min(x2 + pad_x, frame.shape[1])
            y2 = min(y2 + pad_y, frame.shape[0])

            cropped = frame[y1:y2, x1:x2]

            if cropped.size > 0:
                # 🔑 Resize to fixed size for OCR stability
                resized_plate = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LINEAR)

                plates.append((resized_plate, (x1, y1, x2 - x1, y2 - y1), conf))

                if debug:
                    print(f"[DEBUG] Plate detected at ({x1},{y1},{x2},{y2}) with conf {conf:.2f}")

        return plates