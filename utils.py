import cv2
import pytesseract
import easyocr
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageEnhance

# Set Tesseract path if not in PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLOv8 model
model = YOLO("yolov8m.pt")  # or "license_plate_detector.pt" if preferred

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Initialize only once for better performance

def detect_plate_yolo(image):
    results = model(image)[0]
    plates = []
    boxes = []

    for box in results.boxes:
        # Filter for license plate class (assuming class 0 is license plate)
        if int(box.cls[0]) == 0:  # Adjust class index if needed
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Add padding to the bounding box
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            crop = image[y1:y2, x1:x2]
            plates.append(crop)
            boxes.append((x1, y1, x2, y2))

    return plates, boxes

def preprocess_plate_image(plate_img):
    """Enhanced image preprocessing for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def recognize_text_tesseract(plate_img):
    """Recognize text using Tesseract with enhanced preprocessing"""
    processed_img = preprocess_plate_image(plate_img)
    config = "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(processed_img, config=config)
    return text.strip()

def recognize_text_easyocr(plate_img):
    """Recognize text using EasyOCR with enhanced preprocessing"""
    processed_img = preprocess_plate_image(plate_img)
    
    # Convert back to color for EasyOCR (it expects color images)
    color_processed = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    
    # Use EasyOCR to detect text
    results = reader.readtext(color_processed, detail=0, paragraph=False)
    
    # Filter and combine results
    text = " ".join([res for res in results if len(res) > 2])
    
    return text.strip()

def recognize_text_combined(plate_img, confidence_threshold=0.5):
    """
    Combined approach using both EasyOCR and Tesseract
    Returns the result with the highest confidence
    """
    # Preprocess the image
    processed_img = preprocess_plate_image(plate_img)
    
    # EasyOCR recognition
    color_processed = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    easy_results = reader.readtext(color_processed, detail=1, paragraph=False)
    
    # Filter EasyOCR results by confidence
    easy_text = ""
    if easy_results:
        # Get the result with highest confidence
        best_result = max(easy_results, key=lambda x: x[2])
        if best_result[2] >= confidence_threshold:
            easy_text = best_result[1]
    
    # Tesseract recognition
    config = "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    tesseract_text = pytesseract.image_to_string(processed_img, config=config).strip()
    
    # Return the non-empty result with preference to EasyOCR if confidence is high
    if easy_text and len(easy_text) >= 4:  # Minimum length for a license plate
        return easy_text
    elif tesseract_text:
        return tesseract_text
    else:
        return easy_text if easy_text else "No text detected"

# Default to use combined approach
recognize_text = recognize_text_combined