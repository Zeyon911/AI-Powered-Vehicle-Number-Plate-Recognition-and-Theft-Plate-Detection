import cv2
import easyocr
import re

class PlateOCR:
    def __init__(self, languages=['en'], gpu=True):
        # Initialize EasyOCR reader with GPU/CPU option
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.confidence_threshold = 0.4

    def preprocess_for_ocr(self, plate_img):
        """Convert plate image to grayscale for OCR."""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        return gray

    def debug_preprocess(self, plate_img):
        """
        Debug preprocessing: show how OCR 'sees' the plate.
        Produces a thresholded grayscale version.
        """
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def is_indian_plate(self, text: str) -> bool:
        """
        Verify if text matches Indian number plate format.
        Format: XX00XX0000 (State, RTO code, Series, Number)
        Examples: KA01AB1234, MH20EE0071
        """
        pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$'
        return re.match(pattern, text.replace(" ", "").upper()) is not None

    def recognize_text(self, image):
        """Run OCR on the given plate image and validate against Indian format."""
        processed = self.preprocess_for_ocr(image)
        h, w = processed.shape
        resized = cv2.resize(processed, (w * 2, h * 2))  # Upscale for better OCR

        results = self.reader.readtext(resized, detail=1, paragraph=False)

        valid_texts = []
        for box in results:
            text = box[1].upper().replace(" ", "")
            conf = float(box[2])
            if conf >= self.confidence_threshold:
                if 4 <= len(text) <= 15:
                    if self.is_indian_plate(text):
                        valid_texts.append(text)
                    else:
                        valid_texts.append("❌ Not' a 'valid Number 'Plate")
        return valid_texts
