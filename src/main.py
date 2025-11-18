import cv2
from yolov8_detector import detect_plates_yolo
from ocr import PlateOCR

def process_image(image_path):
    image = cv2.imread(image_path)
    ocr = PlateOCR()

    plates = detect_plates_yolo(image)

    for plate_img, (x, y, w, h) in plates:
        text_list = ocr.recognize_text(plate_img)
        if text_list:
            detected_text = text_list[0]
            if "Not a valid" in detected_text:
                print("⚠️ Rejected:", detected_text)
                cv2.putText(image, detected_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                print("✅ Detected Number:", detected_text)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, detected_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_image("test.jpg")
