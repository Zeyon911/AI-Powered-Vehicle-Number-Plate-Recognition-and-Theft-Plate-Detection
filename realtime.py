import streamlit as st
import cv2
import numpy as np
from yolov8_detector import detect_plates_yolo
from ocr import PlateOCR

def realtime_streamlit():
    st.subheader("🎥 Real-time License Plate Detection (Indian Standard)")

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    ocr = PlateOCR()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Unable to access webcam.")
            break

        plates = detect_plates_yolo(frame)

        for plate_img, (x, y, w, h) in plates:
            text_list = ocr.recognize_text(plate_img)
            if text_list:
                detected_text = text_list[0]
                if "Not a valid" in detected_text:
                    cv2.putText(frame, detected_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # red
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                else:
                    cv2.putText(frame, detected_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # green
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert BGR → RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()