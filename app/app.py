import streamlit as st
import cv2
import os
from datetime import datetime

# UI Layout: 1 trang duy nhất
st.set_page_config(page_title="Face Detector", layout="wide")

# Sidebar hoặc Header để hiển thị Config hiện tại
st.sidebar.title("Configuration")
st.sidebar.info(f"RTSP Source: {os.getenv('RTSP_URL')}")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Live RTSP Stream")
    frame_placeholder = st.empty()

with col_right:
    st.subheader("Face Events")
    if "event_logs" not in st.session_state:
        st.session_state.event_logs = []
    log_placeholder = st.empty()

# Model đơn giản (Haar Cascade) - cực nhẹ cho Macbook M3
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Xử lý video
cap = cv2.VideoCapture(os.getenv("RTSP_URL"))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot connect to RTSP stream")
        break

    # Giảm size để xử lý nhanh hơn
    frame_small = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Thêm event vào log
        event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.event_logs.insert(0, f"👤 Person detected - {event_time}")
        
        # Logic gửi Telegram có thể đặt ở đây (nên kèm theo time-debounce)

    # Hiển thị
    frame_placeholder.image(frame_small, channels="BGR")
    
    # Hiển thị tối đa 20 event gần nhất
    log_placeholder.markdown("\n".join(st.session_state.event_logs[:20]))

cap.release()