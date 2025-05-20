import os
import cv2
import asyncio
import logging
import numpy as np
import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import pygame
import threading
import pandas as pd
import ffmpeg
import streamlit as st
import yt_dlp
import time

# Initialize Streamlit session state
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'alarm_triggered' not in st.session_state:
    st.session_state.alarm_triggered = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Logging setup for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debugging aktif.")

# Set YOLO configuration directory
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# Ensure an asyncio loop is available
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
logger.debug("Event loop asyncio sudah siap.")

# Load YOLOv8 model
try:
    model = YOLO("yolov8n.pt")
    logger.debug("Model YOLOv8 berhasil dimuat.")
except RuntimeError as e:
    logger.error(f"Kesalahan saat memuat model YOLOv8: {e}. Pastikan file model tidak rusak.")
    raise e

# Function to play the alarm
def play_alarm():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alarm system.mp3")
        pygame.mixer.music.play(-1)
    except Exception as e:
        logger.error(f"Gagal memutar alarm: {e}")

# Function to stop the alarm
def stop_alarm():
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception as e:
        logger.error(f"Gagal menghentikan alarm: {e}")

# Function to get YouTube stream URL
def get_youtube_stream_ffmpeg(url):
    try:
        ydl_opts = {
            'quiet': True,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_url = info_dict.get('url', None)
            if not video_url:
                raise ValueError("URL video streaming tidak ditemukan.")
            return video_url
    except yt_dlp.utils.DownloadError as e:
        st.error(f"⚠ Gagal mendapatkan URL streaming YouTube: {e}")
    except Exception as e:
        st.error(f"⚠ Terjadi kesalahan: {e}")
    return None

# Streamlit configuration
st.set_page_config(page_title="Smart Security System with YOLOv8", layout="wide")
st.title("\U0001F512 Smart Security System with YOLOv8")

with st.sidebar:
    st.header("\u2699\ufe0f Pengaturan Sistem")
    video_source = st.radio("*Sumber Video*", ["Webcam", "CCTV (HDMI via Capture Card)", "YouTube Live"], index=0)
    conf_threshold = st.slider("*Tingkat Kepercayaan Deteksi*", 0.0, 1.0, 0.5, 0.01)
    youtube_url = st.text_input("Masukkan URL YouTube Live", placeholder="https://www.youtube.com/...")
    
    if st.button("▶ Mulai Pemrosesan"):
        st.session_state.processing = True
        if video_source == "Webcam":
            st.session_state.cap = cv2.VideoCapture(0)
        elif video_source == "CCTV (HDMI via Capture Card)":
            cam_idx = st.sidebar.number_input("Indeks Kamera CCTV", 0, 10, 0)
            st.session_state.cap = cv2.VideoCapture(cam_idx)
        elif video_source == "YouTube Live" and youtube_url:
            stream_url = get_youtube_stream_ffmpeg(youtube_url)
            if stream_url:
                st.session_state.cap = cv2.VideoCapture(stream_url)
            else:
                st.error("⚠ Gagal mendapatkan URL streaming.")
    
    if st.button("⏹ Berhenti Pemrosesan"):
        st.session_state.processing = False
        if st.session_state.cap and st.session_state.cap.isOpened():
            st.session_state.cap.release()
        if st.session_state.alarm_triggered:
            stop_alarm()
            st.session_state.alarm_triggered = False

col1, col2 = st.columns(2)
with col1:
    st.subheader("\U0001F3A5 Live Camera Feed")
    camera_placeholder = st.empty()
with col2:
    st.subheader("\U0001F4CA Grafik Aktivitas")
    heatmap_placeholder = st.empty()

status_text = st.empty()
status_text.info("\U0001F7E2 *Sistem aktif*. Menunggu deteksi...")

# Main processing loop
if st.session_state.processing and st.session_state.cap and st.session_state.cap.isOpened():
    heatmap = np.zeros((360, 640), dtype=np.uint8)
    detection_interval = 5
    
    ret, frame = st.session_state.cap.read()
    if not ret:
        status_text.error("❌ Gagal membaca frame.")
        st.session_state.processing = False
    else:
        frame = cv2.resize(frame, (640, 360))
        heatmap = (heatmap * 0.95).astype(np.uint8)

        if st.session_state.frame_count % detection_interval == 0:
            results = model(frame)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    if confidence > conf_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        label = f"{model.names[int(class_id)]} ({confidence:.2f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if not st.session_state.alarm_triggered:
                            play_alarm()
                            st.session_state.alarm_triggered = True

        camera_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        st.session_state.frame_count += 1
        time.sleep(0.01)  # Prevent high CPU usage
else:
    status_text.warning("⚠ Silakan pilih sumber video dan pastikan kamera terhubung.")
