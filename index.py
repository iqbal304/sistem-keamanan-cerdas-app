import cv2
from pytube import YouTube
import streamlit as st
import numpy as np
import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import pygame
import threading
import pandas as pd
import ffmpeg
import os

# Load model YOLOv8
model = YOLO("yolov8n.pt")

# Function to play the alarm
def play_alarm():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alarm system.mp3")
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"Gagal memutar alarm: {e}")

# Function to stop the alarm
def stop_alarm():
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception as e:
        print(f"Gagal menghentikan alarm: {e}")

# Function to get YouTube stream URL
def get_youtube_stream_ffmpeg(url):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        return stream.url
    except Exception as e:
        st.error(f"⚠ Gagal mendapatkan URL streaming YouTube: {e}")
        return None

# Function to read video using FFmpeg
def read_video_ffmpeg(source):
    try:
        process = (
            ffmpeg.input(source)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
        )
        return process
    except Exception as e:
        st.error(f"⚠ Gagal membaca video dengan FFmpeg: {e}")
        return None

# Streamlit configuration
st.set_page_config(page_title="Smart Security System with FFmpeg", layout="wide")
st.title("\U0001F512 Smart Security System with FFmpeg")

with st.sidebar:
    st.header("\u2699\ufe0f Pengaturan Sistem")
    video_source = st.radio("*Sumber Video*", ["Webcam", "CCTV (HDMI via Capture Card)", "YouTube Live"], index=0)
    conf_threshold = st.slider("*Tingkat Kepercayaan Deteksi*", 0.0, 1.0, 0.5, 0.01)
    max_reps = st.number_input("*Batas Gerakan untuk Alarm*", 1, 50, 5)
    youtube_url = st.text_input("Masukkan URL YouTube Live", placeholder="https://www.youtube.com/...")

col1, col2 = st.columns(2)
with col1:
    st.subheader("\U0001F3A5 Live Camera Feed")
    camera_placeholder = st.empty()
with col2:
    st.subheader("\U0001F4CA Grafik Aktivitas")
    heatmap_placeholder = st.empty()

status_text = st.empty()
status_text.info("\U0001F7E2 *Sistem aktif*. Menunggu deteksi...")

cap = None
if video_source == "Webcam":
    if st.sidebar.button("\U0001F3A5 Mulai Streaming Webcam"):
        cap = cv2.VideoCapture(0)

elif video_source == "CCTV (HDMI via Capture Card)":
    cam_idx = st.sidebar.number_input("Indeks Kamera CCTV", 0, 10, 0)
    if st.sidebar.button("\U0001F517 Sambungkan ke CCTV"):
        cap = cv2.VideoCapture(cam_idx)

elif video_source == "YouTube Live":
    if st.sidebar.button("\U0001F3A5 Mulai Streaming YouTube"):
        if youtube_url:
            stream_url = get_youtube_stream_ffmpeg(youtube_url)
            if stream_url:
                cap = read_video_ffmpeg(stream_url)

if cap:
    heatmap = np.zeros((360, 640), dtype=np.uint8)
    activity_logs = defaultdict(list)
    alarm_triggered = [False]
    heatmap_history = deque(maxlen=100)
    frame_count = 0
    detection_interval = 5

    while True:
        if video_source == "YouTube Live":
            in_bytes = cap.stdout.read(640 * 360 * 3)
            if not in_bytes:
                break
            frame = np.frombuffer(in_bytes, np.uint8).reshape([360, 640, 3])
        else:
            ret, frame = cap.read()
            if not ret:
                status_text.error("❌ Gagal membaca frame.")
                break

        frame = cv2.resize(frame, (640, 360))
        heatmap = (heatmap * 0.95).astype(np.uint8)

        if frame_count % detection_interval == 0:
            frame = detect_suspicious_activity(
                frame, model, conf_threshold, heatmap, [],  # AOI kosong sementara
                activity_logs, max_reps, alarm_triggered, heatmap_history
            )

        camera_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        df_heat = pd.DataFrame(heatmap_history)
        if not df_heat.empty:
            heatmap_placeholder.line_chart(df_heat.set_index("time"))

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_source == "YouTube Live":
        cap.terminate()
    else:
        cap.release()

else:
    status_text.warning("⚠ Silakan pilih sumber video dan pastikan kamera terhubung.")
