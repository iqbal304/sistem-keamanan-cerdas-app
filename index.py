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

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Inisialisasi model YOLO
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Fungsi alarm
def play_alarm():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alarm.mp3")
        pygame.mixer.music.play(-1)
    except Exception as e:
        logger.error(f"Error alarm: {e}")

def stop_alarm():
    try:
        pygame.mixer.music.stop()
    except:
        pass

# Fungsi YouTube stream
def get_youtube_stream(url):
    try:
        ydl_opts = {'format': 'best'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"Error YouTube: {e}")
        return None

# Antarmuka Streamlit
st.set_page_config(page_title="Smart Security System", layout="wide")
st.title("Sistem Keamanan Cerdas")

# Sidebar
with st.sidebar:
    st.header("Pengaturan")
    video_source = st.radio("Sumber Video", 
                           ["Webcam", "CCTV", "YouTube"])
    
    conf_threshold = st.slider("Threshold Deteksi", 0.1, 0.9, 0.5)
    
    youtube_url = ""
    if video_source == "YouTube":
        youtube_url = st.text_input("URL YouTube")

# Area tampilan
col1, col2 = st.columns(2)
frame_placeholder = col1.empty()
heatmap_placeholder = col2.empty()
status = st.empty()

# Variabel kontrol
is_running = False
stop_stream = False

def video_loop():
    global is_running, stop_stream
    
    cap = None
    try:
        if video_source == "Webcam":
            cap = cv2.VideoCapture(0)
        elif video_source == "CCTV":
            cap = cv2.VideoCapture(1)
        elif video_source == "YouTube" and youtube_url:
            stream_url = get_youtube_stream(youtube_url)
            if stream_url:
                cap = cv2.VideoCapture(stream_url)
        
        if not cap or not cap.isOpened():
            status.error("Gagal membuka video")
            return
            
        is_running = True
        stop_stream = False
        heatmap = np.zeros((480, 640), dtype=np.float32)
        
        while is_running and not stop_stream:
            ret, frame = cap.read()
            if not ret:
                status.warning("Gagal membaca frame")
                continue
                
            # Deteksi objek
            results = model(frame, conf=conf_threshold)
            
            # Gambar bounding box
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    
                    # Update heatmap
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    heatmap[cy-10:cy+10, cx-10:cx+10] += 1
                    
                    # Trigger alarm
                    if box.cls in [0, 15, 16]:  # Orang, kucing, anjing
                        play_alarm()
            
            # Normalisasi heatmap
            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Tampilkan frame
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            heatmap_placeholder.image(heatmap_color)
            
            # Reduksi heatmap
            heatmap *= 0.95
            
    except Exception as e:
        status.error(f"Error: {e}")
    finally:
        if cap:
            cap.release()
        stop_alarm()
        is_running = False

# Tombol kontrol
if not is_running:
    if st.sidebar.button("Mulai Deteksi"):
        threading.Thread(target=video_loop, daemon=True).start()
else:
    if st.sidebar.button("Berhenti"):
        stop_stream = True

st.sidebar.info("""
**Panduan:**
1. Pilih sumber video
2. Atur threshold deteksi
3. Klik Mulai Deteksi
4. Alarm akan berbunyi saat terdeteksi objek mencurigakan
""")
