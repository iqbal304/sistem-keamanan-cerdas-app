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

# Konfigurasi logging untuk debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debugging aktif.")

# Set direktori konfigurasi YOLO
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# Memastikan loop asyncio tersedia
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
logger.debug("Event loop asyncio sudah siap.")

# Memuat model YOLOv8
try:
    model = YOLO("yolov8n.pt")
    logger.debug("Model YOLOv8 berhasil dimuat.")
except RuntimeError as e:
    logger.error(f"Kesalahan saat memuat model YOLOv8: {e}. Pastikan file model tidak rusak.")
    raise e

# Fungsi untuk memutar alarm
def play_alarm():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alarm.mp3")
        pygame.mixer.music.play(-1)
    except Exception as e:
        logger.error(f"Gagal memutar alarm: {e}")

# Fungsi untuk menghentikan alarm
def stop_alarm():
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception as e:
        logger.error(f"Gagal menghentikan alarm: {e}")

# Fungsi untuk mendapatkan URL stream YouTube yang diperbaiki
def get_youtube_stream_url(url):
    try:
        ydl_opts = {
            'quiet': True,
            'format': 'best',  # Format terbaik yang tersedia
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            
            # Mencari URL stream dengan kualitas terbaik
            if 'formats' in info_dict:
                for f in info_dict['formats']:
                    if f.get('protocol', '').startswith('http') and f.get('ext') == 'mp4':
                        return f['url']
            
            # Jika tidak ditemukan format mp4, coba format lain
            return info_dict.get('url', None)
            
    except yt_dlp.utils.DownloadError as e:
        st.error(f"⚠ Gagal mendapatkan URL streaming YouTube: {e}")
    except Exception as e:
        st.error(f"⚠ Terjadi kesalahan: {e}")
    return None

# Konfigurasi Streamlit
st.set_page_config(page_title="Smart Security System with YOLOv8", layout="wide")
st.title("\U0001F512 Sistem Keamanan Cerdas dengan YOLOv8")

with st.sidebar:
    st.header("\u2699\ufe0f Pengaturan Sistem")
    video_source = st.radio("*Sumber Video*", ["Webcam", "CCTV (HDMI via Capture Card)", "YouTube Live"], index=0)
    conf_threshold = st.slider("*Tingkat Kepercayaan Deteksi*", 0.0, 1.0, 0.5, 0.01)
    
    # Hanya tampilkan input URL YouTube jika sumber video YouTube Live dipilih
    youtube_url = ""
    if video_source == "YouTube Live":
        youtube_url = st.text_input("Masukkan URL YouTube Live", placeholder="https://www.youtube.com/watch?v=...")

col1, col2 = st.columns(2)
with col1:
    st.subheader("\U0001F3A5 Live Camera Feed")
    camera_placeholder = st.empty()
with col2:
    st.subheader("\U0001F4CA Grafik Aktivitas")
    heatmap_placeholder = st.empty()

status_text = st.empty()
status_text.info("\U0001F7E2 *Sistem aktif*. Menunggu deteksi...")

# Variabel global untuk kontrol streaming
streaming_active = False
stop_button_pressed = False

def start_streaming():
    global streaming_active, stop_button_pressed
    
    cap = None
    try:
        if video_source == "Webcam":
            cap = cv2.VideoCapture(0)
        elif video_source == "CCTV (HDMI via Capture Card)":
            cam_idx = st.session_state.get('cam_idx', 0)
            cap = cv2.VideoCapture(cam_idx)
        elif video_source == "YouTube Live" and youtube_url:
            stream_url = get_youtube_stream_url(youtube_url)
            if stream_url:
                cap = cv2.VideoCapture(stream_url)
            else:
                status_text.error("⚠ Gagal mendapatkan URL streaming YouTube.")
                return
        
        if cap is None or not cap.isOpened():
            status_text.error("⚠ Gagal membuka sumber video.")
            return
        
        streaming_active = True
        stop_button_pressed = False
        
        heatmap = np.zeros((360, 640), dtype=np.uint8)
        activity_logs = defaultdict(list)
        alarm_triggered = False
        frame_count = 0
        detection_interval = 5
        
        while streaming_active and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                status_text.warning("⚠ Frame tidak terbaca. Mencoba lagi...")
                continue
            
            frame = cv2.resize(frame, (640, 360))
            heatmap = (heatmap * 0.95).astype(np.uint8)
            
            if frame_count % detection_interval == 0:
                # Logika Deteksi
                results = model(frame, conf=conf_threshold)
                
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, confidence, class_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        label = f"{model.names[int(class_id)]} ({confidence:.2f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Update heatmap
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        heatmap[center_y-5:center_y+5, center_x-5:center_x+5] += 25
                        
                        # Trigger alarm jika aktivitas mencurigakan terdeteksi
                        if model.names[int(class_id)] in ['person', 'dog', 'cat'] and not alarm_triggered:
                            play_alarm()
                            alarm_triggered = True
                            status_text.warning("\U0001F6A8 Aktivitas mencurigakan terdeteksi!")
            
            # Normalisasi heatmap untuk visualisasi
            heatmap_viz = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_viz = cv2.applyColorMap(heatmap_viz, cv2.COLORMAP_JET)
            
            # Menampilkan frame video dan heatmap
            camera_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            heatmap_placeholder.image(cv2.cvtColor(heatmap_viz, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            frame_count += 1
            
            # Memberi waktu untuk thread lain
            cv2.waitKey(1)
            
    except Exception as e:
        logger.error(f"Error dalam streaming: {e}")
        status_text.error(f"⚠ Error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        stop_alarm()
        streaming_active = False

# Tombol kontrol streaming
if not streaming_active:
    if st.sidebar.button("\U0001F3A5 Mulai Streaming"):
        stop_button_pressed = False
        threading.Thread(target=start_streaming, daemon=True).start()
else:
    if st.sidebar.button("\U00023F9 Hentikan Streaming"):
        stop_button_pressed = True
        streaming_active = False
        status_text.info("\U0001F7E2 Streaming dihentikan.")

# Catatan penting
st.sidebar.markdown("---")
st.sidebar.info("""
**Catatan Penggunaan:**
1. Untuk YouTube Live, pastikan URL valid dan tidak memiliki restriksi
2. Tunggu beberapa detik setelah menekan tombol mulai
3. Alarm akan berbunyi saat terdeteksi orang/hewan
4. Pastikan file alarm.mp3 ada di direktori yang sama
""")
