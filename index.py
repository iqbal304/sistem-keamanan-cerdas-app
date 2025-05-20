import os
import cv2
import logging
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import pygame
import streamlit as st
import yt_dlp
import time
import pandas as pd
from datetime import datetime

# ======================
# INITIAL SETUP
# ======================
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set YOLO config directory
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# Initialize pygame for audio
pygame.mixer.init()

# ======================
# FUNCTION DEFINITIONS
# ======================
def play_alarm():
    """Play alarm sound in loop"""
    try:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load("alarm.mp3")  # Make sure this file exists
            pygame.mixer.music.play(-1)  # -1 for infinite loop
    except Exception as e:
        logger.error(f"Alarm error: {e}")
        st.error(f"Alarm error: {e}")

def stop_alarm():
    """Stop alarm sound"""
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception as e:
        logger.error(f"Stop alarm error: {e}")

def get_youtube_stream(url):
    """Extract YouTube stream URL using yt-dlp"""
    try:
        ydl_opts = {
            'quiet': True,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"YouTube error: {e}")
        return None

def initialize_camera(source, youtube_url=None, cam_index=0):
    """Initialize video capture based on source"""
    try:
        if source == "Webcam":
            cap = cv2.VideoCapture(cam_index)
        elif source == "CCTV (HDMI via Capture Card)":
            cap = cv2.VideoCapture(cam_index)
        elif source == "YouTube Live" and youtube_url:
            stream_url = get_youtube_stream(youtube_url)
            if stream_url:
                cap = cv2.VideoCapture(stream_url)
            else:
                return None
        else:
            return None
        
        # Test if camera opened successfully
        if cap is not None and cap.isOpened():
            return cap
        return None
    except Exception as e:
        st.error(f"Camera initialization error: {e}")
        return None

def save_detection_log(class_name, confidence, position):
    """Save detection logs to CSV"""
    try:
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "object": class_name,
            "confidence": float(confidence),
            "x_position": position[0],
            "y_position": position[1]
        }
        
        # Create or append to log file
        if not os.path.exists("detection_logs.csv"):
            pd.DataFrame([log_entry]).to_csv("detection_logs.csv", index=False)
        else:
            pd.DataFrame([log_entry]).to_csv("detection_logs.csv", 
                                          mode='a', 
                                          header=False, 
                                          index=False)
    except Exception as e:
        logger.error(f"Error saving log: {e}")

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="Smart Security System", layout="wide")
st.title("🛡️ Smart Security System with YOLOv8")

# Session state initialization
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'alarm_triggered' not in st.session_state:
    st.session_state.alarm_triggered = False

# Sidebar controls
with st.sidebar:
    st.header("⚙️ System Settings")
    video_source = st.radio("Video Source", 
                          ["Webcam", "CCTV (HDMI via Capture Card)", "YouTube Live"], 
                          index=0)
    
    # Conditional inputs
    if video_source == "CCTV (HDMI via Capture Card)":
        cam_idx = st.number_input("Camera Index", 0, 10, 0)
    elif video_source == "YouTube Live":
        youtube_url = st.text_input("YouTube URL", placeholder="https://youtube.com/live/...")
    
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.01)
    
    # Alarm settings
    st.subheader("🔔 Alarm Settings")
    alarm_classes = st.multiselect(
        "Classes to trigger alarm",
        options=["person", "gun", "knife", "car", "dog"],
        default=["person", "gun", "knife"]
    )
    
    # Control buttons
    if st.button("🚀 Start Detection"):
        st.session_state.detection_active = True
    
    if st.button("⏹️ Stop Detection"):
        st.session_state.detection_active = False
        stop_alarm()
        st.session_state.alarm_triggered = False

# Main columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()
    
with col2:
    st.subheader("📊 Activity Monitoring")
    
    # Real-time stats
    stats_placeholder = st.empty()
    heatmap_placeholder = st.empty()

# Status text
status_text = st.empty()

# ======================
# MAIN PROCESSING
# ======================
if st.session_state.detection_active:
    # Initialize video capture
    cap = initialize_camera(
        source=video_source,
        youtube_url=youtube_url if video_source == "YouTube Live" else None,
        cam_index=cam_idx if video_source == "CCTV (HDMI via Capture Card)" else 0
    )
    
    if cap is None:
        status_text.error("❌ Failed to initialize video source!")
        st.stop()
    
    # Load YOLO model
    try:
        model = YOLO("yolov8n.pt")
    except Exception as e:
        status_text.error(f"Model loading failed: {e}")
        st.stop()
    
    # Initialize variables
    heatmap = np.zeros((360, 640), dtype=np.uint8)
    detection_logs = []
    frame_count = 0
    detection_interval = 5  # Process every 5 frames
    fps = 15
    
    status_text.success("✅ System active! Detecting objects...")
    
    # Main processing loop
    while cap.isOpened() and st.session_state.detection_active:
        ret, frame = cap.read()
        if not ret:
            status_text.warning("⚠ Video stream ended")
            break
        
        # Frame processing
        frame = cv2.resize(frame, (640, 360))
        heatmap = (heatmap * 0.95).astype(np.uint8)  # Decay heatmap
        
        # Object detection
        if frame_count % detection_interval == 0:
            results = model(frame, verbose=False)
            
            current_detections = []
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    if conf > conf_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        class_name = model.names[int(cls_id)]
                        center = ((x1+x2)//2, (y1+y2)//2)
                        
                        # Store detection info
                        detection_info = {
                            "class": class_name,
                            "confidence": float(conf),
                            "position": center,
                            "timestamp": datetime.now()
                        }
                        current_detections.append(detection_info)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        
                        # Update heatmap
                        cv2.circle(heatmap, center, 10, 255, -1)
                        
                        # Save to log
                        save_detection_log(class_name, conf, center)
                        
                        # Trigger alarm for selected classes
                        if class_name in alarm_classes:
                            if not st.session_state.alarm_triggered:
                                play_alarm()
                                st.session_state.alarm_triggered = True
        
        # Display results
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                              channels="RGB", use_column_width=True)
        
        # Display stats
        if current_detections:
            stats_df = pd.DataFrame(current_detections)
            stats_placeholder.dataframe(
                stats_df[["class", "confidence", "timestamp"]],
                height=200,
                use_container_width=True
            )
        else:
            stats_placeholder.write("No objects detected")
        
        # Display heatmap
        heatmap_placeholder.image(
            cv2.applyColorMap(heatmap, cv2.COLORMAP_JET),
            use_column_width=True,
            caption="Activity Heatmap (Red = High Activity)"
        )
        
        # Control frame rate
        frame_count += 1
        time.sleep(1.0/fps)  # Simple FPS control
    
    # Cleanup
    cap.release()
    stop_alarm()
    status_text.warning("🛑 Detection stopped")

# Show instructions when not running
else:
    status_text.info("System ready. Click 'Start Detection' to begin.")
    
    # Display log history if available
    if os.path.exists("detection_logs.csv"):
        st.subheader("📜 Detection History")
        try:
            log_df = pd.read_csv("detection_logs.csv")
            st.dataframe(log_df, use_container_width=True)
            
            # Show basic statistics
            st.write("### Detection Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Detections", len(log_df))
            with col2:
                st.metric("Most Common Object", 
                          log_df['object'].mode()[0] if not log_df.empty else "N/A")
            with col3:
                st.metric("Average Confidence", 
                          f"{log_df['confidence'].mean():.2f}" if not log_df.empty else "N/A")
        except Exception as e:
            st.error(f"Error loading logs: {e}")
