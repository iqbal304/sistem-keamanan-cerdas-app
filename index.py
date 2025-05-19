import cv2
import streamlit as st
import torch
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import numpy as np
from pytube import YouTube
import yt_dlp
from ultralytics import YOLO

# Load YOLO model
model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)

# Function to process frame with YOLO
def process_frame_with_yolo(frame):
    results = model(frame)
    for result in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = map(int, result[:4]) + result[4:]
        label = model.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Function to get YouTube stream URL
def get_youtube_stream_url(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True,
        'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

# Streamlit page configuration
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🔍 YOLO Object Detection App")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    video_source = st.radio("Video Source", ["Webcam", "URL Stream"], index=0)
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.01)
    youtube_url = st.text_input("YouTube Live URL", placeholder="https://www.youtube.com/...")
    stream_url = None

# Streamlit-WeRTC transformer for webcam
class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process_frame_with_yolo(img)
        return img

# Main application logic
if video_source == "Webcam":
    st.subheader("📷 Webcam Feed")
    st.info("Webcam stream will appear below once activated.")
    webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
elif video_source == "URL Stream":
    st.subheader("🔗 Stream Video from URL")
    if youtube_url:
        stream_url = get_youtube_stream_url(youtube_url)
        if stream_url:
            st.video(stream_url)
        else:
            st.error("❌ Failed to fetch the stream URL. Please check the YouTube link.")
    else:
        st.warning("⚠️ Please provide a valid YouTube Live URL in the sidebar.")

st.success("🎉 YOLO Object Detection App is ready!")
