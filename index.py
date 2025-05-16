import streamlit as st
import numpy as np
import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import av
import cv2  # Pastikan opencv-python-headless terinstal

# Solusi alternatif jika cv2 masih bermasalah
try:
    import cv2
except ImportError:
    st.error("OpenCV tidak terinstal. Menggunakan fallback...")
    # Tambahkan fallback behavior di sini jika diperlukan

# Konfigurasi untuk WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Inisialisasi model YOLOv8 dengan cache
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")
        st.success("Model YOLOv8 berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

# Kelas Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf_threshold = 0.5
        self.aois = []
        self.max_reps = 5
        self.activity_logs = defaultdict(list)
        self.alarm_triggered = False
        self.heatmap = None
        self.heatmap_history = deque(maxlen=100)
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            if self.heatmap is None:
                h, w = img.shape[:2]
                self.heatmap = np.zeros((h, w), dtype=np.uint8)
            
            if model is not None:
                results = model(img)
                
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        label = result.names[int(box.cls[0])]

                        if label == "person" and conf > self.conf_threshold:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            self.heatmap[y1:y2, x1:x2] += 1

                            for idx, (ax1, ay1, ax2, ay2) in enumerate(self.aois):
                                if x1 >= ax1 and y1 >= ay1 and x2 <= ax2 and y2 <= ay2:
                                    self.activity_logs[idx].append(datetime.datetime.now())

                for idx in range(len(self.aois)):
                    cutoff = datetime.datetime.now() - datetime.timedelta(seconds=10)
                    self.activity_logs[idx] = [t for t in self.activity_logs[idx] if t > cutoff]
                    
                    if len(self.activity_logs[idx]) > self.max_reps:
                        if not self.alarm_triggered:
                            self.alarm_triggered = True
                            st.session_state['alarm'] = True
                            st.toast(f"ALARM: Gerakan mencurigakan di Zona {idx+1}!", icon="⚠️")

                self.heatmap = (self.heatmap * 0.95).astype(np.uint8)
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                self.heatmap_history.append({"time": timestamp, "activity": int(np.max(self.heatmap))})
                st.session_state['heatmap_history'] = list(self.heatmap_history)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            st.error(f"Error dalam pemrosesan frame: {e}")
            return frame

# Konfigurasi Streamlit
st.set_page_config(page_title="Smart Security System", layout="wide")
st.title("🔒 Smart Security System - Streamlit Cloud")

# Inisialisasi session state
if 'heatmap_history' not in st.session_state:
    st.session_state['heatmap_history'] = []
if 'alarm' not in st.session_state:
    st.session_state['alarm'] = False

with st.sidebar:
    st.header("⚙️ Pengaturan Sistem")
    conf_threshold = st.slider("Tingkat Kepercayaan Deteksi", 0.0, 1.0, 0.5, 0.01)
    max_reps = st.number_input("Batas Gerakan untuk Alarm", 1, 50, 5)
    
    st.subheader("📍 Zona Pengawasan (AOI)")
    num_aois = st.number_input("Jumlah Zona", 0, 5, 1)
    aois = []
    for i in range(num_aois):
        with st.expander(f"Pengaturan Zona {i+1}"):
            x1 = st.slider(f"X1 (Kiri)", 0, 1000, 200, key=f"x1_{i}")
            y1 = st.slider(f"Y1 (Atas)", 0, 1000, 200, key=f"y1_{i}")
            x2 = st.slider(f"X2 (Kanan)", 0, 1000, 800, key=f"x2_{i}")
            y2 = st.slider(f"Y2 (Bawah)", 0, 1000, 600, key=f"y2_{i}")
            aois.append((x1, y1, x2, y2))

col1, col2 = st.columns(2)
with col1:
    st.subheader("🎥 Live Camera Feed")
    ctx = webrtc_streamer(
        key="smart-security",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if ctx.video_processor:
        ctx.video_processor.conf_threshold = conf_threshold
        ctx.video_processor.aois = aois
        ctx.video_processor.max_reps = max_reps

with col2:
    st.subheader("📈 Grafik Aktivitas")
    if st.session_state['heatmap_history']:
        df_heat = pd.DataFrame(st.session_state['heatmap_history'])
        st.line_chart(df_heat.set_index("time"))
    else:
        st.info("Menunggu data aktivitas...")
    
    if st.session_state['alarm']:
        st.warning("🚨 ALARM AKTIF: Gerakan mencurigakan terdeteksi!")
        if st.button("Matikan Alarm"):
            st.session_state['alarm'] = False
            st.toast("Alarm dimatikan", icon="✅")

st.info("ℹ️ Sistem ini menggunakan WebRTC untuk streaming video langsung dari perangkat Anda. Pastikan Anda mengizinkan akses kamera.")
