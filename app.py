# Streamlit UI for the AI Squat Trainer (Not business logic)
# Uses streamlit-webrtc for fast ~30fps display

import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from pose import PoseDetector

# ── WebRTC STUN config (needed for non-localhost deployments) ─────────────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

st.title("Fitness Assistant")

source = st.radio("Input Source", ["Webcam", "Video File"], horizontal=True)

# Controls how much landmark jitter is smoothed out.
# Higher value => smoother landmarks (but slightly less responsive).
smoothness = st.slider(
    "Landmark smoothness",
    min_value=0.1,
    max_value=0.9,
    value=0.4,
    step=0.05,
)

# MediaPipe pose model complexity (trade-off: accuracy vs speed).
model_complexity = st.slider(
    "Model complexity",
    min_value=1,
    max_value=2,
    value=1,
    step=1,
)

# ── Video processor: called per frame by webrtc_streamer ─────────────────────
class SquatProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = PoseDetector(alpha=smoothness, model_complexity=model_complexity)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")          # av frame → numpy
        img = cv2.resize(img, (640, 480))
        img = self.detector.process_frame(img)           # your pose logic
        return av.VideoFrame.from_ndarray(img, format="bgr24")  # numpy → av frame


# ── Webcam mode via webrtc_streamer ──────────────────────────────────────────
if source == "Webcam":
    st.info("Click **START** below to begin.")
    webrtc_streamer(
        key=f"squat-trainer-{smoothness}-{model_complexity}",
        video_processor_factory=SquatProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,      # ⚡ runs recv() in a thread — non-blocking
    )

# ── Video file mode (still uses cv2 + st.image, webrtc needs live stream) ────
else:
    input_path = st.text_input(
        "Video file path",
        value=r"D:\AIDS\cv\fitness_assist\inputs\VID_20260316_132404.mp4"
    )
    run = st.checkbox("▶ Play")
    
    frame_placeholder = st.empty()

    if run and input_path:
        detector = PoseDetector(alpha=smoothness, model_complexity=model_complexity)
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("❌ Could not open file. Check the path.")
            st.stop()

        while run:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)    # loop video
                continue

            frame = cv2.resize(frame, (640, 480))
            frame = detector.process_frame(frame)
            frame_placeholder.image(frame, channels="BGR")

        cap.release()