import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from PIL import Image

# Load your model only once for efficiency
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("deepfake_detection_best.keras")
    return model

model = load_model()

# Streamlit UI
st.title("🎭 Deepfake Video Detection")
st.write("Upload a video file to analyze whether it’s **real or fake** using a deep learning model.")

# File uploader
uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov", "mkv"])

# Preprocessing function
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))  # adjust if your model uses different input size
    frame = frame.astype("float32") / 255.0
    return np.expand_dims(frame, axis=0)

if uploaded_video is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Display uploaded video
    st.video(video_path)

    # Process video
    st.write("🔍 Analyzing video frames...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sampled_predictions = []
    frame_interval = max(total_frames // 20, 1)  # sample ~20 frames evenly

    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pred = model.predict(preprocess_frame(frame_rgb))[0][0]
        sampled_predictions.append(pred)

    cap.release()

    if len(sampled_predictions) > 0:
        avg_pred = np.mean(sampled_predictions)
        label = "FAKE" if avg_pred > 0.5 else "REAL"
        confidence = avg_pred if label == "FAKE" else 1 - avg_pred

        st.subheader(f"🧠 Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")
    else:
        st.warning("Could not extract frames from the video. Try another file.")

st.markdown("---")
st.caption("Model: `deepfake_detection_best.keras` | Built with Streamlit & TensorFlow")
