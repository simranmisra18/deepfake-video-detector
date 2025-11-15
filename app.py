import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile

st.title("🎭 Deepfake Detection using Meso4 + GRU")
st.write("Upload a video to detect whether it’s **real or fake** using your trained model.")

# Lazy load model (so Streamlit UI loads instantly)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "deepfake_detection_best.keras",
        compile=False,
        safe_mode=False
    )
    return model

model = load_model()

# --- Helper functions ---
def extract_frame_sequences(video_path, sequence_length=10, frame_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    sequences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

        if len(frames) == sequence_length:
            sequences.append(np.array(frames))
            frames = []

    cap.release()
    return np.array(sequences)

def predict_video(model, video_path):
    sequences = extract_frame_sequences(video_path)
    if len(sequences) == 0:
        return None, None
    preds = model.predict(sequences)
    avg_pred = np.mean(preds, axis=0)
    return avg_pred, preds

# --- Streamlit Upload + Prediction ---
uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    st.video(video_path)

    st.write("⏳ Extracting frames and running prediction...")
    progress = st.progress(0)
    avg_pred, all_preds = predict_video(model, video_path)
    progress.progress(100)

    if avg_pred is None:
        st.error("No valid frames detected in the video.")
    else:
        label = "FAKE" if avg_pred[1] > avg_pred[0] else "REAL"
        confidence = float(max(avg_pred))
        st.subheader(f"🧠 Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")

st.caption("Model: Meso4 + Bi-GRU | Developed with TensorFlow & Streamlit")
