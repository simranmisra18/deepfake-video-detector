import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile

st.title("🎭 Deepfake Detection (Meso4 + Bi-GRU)")
st.write("Upload a video to detect whether it’s **REAL or FAKE** using your trained model.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("deepfake_detection_best.h5", compile=False)
    return model

model = load_model()

# --- Helper functions ---
def extract_frame_sequences(video_path, sequence_length=10, frame_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames, sequences = [], []
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

uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    st.video(video_path)

    st.write("⏳ Extracting frames and running predictions...")
    sequences = extract_frame_sequences(video_path)

    if len(sequences) == 0:
        st.error("No valid frame sequences found in this video.")
    else:
        preds = model.predict(sequences)
        avg_pred = np.mean(preds, axis=0)
        label = "FAKE" if avg_pred[1] > avg_pred[0] else "REAL"
        confidence = float(max(avg_pred))
        st.subheader(f"🧠 Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")

st.caption("Model: deepfake_detection_best.h5 | Framework: TensorFlow + Streamlit")
