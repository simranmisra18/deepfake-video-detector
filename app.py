import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
import cv2
from utils.video_processing import extract_frames
from streamlit_gauge import gauge  # pip install streamlit-gauge

# --- Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_detection_best.keras")

model = load_model()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Zero-Trust Deepfake Detector", layout="wide")
st.title("🛡️ Zero-Trust Deepfake Detector")
st.write("Adversarially Robust Detection with Full Explainability")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Video")
    uploaded_file = st.file_uploader("Choose a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.video(tfile.name)
        analyze = st.button("🔍 Analyze Video")

with col2:
    st.header("🎯 Detection Results")
    placeholder = st.empty()

if uploaded_file and analyze:
    frames = extract_frames(tfile.name, frame_rate=1)
    predictions = []

    for frame in frames:
        img = cv2.resize(frame, (299, 299))  # Xception input
        img_array = np.expand_dims(img, axis=0) / 255.0
        pred = model.predict(img_array)[0][0]
        predictions.append(pred)

    avg_pred = np.mean(predictions)
    fake_prob = avg_pred * 100
    real_prob = (1 - avg_pred) * 100

    label = "✅ LIKELY REAL" if avg_pred < 0.5 else "❌ LIKELY FAKE"
    color = "green" if avg_pred < 0.5 else "red"

    with col2:
        st.markdown(f"### <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {max(fake_prob, real_prob):.2f}%")
        st.write(f"**Real Probability:** {real_prob:.2f}%")
        st.write(f"**Fake Probability:** {fake_prob:.2f}%")
        gauge(label="Confidence Level", value=int(max(fake_prob, real_prob)), min_value=0, max_value=100, delta=int((fake_prob - real_prob)/2))
