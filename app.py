import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
import cv2
from tensorflow.keras.preprocessing import image
import plotly.graph_objects as go
from utils.video_processing import extract_frames

# -------------------- APP CONFIG --------------------
st.set_page_config(page_title="Zero-Trust Deepfake Detector", layout="wide")
st.title("🛡️ Zero-Trust Deepfake Detector")
st.caption("Adversarially Robust Detection with Full Explainability")

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_detection_best.keras")

with st.spinner("Loading detection model..."):
    model = load_model()

# -------------------- INTERFACE LAYOUT --------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file (MP4, AVI, MOV)", 
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file:
        # Save the uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Display uploaded video
        st.video(tfile.name)
        analyze = st.button("🔍 Analyze Video")

with col2:
    st.header("🎯 Detection Results")
    result_placeholder = st.empty()

# -------------------- PROCESSING --------------------
if uploaded_file and analyze:
    with st.spinner("Analyzing video frames..."):
        frames = extract_frames(tfile.name, frame_rate=1)
        predictions = []

        for frame in frames:
            img = cv2.resize(frame, (299, 299))  # Xception input size
            img_array = np.expand_dims(img, axis=0) / 255.0
            pred = model.predict(img_array, verbose=0)[0][0]
            predictions.append(pred)

        # Aggregate predictions
        avg_pred = np.mean(predictions)
        fake_prob = avg_pred * 100
        real_prob = (1 - avg_pred) * 100
        confidence = max(fake_prob, real_prob)
        label = "✅ LIKELY REAL" if avg_pred < 0.5 else "❌ LIKELY FAKE"
        color = "#00CC66" if avg_pred < 0.5 else "#FF4B4B"

    # -------------------- DISPLAY RESULTS --------------------
    with col2:
        st.markdown(
            f"""
            <div style="background-color:{color}20;padding:1.2rem;border-radius:10px;">
                <h3 style="color:{color};text-align:center;">{label}</h3>
                <p style="text-align:center;font-size:16px;">
                    Confidence: <b>{confidence:.2f}%</b><br>
                    Real Probability: {real_prob:.2f}%<br>
                    Fake Probability: {fake_prob:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True
        )

        # Plotly Gauge Visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence,
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "#E5F5E0"},
                    {'range': [50, 100], 'color': "#FFEBEE"}
                ],
                'threshold': {'line': {'color': color, 'width': 4}, 'thickness': 0.8, 'value': confidence}
            },
            title={'text': "Confidence Level (%)"}
        ))
        fig.update_layout(height=300, margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
