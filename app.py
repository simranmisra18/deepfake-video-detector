import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, TimeDistributed, Bidirectional, GRU
)
from tensorflow.keras.models import Model, Sequential

st.set_page_config(page_title="Deepfake Detection", layout="centered")
st.title("🎭 Deepfake Detection (Meso4 + Bi-GRU)")
st.write("Upload a video to detect whether it’s **REAL** or **FAKE** using the trained Meso4 model.")

# --- Model parameters ---
SEQ_LEN = 10
FRAME_SIZE = (128, 128)

# --- Build Meso4 architecture (same as in your notebook) ---
@st.cache_resource
def build_meso4_model(input_shape=(SEQ_LEN, 128, 128, 3)):
    # Frame-level CNN
    frame_input = Input(shape=(128, 128, 3))
    x = Conv2D(8, (3, 3), padding="same", activation="relu", kernel_regularizer=l2(0.001))(frame_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)

    x = Conv2D(8, (5, 5), padding="same", activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.15)(x)

    x = Conv2D(16, (5, 5), padding="same", activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(16, (5, 5), padding="same", activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    frame_cnn = Model(inputs=frame_input, outputs=x, name="Meso4_FrameCNN")

    # Video-level temporal model
    model = Sequential([
        TimeDistributed(frame_cnn, input_shape=input_shape),
        BatchNormalization(),
        Bidirectional(GRU(128, return_sequences=False, kernel_regularizer=l2(0.005))),
        Dropout(0.5),
        Dense(64, activation="relu", kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation="softmax")
    ])
    return model


# --- Load weights ---
@st.cache_resource
def load_model():
    model = build_meso4_model()
    model.load_weights("deepfake_detection_best.weights.h5")
    return model


model = load_model()


# --- Helper: extract sequences from video ---
def extract_frame_sequences(video_path, sequence_length=SEQ_LEN, frame_size=FRAME_SIZE):
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


# --- Streamlit UI ---
uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov", "mkv"])
if uploaded_video:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())
    video_path = temp_file.name
    st.video(video_path)

    st.info("⏳ Extracting frames and running predictions...")
    sequences = extract_frame_sequences(video_path)

    if len(sequences) == 0:
        st.error("No valid 10-frame sequences found in this video.")
    else:
        preds = model.predict(sequences)
        avg_pred = np.mean(preds, axis=0)
        label = "FAKE" if avg_pred[1] > avg_pred[0] else "REAL"
        confidence = float(max(avg_pred))

        st.subheader(f"🧠 Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        st.success("✅ Analysis complete!")


st.caption("Model: Meso4 + Bi-GRU | Framework: TensorFlow + Streamlit")
