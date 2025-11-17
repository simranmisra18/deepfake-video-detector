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
st.title("🎭 Deepfake Detection (Xception + Bi-GRU with Grad-CAM)")
st.write("Upload a video to detect whether it’s **REAL** or **FAKE**, and visualize Grad-CAM activation maps.")

SEQ_LEN = 10
FRAME_SIZE = (128, 128)

# ----------------------------
# Build Xception architecture
# ----------------------------
@st.cache_resource
def build_meso4_model(input_shape=(SEQ_LEN, 128, 128, 3)):
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

    x = Conv2D(16, (5, 5), padding="same", activation="relu", kernel_regularizer=l2(0.001), name="last_conv")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    frame_cnn = Model(inputs=frame_input, outputs=x, name="Xception_FrameCNN")

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


@st.cache_resource
def load_model():
    model = build_meso4_model()
    model.load_weights("deepfake_detection_best.weights.h5")
    return model

model = load_model()


# ----------------------------
# Grad-CAM Implementation
# ----------------------------
def generate_gradcam(model, frame, layer_name="last_conv", cls_idx=None):
    """
    Compute GradCAM for a single frame through the frame-level CNN.
    """
    frame_cnn = model.layers[0].layer  # Extract the frame-level CNN (TimeDistributed layer)
    grad_model = Model(
        inputs=frame_cnn.input,
        outputs=[frame_cnn.get_layer(layer_name).output, frame_cnn.output]
    )

    with tf.GradientTape() as tape:
        inputs = np.expand_dims(frame, axis=0)
        conv_outputs, predictions = grad_model(inputs)
        if cls_idx is None:
            cls_idx = np.argmax(predictions[0])
        loss = predictions[:, cls_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = cv2.resize(heatmap.numpy(), (frame.shape[1], frame.shape[0]))

    # Overlay heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    return overlay


# ----------------------------
# Extract sequences
# ----------------------------
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


# ----------------------------
# Streamlit UI
# ----------------------------
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

        # Select a frame from first sequence for Grad-CAM
        st.write("---")
        st.write("🔍 Grad-CAM visualization for one representative frame:")
        test_frame = sequences[0][5]  # middle frame of first sequence
        gradcam_image = generate_gradcam(model, test_frame, layer_name="last_conv", cls_idx=np.argmax(avg_pred))

        st.image([test_frame, gradcam_image], caption=["Original Frame", "Grad-CAM Activation"], width=300)
        st.success("✅ Analysis and Grad-CAM complete!")

st.caption("Model: Xception + Bi-GRU | Grad-CAM Visualization Enabled | TensorFlow + Streamlit")
