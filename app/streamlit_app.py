import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CFG
from src.models.fusion import FusionNet
from src.data.face_utils import capture_frame_from_webcam
from src.data.audio_utils import record_audio, extract_mfcc_from_audio
from src.data.wesad_loader import WESADDataset
from src.data.deap_loader import DEAPDataset

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Multimodal Stress Detection", layout="centered")
st.title("📊 Multimodal Stress Detection")
st.write("Fusion of **Face + Audio + Physiological Sensors**")
st.info("Make sure model is saved at `artifacts/fusion_model.pth`")

# -------------------------------
# Session state
# -------------------------------
for key in ["face_tensor","audio_tensor","sensors_tensor","ground_truth"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------
# Load Model
# -------------------------------
sensor_channels = 6 if getattr(CFG, "dataset", None) == "wesad" else 32
model = FusionNet(num_classes=CFG.hparams.num_classes, sensor_in_channels=sensor_channels)
model_path = Path("artifacts/fusion_model.pth")
if model_path.exists():
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    st.success("✅ Model loaded successfully!")
else:
    st.warning("⚠️ Model checkpoint not found. Please train first.")

# -------------------------------
# Confusion Matrix Plot
# -------------------------------
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    st.pyplot(plt)

# -------------------------------
# Face Capture
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("📸 Capture Webcam"):
        try:
            face_tensor, full_frame = capture_frame_from_webcam(resize=(224,224))
            st.session_state.face_tensor = face_tensor
            st.image(full_frame, caption="Webcam Capture", use_container_width=True)
            st.success("✅ Frame captured")
        except Exception as e:
            st.error(f"Webcam capture failed: {e}")

# -------------------------------
# Audio Capture
# -------------------------------
with col2:
    rec_dur = st.number_input("🎤 Audio record seconds", min_value=1, max_value=10, value=3, step=1)
    if st.button("🎙️ Record Audio"):
        try:
            audio_np = record_audio(duration=int(rec_dur), sr=16000)
            st.session_state.audio_tensor = extract_mfcc_from_audio(audio_np, sr=16000, n_mfcc=128, max_frames=128)
            st.success("✅ Audio recorded & MFCC extracted")
        except Exception as e:
            st.error(f"Audio recording failed: {e}")
            st.session_state.audio_tensor = None

# -------------------------------
# Sensors Input
# -------------------------------
st.markdown("---")
st.subheader("📟 Sensors Input")
uploaded = st.file_uploader("Upload sensor numpy (.npy) file [C,T] or [B,C,T] (optional)", type=["npy"])
if uploaded:
    try:
        arr = np.load(uploaded)
        if arr.ndim == 2: arr = arr[np.newaxis,...]
        st.session_state.sensors_tensor = torch.tensor(arr, dtype=torch.float32)
        st.success(f"✅ Loaded sensors shape {arr.shape}")
    except Exception as e:
        st.error(f"Failed to load numpy: {e}")

col3, col4 = st.columns(2)
with col3:
    if st.button("🔹 Load Batch from WESAD"):
        ds = WESADDataset()
        dl = DataLoader(ds, batch_size=1, shuffle=True)
        sensors, label = next(iter(dl))
        st.session_state.sensors_tensor = sensors
        st.session_state.ground_truth = label
        st.info(f"WESAD sample loaded. True Label = {label.item()}")
with col4:
    if st.button("🔹 Load Batch from DEAP"):
        ds = DEAPDataset()
        dl = DataLoader(ds, batch_size=1, shuffle=True)
        sensors, label = next(iter(dl))
        st.session_state.sensors_tensor = sensors
        st.session_state.ground_truth = label
        st.info(f"DEAP sample loaded. True Label = {label.item()}")

# -------------------------------
# Run Prediction + Metrics
# -------------------------------
st.markdown("---")
if st.button("🚀 Run Model Prediction"):
    if not model_path.exists():
        st.error("No model file found. Train first.")
    elif st.session_state.sensors_tensor is None:
        st.error("Upload/load sensors first.")
    else:
        sensors = st.session_state.sensors_tensor
        if sensors.dim() == 2: sensors = sensors.unsqueeze(0)
        if sensors.dim() == 3 and sensors.shape[2]==sensor_channels:
            sensors = sensors.transpose(1,2)  # [B,C,T]

        batch_size = sensors.shape[0]
        face_input = st.session_state.face_tensor.repeat(batch_size,1,1,1) if st.session_state.face_tensor is not None else torch.zeros((batch_size,3,112,112))
        audio_input = st.session_state.audio_tensor.repeat(batch_size,1,1) if st.session_state.audio_tensor is not None else torch.zeros((batch_size,128,128))

        with torch.no_grad():
            logits = model(face_input, audio_input, sensors)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            if st.session_state.ground_truth is not None:
                true_labels = st.session_state.ground_truth
                true_labels = true_labels.cpu().numpy() if torch.is_tensor(true_labels) else true_labels

                acc = accuracy_score(true_labels, pred)
                prec = precision_score(true_labels, pred, average='weighted', zero_division=0)
                rec = recall_score(true_labels, pred, average='weighted', zero_division=0)
                f1 = f1_score(true_labels, pred, average='weighted', zero_division=0)
                loss_val = F.cross_entropy(logits, torch.tensor(true_labels, dtype=torch.long)).item()
                cm = confusion_matrix(true_labels, pred)

                st.write(f"**Accuracy:** {acc:.3f}")
                st.write(f"**Precision:** {prec:.3f}")
                st.write(f"**Recall:** {rec:.3f}")
                st.write(f"**F1-score:** {f1:.3f}")
                st.write(f"**Loss:** {loss_val:.3f}")
                plot_confusion_matrix(cm, ["Low","Medium","High"])

        st.success(f"🎯 Predicted Stress Levels: {[['Low','Medium','High'][int(p)] for p in pred]}")