import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
from pathlib import Path
import numpy as np

from src.config import CFG
from src.models.fusion import FusionNet
from src.data.face_utils import capture_frame_from_webcam
from src.data.audio_utils import record_audio, extract_mfcc_from_audio
from src.data.wesad_loader import WESADDataset
from src.data.deap_loader import DEAPDataset
from torch.utils.data import DataLoader

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Multimodal Stress Detection", layout="centered")

st.title("📊 Multimodal Stress Detection")
st.write("Fusion of **Face + Audio + Physiological Sensors**")
st.info("Make sure you have trained model saved at `artifacts/fusion_model.pth`")

# -------------------------------
# Load Model
# -------------------------------
sensor_channels = 6 if getattr(CFG, "dataset", None) == "wesad" else 32
model = FusionNet(num_classes=CFG.hparams.num_classes, sensor_in_channels=sensor_channels)

model_path = Path("artifacts/fusion_model.pth")
if not model_path.exists():
    st.warning("⚠️ Model checkpoint not found. Please train first.")
else:
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    st.success("✅ Model loaded successfully!")

# -------------------------------
# Session state init
# -------------------------------
if "face_tensor" not in st.session_state:
    st.session_state.face_tensor = None
if "audio_tensor" not in st.session_state:
    st.session_state.audio_tensor = None
if "sensors_tensor" not in st.session_state:
    st.session_state.sensors_tensor = None  

# -------------------------------
# Face Capture (Webcam)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("📸 Capture Face (Webcam)"):
        try:
            face_t, disp_img = capture_frame_from_webcam(resize=(112,112))
            st.session_state.face_tensor = face_t
            st.image(disp_img, use_container_width=True, caption="Captured (resized)")
            
            st.success("✅ Face captured, cropped and preprocessed.")
        except Exception as e:
            st.error(f"Webcam capture failed: {e}")
     
# -------------------------------
# Audio Capture (Mic)
# -------------------------------
with col2:
    rec_dur = st.number_input("🎤 Audio record seconds", min_value=1, max_value=10, value=3, step=1)
    if st.button("🎙️ Record Audio"):
        try:
            audio_np = record_audio(duration=int(rec_dur), sr=16000)
            st.session_state.audio_tensor = extract_mfcc_from_audio(
                audio_np, sr=16000, n_mfcc=40, max_frames=100
            )  # [1,100,40]
            st.success("✅ Audio recorded and MFCC extracted.")
        except Exception as e:
            st.error(f"Audio recording failed: {e}")

# -------------------------------
# Sensors Input (Upload or Dataset)
# -------------------------------
st.markdown("---")
st.subheader("📟 Sensors Input")

uploaded = st.file_uploader("Upload sensor numpy (.npy) file [C,T] or [B,C,T] (optional)", type=["npy"])
if uploaded is not None:
    try:
        arr = np.load(uploaded)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        st.session_state.sensors_tensor = torch.tensor(arr, dtype=torch.float32)
        st.success(f"✅ Loaded uploaded sensors data shape {arr.shape}")
    except Exception as e:
        st.error(f"Failed to load numpy file: {e}")

col3, col4 = st.columns(2)
with col3:
    if st.button("🔹 Load One Batch from WESAD"):
        ds = WESADDataset()
        dl = DataLoader(ds, batch_size=1, shuffle=True)
        sensors, label = next(iter(dl))
        st.session_state.sensors_tensor = sensors
        st.info(f"WESAD sample loaded. True Label = {label.item()}")

with col4:
    if st.button("🔹 Load One Batch from DEAP"):
        ds = DEAPDataset()
        dl = DataLoader(ds, batch_size=1, shuffle=True)
        sensors, label = next(iter(dl))
        st.session_state.sensors_tensor = sensors
        st.info(f"DEAP sample loaded. True Label = {label.item()}")

# -------------------------------
# Run Model Prediction
# -------------------------------
st.markdown("---")
if st.button("🚀 Run Model Prediction"):
    if not model_path.exists():
        st.error("No model file found. Train first.")
    elif st.session_state.sensors_tensor is None:
        st.error("Please upload or load sensors data first.")
    else:
        sensors = st.session_state.sensors_tensor
        if sensors.dim() == 2:
            sensors = sensors.unsqueeze(0)
        if sensors.dim() == 3 and sensors.shape[2] == sensor_channels:
            sensors = sensors.transpose(1,2)  # [B,C,T]

        # face/audio fallback: if not captured, use zeros with appropriate shapes
        if st.session_state.face_tensor is None:
           face = torch.zeros((sensors.shape[0], 3, 112, 112), dtype=torch.float32)
        else:
           face = st.session_state.face_tensor.repeat(sensors.shape[0],1,1,1)

        if st.session_state.audio_tensor is None:
           audio = torch.zeros((sensors.shape[0], 100, 40), dtype=torch.float32)
        else:
           audio = st.session_state.audio_tensor.repeat(sensors.shape[0],1,1)

        with torch.no_grad():
            logits = model(face, audio, sensors)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        labels = ["Low","Medium","High"]
        st.success(f"🎯 Predicted Stress Levels: {[labels[int(p)] for p in pred]}")