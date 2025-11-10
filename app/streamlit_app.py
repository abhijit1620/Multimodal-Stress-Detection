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
from src.models.fusion import FusionModel
from src.data.face_utils import capture_frame_from_webcam
from src.data.audio_utils import record_audio, extract_mfcc_from_audio
from src.data.wesad_loader import WESADDataset
from src.data.deap_loader import DEAPDataset


# -------------------- UI SETUP --------------------
st.set_page_config(page_title="Multimodal Stress Detection", layout="centered")
st.title("📊 Multimodal Stress Detection Dashboard")
st.write("Fusion of **Face + Audio + Physiological Sensors**")


# -------------------- SESSION STATE --------------------
for key in ["face_tensor", "audio_tensor", "sensors_tensor", "ground_truth"]:
    if key not in st.session_state:
        st.session_state[key] = None


# -------------------- MODEL LOAD --------------------
num_classes = getattr(CFG.hparams, "num_classes", 3)
model = FusionModel(num_classes=num_classes)
model_path = Path("artifacts/fusion_model.pth")

if model_path.exists():
    checkpoint = torch.load(str(model_path), map_location="cpu")
    model_state = model.state_dict()

    # load only matching layers
    filtered = {k: v for k, v in checkpoint.items()
                if k in model_state and v.shape == model_state[k].shape}

    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)
    model.eval()
else:
    st.warning("⚠️ Model checkpoint not found. Train first.")


# -------------------- CONFUSION MATRIX --------------------
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    st.pyplot(plt)


# -------------------- WEBCAM CAPTURE --------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("📸 Capture Webcam"):
        try:
            face_tensor, img = capture_frame_from_webcam(resize=(224, 224))
            st.session_state.face_tensor = face_tensor
            st.image(img, caption="Webcam Capture")
            st.success("✅ Face captured")
        except Exception as e:
            st.error(f"Webcam capture failed: {e}")


# -------------------- AUDIO CAPTURE --------------------
with col2:
    rec_dur = st.number_input("🎤 Audio Duration (sec)", 1, 10, 3)
    if st.button("🎙️ Record Audio"):
        try:
            audio_file = record_audio(seconds=int(rec_dur), sr=16000)
            st.session_state.audio_tensor = extract_mfcc_from_audio(
                audio_file, sr=16000, n_mfcc=128, max_frames=128
            )
            st.success("✅ Audio recorded & MFCC extracted")
        except Exception as e:
            st.error(f"Audio failed: {e}")
            st.session_state.audio_tensor = None


# -------------------- SENSOR UPLOAD --------------------
st.markdown("---")
st.subheader("📟 Sensor Data Input")

uploaded = st.file_uploader("Upload sensor (.npy) file", type=["npy"])
if uploaded:
    try:
        arr = np.load(uploaded)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        st.session_state.sensors_tensor = torch.tensor(arr, dtype=torch.float32)
        st.success(f"✅ Loaded sensors shape: {arr.shape}")
    except Exception as e:
        st.error(f"❌ Failed to load: {e}")


# -------------------- LOAD SAMPLE FROM DATASETS --------------------
col3, col4 = st.columns(2)
with col3:
    if st.button("🔹 WESAD Sample"):
        ds = WESADDataset()
        x, y = next(iter(DataLoader(ds, batch_size=1, shuffle=True)))
        st.session_state.sensors_tensor = x
        st.session_state.ground_truth = y
        st.info(f"✅ Loaded WESAD Sample (true: {y.item()})")

with col4:
    if st.button("🔹 DEAP Sample"):
        ds = DEAPDataset()
        x, y = next(iter(DataLoader(ds, batch_size=1, shuffle=True)))
        st.session_state.sensors_tensor = x
        st.session_state.ground_truth = y
        st.info(f"✅ Loaded DEAP Sample (true: {y.item()})")


# -------------------- RUN PREDICTION --------------------
st.markdown("---")
if st.button("🚀 Run Prediction"):

    sensors = st.session_state.sensors_tensor
    if sensors is None:
        st.error("⚠️ Upload or load sensors first")
    else:
        if sensors.dim() == 2:
            sensors = sensors.unsqueeze(0)

        B = sensors.shape[0]

        # safe placeholders
        face = (st.session_state.face_tensor.repeat(B, 1, 1, 1)
                if isinstance(st.session_state.face_tensor, torch.Tensor)
                else torch.zeros((B, 3, 224, 224)))

        audio = (st.session_state.audio_tensor.repeat(B, 1, 1)
                if isinstance(st.session_state.audio_tensor, torch.Tensor)
                else torch.zeros((B, 128, 128)))

        with torch.no_grad():
            logits = model(face, audio, sensors)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        st.success(f"🎯 Prediction: {[['Low','Medium','High'][int(p)] for p in pred]}")


        # ✅ Metrics only if ground truth exists
        if st.session_state.ground_truth is not None:
            true = st.session_state.ground_truth.cpu().numpy()

            acc  = accuracy_score(true, pred)
            prec = precision_score(true, pred, average="weighted", zero_division=0)
            rec  = recall_score(true, pred, average="weighted", zero_division=0)
            f1   = f1_score(true, pred, average="weighted", zero_division=0)
            loss_v = F.cross_entropy(logits, torch.tensor(true, dtype=torch.long)).item()

            st.write(f" Accuracy: {acc:.3f}")
            st.write(f" Precision: {prec:.3f}")
            st.write(f" Recall: {rec:.3f}")
            st.write(f" F1-score: {f1:.3f}")
            st.write(f" Loss: {loss_v:.3f}")

            cm = confusion_matrix(true, pred, labels=[0, 1, 2])
            plot_confusion_matrix(cm, ["Low", "Medium", "High"])