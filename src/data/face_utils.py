import cv2
import torch
import numpy as np

def capture_frame_from_webcam(resize=(224, 224)):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # ✅ Force camera to brighten
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 1.0)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # lower negative = brighter on many webcams

    # Give webcam time to adjust brightness
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture frame")

    # ✅ Extra enhancement: brighten dark frames
    frame = cv2.convertScaleAbs(frame, alpha=1.7, beta=50)

    # Convert to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img_resized = cv2.resize(img, resize)

    # Convert to tensor
    tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    return tensor, img_resized