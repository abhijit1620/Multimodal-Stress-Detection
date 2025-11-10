import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import torch


def record_audio(filename="recorded.wav", duration=3, sr=16000, **kwargs):
    """
    Record microphone audio and save to .wav file.
    Accepts duration or seconds.
    """
    if "seconds" in kwargs:
        duration = kwargs["seconds"]

    print("🎤 Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(filename, audio, sr)
    print("✅ Audio saved:", filename)
    return filename


def extract_mfcc_from_audio(filename, sr=16000, n_mfcc=128, max_frames=128):
    """
    Extract MFCC features and convert to tensor shape [1, n_mfcc, max_frames]
    """
    y, sr = librosa.load(filename, sr=sr)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Force fixed length (pad or crop)
    if mfcc.shape[1] < max_frames:
        pad = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_frames]

    # Convert to tensor [1, n_mfcc, max_frames]
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
    return mfcc_tensor