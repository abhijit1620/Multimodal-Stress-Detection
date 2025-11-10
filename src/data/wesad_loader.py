import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WESADDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/wesad",
        sensor: str = "chest",
        preload: bool = True,
    ):
        self.data_dir = os.path.expanduser(data_dir)
        self.sensor = sensor
        self.preload = preload

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"wesad folder not found: {self.data_dir}")

        self.subjects = self._discover_subjects()
        self._files = self._gather_pkl_files()

        self.data = []
        self.labels = []

        if self.preload:
            self._preload_all()

    def _discover_subjects(self) -> List[str]:
        names = []
        for entry in sorted(os.listdir(self.data_dir)):
            p = os.path.join(self.data_dir, entry)
            if os.path.isdir(p) and entry.lower().startswith("s"):
                names.append(entry)
        return names

    def _gather_pkl_files(self) -> List[str]:
        files = []
        for subj in self.subjects:
            pkl_nested = os.path.join(self.data_dir, subj, f"{subj}.pkl")
            if os.path.isfile(pkl_nested):
                files.append(pkl_nested)
        if not files:
            raise FileNotFoundError("No wesad .pkl files found")
        return sorted(files)

    def _load_pkl(self, path: str):
        with open(path, "rb") as f:
            try:
                return pickle.load(f, encoding="latin1")
            except:
                f.seek(0)
                return pickle.load(f)

    def _preload_all(self):
        fixed_length = 4000000  # choose a suitable length for your data

        for path in self._files:
            d = self._load_pkl(path)

            if "signal" not in d:
                continue

            sig = d["signal"]

            # ✅ Chest has nested signals -> concatenate all numeric sub-keys
            if isinstance(sig[self.sensor], dict):
                parts = []
                for k, v in sig[self.sensor].items():
                    try:
                        arr = np.asarray(v, dtype=np.float32)
                        parts.append(arr)
                    except:
                        continue

                if len(parts) == 0:
                    continue

                # concatenate across last axis
                try:
                    signal = np.concatenate(parts, axis=-1)
                except:
                    signal = parts[0]

            else:
                signal = np.asarray(sig[self.sensor], dtype=np.float32)

            # Pad or truncate signal to fixed_length
            if signal.shape[0] > fixed_length:
                signal = signal[:fixed_length]
            elif signal.shape[0] < fixed_length:
                pad_width = fixed_length - signal.shape[0]
                signal = np.pad(signal, ((0, pad_width), (0, 0)), mode='constant')

            # ✅ label
            label = d.get("label", None)
            if label is None:
                label = d.get("labels", None)

            try:
                arr = np.asarray(label)
                if arr.size > 1:
                    arr = arr.astype(int)
                    label = int(np.bincount(arr).argmax())
                else:
                    label = int(arr.squeeze())
            except Exception:
                label = 0

            self.data.append(signal)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = np.asarray(self.data[idx], dtype=np.float32)
        label = int(self.labels[idx])

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label


# Quick test
if __name__ == "__main__":
    ds = WESADDataset(data_dir="data/wesad", sensor="chest", preload=True)
    print("Loaded:", len(ds))
    x, y = ds[0]
    print(x.shape, y)