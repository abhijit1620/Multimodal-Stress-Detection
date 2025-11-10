import os
import numpy as np
import torch
from torch.utils.data import Dataset

class DEAPDataset(Dataset):
    def __init__(self, data_dir="data/deap/data_preprocessed_python", preload=True):
        self.data_dir = data_dir
        self.samples = []

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"❌ deap folder not found at {self.data_dir}")

        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith(".dat")])
        print(f"✅ Found {len(files)} subject files")

        if preload:
            for f in files:
                path = os.path.join(self.data_dir, f)

                arr = np.load(path, allow_pickle=True)
                if not isinstance(arr, dict):
                    arr = arr.item()

                signals = arr["data"]        # (40, 40, 8064)
                labels = arr["labels"]       # (40, 4)

                for trial_idx in range(signals.shape[0]):
                    x = signals[trial_idx, :32, :]   # EEG only
                    y = labels[trial_idx]

                    label = 1 if y[1] > 5 else 0     # binary stress

                    self.samples.append((x, label))

        print(f"✅ Loaded {len(self.samples)} deap samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y