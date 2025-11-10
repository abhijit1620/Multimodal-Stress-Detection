import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.wesad_loader import WESADDataset
from src.data.deap_loader import DEAPDataset
from src.data.combined_dataset import CombinedDataset
from src.models.fusion import FusionModel
from pathlib import Path


def preprocess_sensors(x):
    """
    Ensures x → [B, 8, T] for model input.
    - WESAD chest: already 8 channels
    - DEAP: 32 EEG channels → take first 8
    - If [B, T, C] → permute → [B, C, T]
    """
    # [B, T, C] → [B, C, T]
    if x.ndim == 3 and x.shape[2] > x.shape[1]:
        x = x.permute(0, 2, 1)

    # If 32 EEG channels → take first 8
    if x.shape[1] > 8:
        x = x[:, :8, :]

    return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🚀 Loading Datasets...")

    wesad = WESADDataset(data_dir="data/WESAD", sensor="chest", preload=True)
    deap = DEAPDataset(data_dir="data/DEAP", preload=True)

    dataset = CombinedDataset([wesad, deap])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"✅ Total samples for training: {len(dataset)}")

    model = FusionModel(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 3

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

        for sensors, label in pbar:
            sensors = preprocess_sensors(sensors).to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(sensors=sensors, face=None, audio=None)

            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

    # ✅ Save trained fusion model
    save_path = Path("artifacts/fusion_model.pth")
    save_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete! Model saved at {save_path}")


if __name__ == "__main__":
    main()