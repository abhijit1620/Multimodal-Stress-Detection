import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from src.data.wesad_loader import WESADDataset
from src.data.deap_loader import DEAPDataset
from src.models.fusion import FusionModel


def train(dataset_name="wesad", epochs=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ✅ Load Dataset
    if dataset_name.lower() == "wesad":
        print("Training on WESAD…")
        ds = WESADDataset(preload=True)
    elif dataset_name.lower() == "deap":
        print("Training on DEAP…")
        ds = DEAPDataset(preload=True)
    else:
        raise ValueError("Dataset must be 'wesad' or 'deap'")

    dl = DataLoader(ds, batch_size=1, shuffle=True)

    # ✅ Model, Optimizer
    model = FusionModel(num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}")

        for sensors, label in pbar:
            sensors = sensors.to(device)
            label = label.to(device)

            # ✅ Fix shape for both datasets: [B, 8, T]
            if sensors.ndim == 3 and sensors.shape[1] < sensors.shape[2]:
                sensors = sensors.permute(0, 2, 1)

            # ✅ DEAP: slice first 8 channels
            if sensors.shape[1] > 8:
                sensors = sensors[:, :8, :]

            optimizer.zero_grad()
            logits = model(sensors=sensors, face=None, audio=None)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

    # ✅ Save final model
    save_path = Path("artifacts/fusion_model.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved at {save_path}")


if __name__ == "__main__":
    # ✅ Train on both
    train("wesad", epochs=50)
    train("deap", epochs=50)