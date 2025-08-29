import torch
from torch.utils.data import DataLoader
import torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import os

# 🔹 absolute imports
from src.config import CFG
from src.models.fusion import FusionNet
from src.data.wesad_loader import WESADDataset
from src.data.deap_loader import DEAPDataset


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    dev = device()

    # Sensor input channels depend on dataset (wesad=6, deap=32)
    in_channels = 6
    if getattr(CFG, "dataset", None) == "deap":
        in_channels = 32

    model = FusionNet(
        num_classes=CFG.hparams.num_classes,
        sensor_in_channels=in_channels
    ).to(dev)

    # --------------------------
    # Select Dataset
    # --------------------------
    if getattr(CFG, "dataset", None) == "wesad":
        ds = WESADDataset()
    elif getattr(CFG, "dataset", None) == "deap":
        ds = DEAPDataset()
    else:
        raise ValueError(f"Unknown dataset: {getattr(CFG, 'dataset', None)}")

    dl = DataLoader(ds, batch_size=CFG.hparams.batch_size, shuffle=True)

    # --------------------------
    # Training loop
    # --------------------------
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=CFG.hparams.lr)
    model.train()

    for ep in range(CFG.hparams.epochs):
        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{CFG.hparams.epochs}")
        for x, y in pbar:
            x, y = x.to(dev), y.to(dev)

            # ✅ Fix sensor shape
            if x.dim() == 3:
                if x.shape[2] == in_channels:     # [B, T, C] → transpose
                    x = x.transpose(1, 2)        # -> [B, C, T]
                elif x.shape[1] == in_channels:  # already [B, C, T]
                    pass
                else:
                    raise ValueError(
                        f"Unexpected sensor shape {x.shape}, expected channel={in_channels}"
                    )

            # Dummy face + audio until real preprocessing is ready
            logits = model(
                torch.zeros((x.size(0), 3, 224, 224), device=dev),   # fake face
                torch.zeros((x.size(0), 1, 128, 128), device=dev),   # fake audio
                x                                                    # real sensors
            )

            # --------------------------
            # Loss + Backprop
            # --------------------------
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=loss.item())

    print("✅ Training finished.")

    # 🔹 Save trained model
    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/fusion_model.pth")
    print("💾 Model saved at artifacts/fusion_model.pth")


if __name__ == "__main__":
    main()
