import torch
from torch.utils.data import DataLoader
from src.data.deap_loader import DEAPDataset
from src.data.wesad_loader import WESADDataset

# ---- Check DEAP ----
print("🔹 Checking DEAP dataset...")
deap_ds = DEAPDataset()
deap_dl = DataLoader(deap_ds, batch_size=4, shuffle=True)
x, y = next(iter(deap_dl))
print("DEAP batch shape:", x.shape, "| Labels:", y)

# ---- Check WESAD ----
print("\n🔹 Checking WESAD dataset...")
wesad_ds = WESADDataset()
wesad_dl = DataLoader(wesad_ds, batch_size=4, shuffle=True)
x, y = next(iter(wesad_dl))
print("WESAD batch shape:", x.shape, "| Labels:", y)
