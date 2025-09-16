from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    data_root: Path = Path("data")
    artifacts: Path = Path("artifacts")

@dataclass
class HyperParams:
    lr: float = 1e-3          # learning rate (stable)
    batch_size: int = 16      # thoda bada karo (GPU hai to 32 bhi try kar)
    epochs: int = 30          # kam se kam 20–50 epochs for convergence
    num_classes: int = 3      # Low, Medium, High

@dataclass
class Config:
    paths: Paths = Paths()
    hparams: HyperParams = HyperParams()
    dataset: str = "deap"   # options:  "wesad", "deap"

# global config object
CFG = Config()
