from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    data_root: Path = Path("data")
    artifacts: Path = Path("artifacts")

@dataclass
class HyperParams:
    lr: float = 1e-3
    batch_size: int = 8
    epochs: int = 1
    num_classes: int = 3  # Low, Medium, High

@dataclass
class Config:
    paths: Paths = Paths()
    hparams: HyperParams = HyperParams()
    dataset: str = "wesad"   # options:  "wesad", "deap"

# global config object
CFG = Config()
