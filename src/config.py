from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    data_root: Path = Path("data")
    artifacts: Path = Path("artifacts")

    # ✅ Add dataset paths
    wesad_path: Path = Path("data/wesad")
    deap_path: Path = Path("data/deap")

@dataclass
class HyperParams:
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 30
    num_classes: int = 3

@dataclass
class Config:
    paths: Paths = Paths()
    hparams: HyperParams = HyperParams()
    
    # ✅ Which dataset to use
    dataset: str = "deap"   # or "deap"

# ✅ Global config
CFG = Config()