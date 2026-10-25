"""Paths and hyperparameters, in one place so that every run is fully described by a config.json."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "FracAtlas"        # the unzipped download, never modified
INTERIM_DIR = ROOT / "data" / "interim"              # audit tables
PROCESSED_DIR = ROOT / "data" / "processed"          # square, resized grayscale PNGs
SPLITS_CSV = ROOT / "data" / "splits.csv"            # one row per image: split, group, label, metadata
RUNS_DIR = ROOT / "runs"                             # one folder per training run

# Body-part and acquisition columns of FracAtlas' dataset.csv, kept for subgroup analysis.
META_COLS = ["hand", "leg", "hip", "shoulder", "mixed", "hardware", "multiscan", "frontal", "lateral", "oblique"]

# ImageNet statistics: a pretrained network expects inputs normalized the way it was trained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    model: str = "resnet18"        # "resnet18" (transfer learning) or "simple_cnn" (from scratch)
    pretrained: bool = True        # ImageNet weights for resnet18
    processed_size: int = 256      # size of the cached PNGs
    image_size: int = 224          # size seen by the network
    batch_size: int = 32
    epochs: int = 30
    freeze_epochs: int = 3         # first epochs train the head only, backbone frozen
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4      # smaller: pretrained features only need small adjustments
    weight_decay: float = 1e-4
    dropout: float = 0.2
    patience: int = 7              # early stopping on validation average precision
    pos_weight: bool = True        # weight the positive class by n_neg / n_pos in the loss
    max_grad_norm: float = 1.0     # gradient clipping, 0 to disable
    num_workers: int = 4
    seed: int = 42
    run_name: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
