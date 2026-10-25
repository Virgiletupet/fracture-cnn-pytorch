"""Images -> tensors.

Shapes and dtypes, from disk to the network:
    PNG on disk          (256, 256)          uint8, grayscale, values 0..255
    decode_image         (1, 256, 256)       uint8 tensor, channels first
    train transforms     (1, 224, 224)       uint8: random crop, flip, rotation, brightness/contrast
    ToDtype(scale=True)  (1, 224, 224)       float32 in [0, 1]
    GrayToRGB            (3, 224, 224)       ImageNet models expect 3 channels
    Normalize            (3, 224, 224)       float32, roughly zero mean and unit variance per channel
    DataLoader batch     (32, 3, 224, 224)   float32; labels (32,) float32 in {0, 1}
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

from .config import IMAGENET_MEAN, IMAGENET_STD, PROCESSED_DIR, SPLITS_CSV, TrainConfig


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Grayscale, pad to a square with black (X-ray backgrounds are dark), then resize.
    Padding instead of stretching keeps anatomy undistorted and hides the original format,
    which correlates with the label in FracAtlas."""
    img = img.convert("L")
    side = max(img.size)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def prepare_images(size: int = 256) -> None:
    """Cache every kept image as a small PNG once: decoding 2880x2304 JPEGs at every epoch would be slow."""
    df = pd.read_csv(SPLITS_CSV)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for path, image_id in zip(df["path"], df["image_id"]):
        out = PROCESSED_DIR / f"{Path(image_id).stem}.png"
        if not out.exists():
            with Image.open(path) as im:
                letterbox(im, size).save(out)
    print(f"{len(df)} images cached in {PROCESSED_DIR}")


class GrayToRGB(nn.Module):
    """(1, H, W) -> (3, H, W) by repeating the channel. A class rather than a lambda, so that
    DataLoader workers (separate processes on macOS) can pickle it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(3, 1, 1)


def build_transforms(train: bool, image_size: int = 224) -> v2.Compose:
    """Augmentation only for training. Validation and test images are only resized and normalized,
    so that the evaluation is deterministic."""
    if train:
        geometric = [
            v2.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.RandomHorizontalFlip(),       # left and right limbs are mirror images
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.2, contrast=0.2),   # exposure varies between devices
        ]
    else:
        geometric = [v2.Resize(image_size, antialias=True)]
    return v2.Compose([
        *geometric,
        v2.ToDtype(torch.float32, scale=True),   # uint8 0..255 -> float32 0..1
        GrayToRGB(),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FractureDataset(Dataset):
    """One item = (image tensor (3, 224, 224) float32, label () float32, image_id str)."""

    def __init__(self, frame: pd.DataFrame, transform: v2.Compose, image_dir: Path = PROCESSED_DIR):
        self.image_dir = image_dir
        self.ids = frame["image_id"].tolist()
        self.labels = frame["label"].astype("float32").tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        path = self.image_dir / f"{Path(self.ids[i]).stem}.png"
        image = decode_image(str(path), mode=ImageReadMode.GRAY)      # (1, 256, 256) uint8
        return self.transform(image), torch.tensor(self.labels[i]), self.ids[i]


def load_splits() -> pd.DataFrame:
    return pd.read_csv(SPLITS_CSV)


def make_loaders(cfg: TrainConfig, frame: pd.DataFrame | None = None, image_dir: Path = PROCESSED_DIR) -> dict[str, DataLoader]:
    """Shuffle the training set only. drop_last avoids a final batch of 1, which BatchNorm cannot normalize."""
    frame = load_splits() if frame is None else frame
    loaders = {}
    for split in ["train", "val", "test"]:
        part = frame[frame["split"] == split]
        if part.empty:
            continue
        train = split == "train"
        loaders[split] = DataLoader(
            FractureDataset(part, build_transforms(train, cfg.image_size), image_dir),
            batch_size=cfg.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=cfg.num_workers,
            persistent_workers=cfg.num_workers > 0,
        )
    return loaders
