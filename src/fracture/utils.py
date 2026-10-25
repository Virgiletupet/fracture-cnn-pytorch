"""Small helpers: reproducibility, device selection, JSON I/O."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random generator we use. GPU kernels (MPS, CUDA) can still be slightly non-deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Apple GPU (MPS) on a Mac, CUDA on an NVIDIA machine, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
