"""Step 4. Train a model. Examples:
    python scripts/04_train.py --model resnet18 --run-name resnet18
    python scripts/04_train.py --model simple_cnn --run-name simple_cnn --epochs 40 --freeze-epochs 0 --lr-head 1e-3
"""
import argparse
from dataclasses import fields

from fracture.config import TrainConfig
from fracture.train import fit

if __name__ == "__main__":          # required on macOS: DataLoader workers re-import this file
    parser = argparse.ArgumentParser()
    for f in fields(TrainConfig):   # every config field becomes a command-line option
        if f.type in ("bool", bool):
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    cfg = TrainConfig(**vars(parser.parse_args()))
    fit(cfg)
