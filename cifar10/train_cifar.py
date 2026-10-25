"""CIFAR-10, part 2 of the original project, rebuilt in PyTorch: CNNs trained from scratch.

What it shows:
- multi-class classification: logits (N, 10) and CrossEntropyLoss with int64 targets of shape (N,);
- four architectures and their trade-offs (accuracy, parameters, time per epoch);
- a proper protocol: 45,000 train / 5,000 validation (model selection) / 10,000 test (scored once).
  The original notebook trained on the validation images too, so its "validation accuracy" was optimistic.

Run: python cifar10/train_cifar.py            (about 15 minutes on an Apple M-series GPU)
Outputs: runs/cifar10/results.md, runs/cifar10/shapes.txt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data" / "cifar10", ROOT / "runs" / "cifar10"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)     # per-channel statistics of the training set


# ---------------------------------------------------------------- models

def conv_bn(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class BasicCNN(nn.Module):
    """The original "basic CNN": 3 conv blocks (32-64-64) without BatchNorm, then a dense layer."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # (N, 32, 16, 16)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # (N, 64, 8, 8)
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # (N, 64, 4, 4)
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 10))

    def forward(self, x):
        return self.classifier(self.features(x))                            # (N, 10) logits


class VGGBN(nn.Module):
    """Deeper VGG-style network: two 3x3 convs per stage, BatchNorm, global average pooling."""

    def __init__(self, widths=(64, 128, 256)):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers += [conv_bn(c, w), conv_bn(w, w), nn.MaxPool2d(2)]      # 32 -> 16 -> 8 -> 4
            c = w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class DepthwiseSeparable(nn.Module):
    """3x3 depthwise conv (one filter per channel, groups=c_in) then 1x1 pointwise conv.
    Weights: 9*c_in + c_in*c_out instead of 9*c_in*c_out, about 8 to 9 times fewer for wide layers."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False), nn.BatchNorm2d(c_in), nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseCNN(nn.Module):
    """VGGBN with every conv after the first replaced by a depthwise separable block (MobileNet idea)."""

    def __init__(self, widths=(64, 128, 256)):
        super().__init__()
        layers, c = [conv_bn(3, widths[0])], widths[0]
        for k, w in enumerate(widths):
            layers += ([] if k == 0 else [DepthwiseSeparable(c, w)]) + [DepthwiseSeparable(w, w), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    """Residual block: out = ReLU(F(x) + shortcut(x)). The identity path lets gradients flow through deep stacks."""

    def __init__(self, c_in: int, c_out: int, stride: int):
        super().__init__()
        self.f = nn.Sequential(conv_bn(c_in, c_out, stride), nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out))
        self.shortcut = nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm2d(c_out))   # match the shape when it changes

    def forward(self, x):
        return torch.relu(self.f(x) + self.shortcut(x))


class SmallResNet(nn.Module):
    """ResNet-20-like network for 32x32 inputs. The stem is a 3x3 conv with stride 1: the original project used
    ImageNet's 7x7 stride-2 stem, which throws away half the resolution of a 32x32 image immediately."""

    def __init__(self, widths=(32, 64, 128), blocks=3):
        super().__init__()
        layers, c = [conv_bn(3, widths[0])], widths[0]
        for k, w in enumerate(widths):
            for b in range(blocks):
                layers.append(BasicBlock(c, w, stride=2 if (b == 0 and k > 0) else 1))   # 32 -> 16 -> 8
                c = w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


MODELS = {"basic_cnn": BasicCNN, "vgg_bn": VGGBN, "depthwise_cnn": DepthwiseCNN, "small_resnet": SmallResNet}


def shape_trace(model: nn.Module, x: torch.Tensor) -> str:
    """Output shape and parameter count of every leaf layer, collected with forward hooks."""
    lines, hooks = [], []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(
                lambda m, i, o, name=name: lines.append(f"{name:<28} {type(m).__name__:<18} {str(tuple(o.shape)):<20} {sum(p.numel() for p in m.parameters()):>9,}")))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return "\n".join(lines)


# ---------------------------------------------------------------- data

class CifarSplit(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)   # (N, 32, 32, 3) uint8 -> (N, 3, 32, 32)
        self.labels = torch.from_numpy(labels).long()                 # int64 class indices, as CrossEntropyLoss expects
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def loaders(batch_size: int = 128, workers: int = 4, seed: int = 0):
    train_full = CIFAR10(DATA, train=True, download=True)
    test = CIFAR10(DATA, train=False, download=True)
    x, y = train_full.data, np.array(train_full.targets)
    idx = np.random.default_rng(seed).permutation(len(y))
    tr, va = idx[:45_000], idx[45_000:]                               # validation never seen in training
    norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    train_tf = v2.Compose([v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(), *norm])
    eval_tf = v2.Compose(norm)
    make = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=workers, persistent_workers=True)
    return (make(CifarSplit(x[tr], y[tr], train_tf), True), make(CifarSplit(x[va], y[va], eval_tf), False),
            make(CifarSplit(test.data, np.array(test.targets), eval_tf), False))


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(dim=1).cpu()                 # (N, 10) logits -> (N,) class index
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total


def train_model(name: str, epochs: int, device, data, lr: float = 3e-3):
    torch.manual_seed(0)
    model = MODELS[name]().to(device)
    train_dl, val_dl, test_dl = data
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)              # input: logits (N, 10); target: (N,) int64
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * len(train_dl))  # stepped per batch
    best, best_state, times = 0.0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        times.append(time.time() - t0)
        val = accuracy(model, val_dl, device)
        if val > best:                                                 # keep the weights of the best validation epoch
            best, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"{name:<14} epoch {epoch:2d}  loss {loss.item():.3f}  val acc {val:.4f}  {times[-1]:.1f}s", flush=True)
    model.load_state_dict(best_state)
    return {"model": name, "params": sum(p.numel() for p in model.parameters()), "s_per_epoch": float(np.mean(times)),
            "val_acc": best, "test_acc": accuracy(model, test_dl, device)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    names = args.models.split(",")
    (OUT / "shapes.txt").write_text("\n\n".join(f"== {n} (input (1, 3, 32, 32))\n{shape_trace(MODELS[n](), torch.zeros(1, 3, 32, 32))}" for n in names) + "\n")
    data = loaders()
    results = [train_model(n, args.epochs, device, data) for n in names]
    lines = ["# CIFAR-10 from scratch", "", f"{args.epochs} epochs each, AdamW + one-cycle schedule, crop and flip augmentation. "
             "Model selection on 5,000 validation images, test (10,000) scored once.", "",
             "| Model | Parameters | Seconds per epoch | Best val accuracy | Test accuracy |", "|---|---|---|---|---|"]
    lines += [f"| {r['model']} | {r['params']:,} | {r['s_per_epoch']:.1f} | {r['val_acc']:.3f} | {r['test_acc']:.3f} |" for r in results]
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
