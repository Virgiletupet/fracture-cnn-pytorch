"""Two models, both returning one logit per image: shape (N,), no sigmoid inside.

- SimpleCNN: a small network trained from scratch, the baseline.
- TransferResNet: ResNet-18 pretrained on ImageNet, with a new head. It replaces the VGG16 of the
  original project: 11.2 M parameters instead of 138 M, BatchNorm, and residual connections.

The sigmoid is applied in the loss (BCEWithLogitsLoss, numerically stable) and at prediction time.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class SimpleCNN(nn.Module):
    """4 blocks conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2, then global average pooling and a linear head.

    Shapes for a batch of 32:
        input            (32, 3, 224, 224)
        block 1  (32)    (32, 32, 112, 112)
        block 2  (64)    (32, 64, 56, 56)
        block 3  (128)   (32, 128, 28, 28)
        block 4  (256)   (32, 256, 14, 14)
        avg pool         (32, 256, 1, 1) -> flatten (32, 256)
        linear           (32, 1) -> squeeze (32,)
    """

    def __init__(self, in_channels: int = 3, widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for w in widths:
            # padding=1 keeps H and W with a 3x3 kernel; bias=False because BatchNorm has its own shift
            layers += [nn.Conv2d(c, w, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)          # any input size -> (N, C, 1, 1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))).squeeze(1)

    def param_groups(self, lr_head: float, lr_backbone: float) -> list[dict]:
        # Trained from scratch: one learning rate for everything.
        return [{"params": self.parameters(), "lr": lr_head}]

    def set_backbone_trainable(self, trainable: bool) -> None:
        pass  # nothing pretrained to protect

    @property
    def cam_layer(self) -> nn.Module:
        return self.features[-2]     # last ReLU, before the final pooling (Grad-CAM target)


class TransferResNet(nn.Module):
    """ResNet-18 backbone (ImageNet weights) + Dropout + Linear(512, 1).

        input                       (32, 3, 224, 224)
        stem conv 7x7 s2 + maxpool  (32, 64, 56, 56)
        layer1 .. layer4            (32, 512, 7, 7)
        avgpool + flatten           (32, 512)
        head                        (32, 1) -> squeeze (32,)
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.fc.in_features      # 512
        self.backbone.fc = nn.Identity()                # the backbone now returns (N, 512) features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)

    def param_groups(self, lr_head: float, lr_backbone: float) -> list[dict]:
        # Discriminative learning rates: the new head learns fast, pretrained layers move slowly.
        return [{"params": self.backbone.parameters(), "lr": lr_backbone}, {"params": self.head.parameters(), "lr": lr_head}]

    def set_backbone_trainable(self, trainable: bool) -> None:
        # requires_grad=False: autograd computes no gradient, so the optimizer skips these weights.
        # (In train mode, BatchNorm running statistics still adapt to X-rays; that is intended.)
        for p in self.backbone.parameters():
            p.requires_grad_(trainable)

    @property
    def cam_layer(self) -> nn.Module:
        return self.backbone.layer4


def build_model(name: str, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    if name == "resnet18":
        return TransferResNet(pretrained=pretrained, dropout=dropout)
    if name == "simple_cnn":
        return SimpleCNN(dropout=max(dropout, 0.3))
    raise ValueError(f"unknown model {name!r}")
