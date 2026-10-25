"""Shapes and parameter counts of every model, checked on a dummy batch."""
import torch

from fracture.models import SimpleCNN, TransferResNet


def test_simple_cnn_shapes():
    model = SimpleCNN()
    x = torch.randn(4, 3, 224, 224)                       # (N, C, H, W) float32
    assert model.features(x).shape == (4, 256, 14, 14)    # 4 max-pools: 224 / 2**4 = 14
    out = model(x)
    assert out.shape == (4,) and out.dtype == torch.float32   # one logit per image


def test_resnet_shapes_and_params():
    model = TransferResNet(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    assert model.backbone(x).shape == (2, 512)            # fc replaced by Identity
    assert model(x).shape == (2,)
    n = sum(p.numel() for p in model.parameters())
    assert n == 11_176_512 + 513                          # ResNet-18 without its 1000-class fc, plus Linear(512, 1)


def test_freezing_backbone_leaves_only_the_head_trainable():
    model = TransferResNet(pretrained=False)
    model.set_backbone_trainable(False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["head.1.weight", "head.1.bias"]
