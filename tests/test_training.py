"""A correct training loop must be able to overfit one small batch."""
import torch
from torch import nn

from fracture.models import SimpleCNN


def test_overfit_one_batch():
    torch.manual_seed(0)
    model = SimpleCNN(widths=(8, 16), dropout=0.0)
    x, y = torch.randn(8, 3, 32, 32), torch.tensor([1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()
    first = None
    for _ in range(60):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1 * first
