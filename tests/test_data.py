import pandas as pd
import pytest
import torch

from fracture.config import SPLITS_CSV
from fracture.data import build_transforms


def test_transforms_output_shape_dtype_and_range():
    image = torch.randint(0, 256, (1, 256, 256), dtype=torch.uint8)   # what decode_image returns
    for train in (True, False):
        x = build_transforms(train)(image)
        assert x.shape == (3, 224, 224) and x.dtype == torch.float32
        assert -3 < x.min() and x.max() < 3                               # normalized, not 0..255


@pytest.mark.skipif(not SPLITS_CSV.exists(), reason="run scripts/02_split.py first")
def test_no_leakage_between_splits():
    df = pd.read_csv(SPLITS_CSV)
    assert df.groupby("group")["split"].nunique().max() == 1            # duplicate groups never cross splits
    assert df.groupby("md5")["split"].nunique().max() == 1
    assert not df["truncated"].any() and not df["label_conflict"].any()
    rates = df.groupby("split")["label"].mean()
    assert rates.max() - rates.min() < 0.02                              # stratified
