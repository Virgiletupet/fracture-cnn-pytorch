"""Train / validation / test split, made once and saved, before any model sees the data.

- Grouped: all copies of the same radiograph (duplicate group) land in the same split, otherwise the test set
  would contain images the model has already seen.
- Stratified: each split keeps the ~17.6% fracture rate.
- 70 / 15 / 15: validation is used for every decision (early stopping, threshold), test only once at the end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import INTERIM_DIR, SPLITS_CSV

N_FOLDS = 20                     # 20 folds of 5%: 3 for test, 3 for validation, 14 for training
TEST_FOLDS, VAL_FOLDS = (0, 1, 2), (3, 4, 5)


def make_splits(seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(INTERIM_DIR / "audit.csv")
    df = df[~df["exclude"]].reset_index(drop=True)

    fold = np.empty(len(df), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(sgkf.split(df, y=df["label"], groups=df["group"])):
        fold[idx] = k
    df["split"] = np.where(np.isin(fold, TEST_FOLDS), "test", np.where(np.isin(fold, VAL_FOLDS), "val", "train"))

    # Sanity checks: no group and no identical file shared between splits.
    assert df.groupby("group")["split"].nunique().max() == 1, "a duplicate group spans two splits"
    assert df.groupby("md5")["split"].nunique().max() == 1, "identical files in two splits"

    df.to_csv(SPLITS_CSV, index=False)
    summary = df.groupby("split")["label"].agg(images="size", fractured="sum", rate="mean").loc[["train", "val", "test"]]
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    return df
