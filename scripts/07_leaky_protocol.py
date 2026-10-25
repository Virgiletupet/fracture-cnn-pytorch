"""Experiment: how optimistic was the original Keras protocol?

The original project (1) wrote augmented copies to disk BEFORE splitting, (2) split images at random,
ignoring duplicates, (3) used the test set for early stopping and (4) picked the best-F1 threshold on it.

Here we replay that protocol on the train + validation pool only, while the clean test set stays untouched.
The same model is then scored twice:
- on its own "leaky test" (what the original protocol would have reported);
- on the clean held-out test set (what it is really worth).
"""
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision.io import ImageReadMode, decode_image, write_png
from torchvision.transforms import v2

from fracture.config import PROCESSED_DIR, ROOT, RUNS_DIR, TrainConfig
from fracture.data import load_splits, make_loaders
from fracture.evaluate import load_run
from fracture.metrics import confusion, ranking_metrics, rates, threshold_max_f1
from fracture.train import fit, predict

LEAKY_DIR = ROOT / "data" / "processed_leaky"
RUN = "leaky_protocol"


def build_leaky_frame(seed: int = 0) -> pd.DataFrame:
    torch.manual_seed(seed)
    LEAKY_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_splits()
    pool = frame[frame["split"] != "test"].copy()
    # Same augmentation family as the original ImageDataGenerator: rotation 15, shifts 10 %, flips
    augment = v2.Compose([v2.RandomAffine(degrees=15, translate=(0.1, 0.1)), v2.RandomHorizontalFlip()])
    copies = []
    for image_id in pool["image_id"]:
        stem = image_id.removesuffix(".jpg")
        src, link = PROCESSED_DIR / f"{stem}.png", LEAKY_DIR / f"{stem}.png"
        if not link.exists():
            os.symlink(src, link)
        out = LEAKY_DIR / f"{stem}_aug.png"
        if not out.exists():
            write_png(augment(decode_image(str(src), mode=ImageReadMode.GRAY)), str(out))
        copies.append(f"{stem}_aug.jpg")
    aug = pool.assign(image_id=copies)
    both = pd.concat([pool, aug], ignore_index=True)        # augmentation BEFORE the split: the flaw
    tr, te = train_test_split(both.index, test_size=0.2, stratify=both["label"], random_state=seed)  # ignores groups
    both.loc[tr, "split"], both.loc[te, "split"] = "train", "val"   # "val" = the leaky test, also used for early stopping
    # The clean test set, for the honest score of the same model
    clean = frame[frame["split"] == "test"].copy()
    for image_id in clean["image_id"]:
        link = LEAKY_DIR / image_id.replace(".jpg", ".png")
        if not link.exists():
            os.symlink(PROCESSED_DIR / link.name, link)
    return pd.concat([both, clean], ignore_index=True)


def main():
    frame = build_leaky_frame()
    leaky_test = frame[frame["split"] == "val"]
    twins = leaky_test["image_id"].str.removesuffix("_aug.jpg").str.removesuffix(".jpg")
    train_stems = set(frame.loc[frame["split"] == "train", "image_id"].str.removesuffix("_aug.jpg").str.removesuffix(".jpg"))
    print(f"leaky test: {len(leaky_test)} images, {twins.isin(train_stems).mean():.0%} have their twin in training")

    cfg = TrainConfig(model="resnet18", run_name=RUN, epochs=30, patience=7)
    if not (RUNS_DIR / RUN / "best.pt").exists():
        fit(cfg, frame, image_dir=LEAKY_DIR)
    cfg, model, device = load_run(RUN)
    loaders = make_loaders(cfg, frame, image_dir=LEAKY_DIR)
    pl, yl, _, _ = predict(model, loaders["val"], device)
    pc, yc, _, _ = predict(model, loaders["test"], device)

    t = threshold_max_f1(yl, pl)                             # tuned on the leaky test, like the original
    rows = []
    for name, y, p in [("leaky test (as the original protocol reports)", yl, pl), ("clean held-out test, same model and threshold", yc, pc)]:
        r, m = rates(confusion(y, p, t)), ranking_metrics(y, p, with_ci=False)
        rows.append(f"| {name} | {len(y)} | {m['roc_auc']:.3f} | {m['avg_precision']:.3f} | {r['f1']:.3f} | {r['recall']:.3f} | {r['precision']:.3f} |")
    report = "\n".join([
        "# Replaying the original protocol", "",
        f"Augmented copies written before a random split: {twins.isin(train_stems).mean():.0%} of the leaky test images "
        "have their original or augmented twin in the training set. Early stopping and the threshold "
        f"(t = {t:.3f}, best F1) use that same leaky test.", "",
        "| Evaluated on | Images | ROC-AUC | AP | F1 | Recall | Precision |", "|---|---|---|---|---|---|---|", *rows, "",
        "The gap between the two rows is the optimism created by the protocol, not by the model.",
    ])
    (RUNS_DIR / RUN / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
