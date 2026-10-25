"""Training: the explicit PyTorch loop that Keras' model.fit() hides.

One run = one folder runs/<name>/ with config.json, history.json, best.pt and curves.png.
"""
from __future__ import annotations

import time
from datetime import datetime

import torch
from torch import nn

from .config import RUNS_DIR, TrainConfig
from .data import load_splits, make_loaders
from .metrics import ranking_metrics
from .models import build_model
from .utils import count_params, get_device, save_json, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm: float = 0.0) -> float:
    """One pass over the training set. Returns the mean loss per image."""
    model.train()                                   # dropout on, BatchNorm uses batch statistics
    total, seen = 0.0, 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)   # (N, 3, 224, 224) float32, (N,) float32
        optimizer.zero_grad()                       # gradients accumulate in .grad: reset them first
        logits = model(images)                      # forward pass, (N,), builds the autograd graph
        loss = criterion(logits, labels)            # scalar: mean weighted binary cross-entropy
        loss.backward()                             # backpropagation: fills p.grad for every trainable p
        if max_grad_norm:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)   # after backward, before step
        optimizer.step()                            # update: AdamW uses p.grad and its moment estimates
        total += loss.item() * labels.size(0)       # .item(): Python float, does not keep the graph alive
        seen += labels.size(0)
    return total / seen


@torch.no_grad()                                    # no graph: less memory, faster
def predict(model, loader, device, criterion=None):
    """Probabilities for a whole split, in loader order. Returns (probs, labels, ids, mean loss)."""
    model.eval()                                    # dropout off, BatchNorm uses its running statistics
    probs, ys, ids, total = [], [], [], 0.0
    for images, labels, batch_ids in loader:
        logits = model(images.to(device))
        if criterion is not None:
            total += criterion(logits, labels.to(device)).item() * labels.size(0)
        probs.append(torch.sigmoid(logits).cpu())   # logit -> probability, only at prediction time
        ys.append(labels)
        ids.extend(batch_ids)
    y = torch.cat(ys).numpy()
    return torch.cat(probs).numpy(), y, ids, total / len(y)


def fit(cfg: TrainConfig, frame=None, image_dir=None) -> str:
    set_seed(cfg.seed)
    device = get_device()
    frame = load_splits() if frame is None else frame
    loaders = make_loaders(cfg, frame, image_dir) if image_dir else make_loaders(cfg, frame)
    run = cfg.run_name or f"{cfg.model}-{datetime.now():%Y%m%d-%H%M}"
    out = RUNS_DIR / run
    out.mkdir(parents=True, exist_ok=True)
    save_json(cfg.to_dict(), out / "config.json")

    model = build_model(cfg.model, cfg.pretrained, cfg.dropout).to(device)

    # Class imbalance, handled once, in the loss: each fracture counts n_neg / n_pos times (about 4.7).
    train_labels = frame.loc[frame["split"] == "train", "label"]
    pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum() if cfg.pos_weight else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))

    optimizer = torch.optim.AdamW(model.param_groups(cfg.lr_head, cfg.lr_backbone), weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)   # stepped once per epoch

    print(f"run {run} on {device}: {count_params(model):,} parameters, pos_weight {pos_weight:.2f}, "
          f"train {len(loaders['train'].dataset)} / val {len(loaders['val'].dataset)} images")

    if cfg.freeze_epochs:
        model.set_backbone_trainable(False)         # warm-up: train the new head on frozen features
    history, best_ap, bad_epochs = [], -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        if epoch == cfg.freeze_epochs + 1:
            model.set_backbone_trainable(True)      # fine-tune the whole network
        t0 = time.time()
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device, cfg.max_grad_norm)
        p, y, _, val_loss = predict(model, loaders["val"], device, criterion)
        val = ranking_metrics(y, p, with_ci=False)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val["roc_auc"],
               "val_ap": val["avg_precision"], "lr": optimizer.param_groups[-1]["lr"], "seconds": time.time() - t0}
        history.append(row)
        improved = val["avg_precision"] > best_ap
        print(f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}  AUC {val['roc_auc']:.3f}  "
              f"AP {val['avg_precision']:.3f}  {row['seconds']:.0f}s{'  *' if improved else ''}", flush=True)
        # Early stopping on validation average precision: the metric that matters with rare positives.
        if improved:
            best_ap, bad_epochs = val["avg_precision"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_ap": best_ap}, out / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"early stopping: no improvement for {cfg.patience} epochs")
                break
    save_json(history, out / "history.json")
    plot_history(history, out / "curves.png")
    return run


def plot_history(history: list[dict], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set(title="Weighted BCE loss", xlabel="epoch")
    ax[0].legend()
    ax[1].plot(ep, [h["val_auc"] for h in history], label="ROC-AUC")
    ax[1].plot(ep, [h["val_ap"] for h in history], label="average precision")
    ax[1].set(title="Validation", xlabel="epoch", ylim=(0, 1))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
