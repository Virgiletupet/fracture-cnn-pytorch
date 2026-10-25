"""Metrics, decision thresholds and confidence intervals.

Convention: a prediction is positive (fracture) when the probability p >= threshold.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, int]:
    pred = p >= threshold
    y = y.astype(bool)
    return {"tp": int((pred & y).sum()), "fp": int((pred & ~y).sum()), "fn": int((~pred & y).sum()), "tn": int((~pred & ~y).sum())}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n. Better than the normal approximation
    when n is small or k/n is close to 0 or 1 (it never leaves [0, 1])."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, float(center - half)), min(1.0, float(center + half)))   # clip float rounding


def rates(c: dict[str, int]) -> dict[str, float | tuple[float, float]]:
    """Recall (sensitivity), specificity, precision (PPV), NPV, F1, accuracy, with Wilson intervals."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

    def ratio(k: int, n: int) -> float:
        return k / n if n else float("nan")

    recall, precision = ratio(tp, tp + fn), ratio(tp, tp + fp)
    return {
        "recall": recall, "recall_ci": wilson(tp, tp + fn),
        "specificity": ratio(tn, tn + fp), "specificity_ci": wilson(tn, tn + fp),
        "precision": precision, "precision_ci": wilson(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
    }


def threshold_for_recall(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest threshold whose recall is still >= target (the most specific operating point that
    meets the clinical requirement). Recall only changes at the scores of positive examples."""
    pos = np.sort(p[y == 1])[::-1]                      # positive scores, highest first
    k = max(1, int(np.ceil(target * len(pos))))         # positives that must be caught
    return float(pos[k - 1])                            # threshold = score of the k-th positive


def threshold_max_f1(y: np.ndarray, p: np.ndarray) -> float:
    candidates = np.unique(p)
    f1 = [rates(confusion(y, p, t))["f1"] for t in candidates]
    return float(candidates[int(np.argmax(f1))])


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float],
                 n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap: resample the test images with replacement and recompute the metric."""
    rng = np.random.default_rng(seed)
    n, values = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():   # a resample with a single class: AUC undefined
            continue
        values.append(metric(y[idx], p[idx]))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def ranking_metrics(y: np.ndarray, p: np.ndarray, with_ci: bool = True) -> dict:
    """Threshold-free metrics. ROC-AUC: probability that a random fracture scores above a random normal image.
    Average precision (PR-AUC): more informative when positives are rare; a random model scores the prevalence."""
    out = {"roc_auc": float(roc_auc_score(y, p)), "avg_precision": float(average_precision_score(y, p)),
           "prevalence": float(y.mean()), "brier": float(np.mean((p - y) ** 2))}
    if with_ci:
        out["roc_auc_ci"] = bootstrap_ci(y, p, roc_auc_score)
        out["avg_precision_ci"] = bootstrap_ci(y, p, average_precision_score)
    return out
