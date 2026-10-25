"""Evaluation of a trained run, in the order a clinician-facing report needs:

1. thresholds are chosen on the VALIDATION set (recall target, or best F1 for comparison);
2. the TEST set is scored once, at those frozen thresholds, with 95% confidence intervals;
3. subgroups (body part, implants, image format) reveal where the model fails or cheats;
4. plots: ROC, precision-recall, confusion matrix, calibration.

Writes runs/<name>/report.md, metrics.json, predictions.csv and figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .config import RUNS_DIR, TrainConfig
from .data import load_splits, make_loaders
from .metrics import confusion, ranking_metrics, rates, threshold_for_recall, threshold_max_f1
from .models import build_model
from .train import predict
from .utils import get_device, load_json, save_json

RECALL_TARGET = 0.90   # triage: missing a fracture costs more than a false alarm read by a radiologist


def load_run(run: str, device=None):
    out = RUNS_DIR / run
    cfg = TrainConfig(**load_json(out / "config.json"))
    device = device or get_device()
    model = build_model(cfg.model, pretrained=False, dropout=cfg.dropout)   # weights come from the checkpoint
    state = torch.load(out / "best.pt", map_location="cpu")
    model.load_state_dict(state["model"])
    return cfg, model.to(device).eval(), device


def fmt(r: dict, key: str) -> str:
    lo, hi = r[f"{key}_ci"]
    return f"{r[key]:.3f} [{lo:.3f}, {hi:.3f}]"


def evaluate_run(run: str, frame: pd.DataFrame | None = None) -> dict:
    out = RUNS_DIR / run
    cfg, model, device = load_run(run)
    frame = load_splits() if frame is None else frame
    loaders = make_loaders(cfg, frame)
    pv, yv, idv, _ = predict(model, loaders["val"], device)
    pt, yt, idt, _ = predict(model, loaders["test"], device)

    # 1. Thresholds, on validation only
    thresholds = {
        f"recall>={RECALL_TARGET:.2f} (val)": threshold_for_recall(yv, pv, RECALL_TARGET),
        "max F1 (val)": threshold_max_f1(yv, pv),
        "0.5 (default)": 0.5,
    }
    # 2. Test, scored once per frozen threshold
    results = {"run": run, "n_test": int(len(yt)), "n_test_fractures": int(yt.sum()),
               "val": ranking_metrics(yv, pv, with_ci=False), "test": ranking_metrics(yt, pt), "operating_points": {}}
    for name, t in thresholds.items():
        c = confusion(yt, pt, t)
        results["operating_points"][name] = {"threshold": t, **c, **rates(c)}
    # What the original protocol did: pick the best-F1 threshold on the test set itself (optimistic, for comparison only)
    t_cheat = threshold_max_f1(yt, pt)
    results["optimistic_f1_threshold_on_test"] = {"threshold": t_cheat, "f1": rates(confusion(yt, pt, t_cheat))["f1"]}

    # 3. Subgroups, at the recall-target threshold
    t_main = thresholds[f"recall>={RECALL_TARGET:.2f} (val)"]
    test = frame[frame["split"] == "test"].set_index("image_id").loc[idt].reset_index()
    test["prob"], test["pred"] = pt, (pt >= t_main).astype(int)
    groups = {"hand": test["hand"] == 1, "leg": test["leg"] == 1, "hip": test["hip"] == 1, "shoulder": test["shoulder"] == 1,
              "mixed regions": test["mixed"] == 1, "with implants": test["hardware"] == 1, "without implants": test["hardware"] == 0,
              "portrait 373x454": test["shape"] == "portrait", "landscape 454x373": test["shape"] == "landscape",
              "large format": test["shape"] == "large"}
    sub = []
    for name, mask in groups.items():
        g = test[mask]
        c = confusion(g["label"].to_numpy(), g["prob"].to_numpy(), t_main)
        r = rates(c)
        sub.append({"subgroup": name, "images": len(g), "fractures": int(g["label"].sum()), **c,
                    "recall": r["recall"], "recall_ci": r["recall_ci"], "specificity": r["specificity"]})
    results["subgroups"] = sub

    # 4. Files
    test[["image_id", "label", "prob", "pred", "hand", "leg", "hip", "shoulder", "hardware", "shape"]].to_csv(out / "predictions_test.csv", index=False)
    pd.DataFrame({"image_id": idv, "label": yv, "prob": pv}).to_csv(out / "predictions_val.csv", index=False)
    save_json(results, out / "metrics.json")
    plot_evaluation(yt, pt, results, out)
    write_report(results, out / "report.md")
    print((out / "report.md").read_text())
    return results


def write_report(r: dict, path) -> None:
    t = r["test"]
    lines = [f"# Evaluation of run `{r['run']}`", "",
             f"Test set: {r['n_test']} images, {r['n_test_fractures']} fractures (prevalence {t['prevalence']:.1%}). "
             "Thresholds were fixed on the validation set; the test set was scored once.", "",
             "## Threshold-free metrics (test, 95% bootstrap CI)", "",
             f"- ROC-AUC: {fmt(t, 'roc_auc')} (validation {r['val']['roc_auc']:.3f})",
             f"- Average precision (PR-AUC): {fmt(t, 'avg_precision')} (random model = {t['prevalence']:.3f})",
             f"- Brier score: {t['brier']:.3f} (probabilities are not calibrated: the loss weights fractures by pos_weight)", "",
             "## Operating points (test, 95% Wilson CI)", "",
             "| Threshold rule | t | TP | FN | FP | TN | Recall | Specificity | Precision | F1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, o in r["operating_points"].items():
        lines.append(f"| {name} | {o['threshold']:.3f} | {o['tp']} | {o['fn']} | {o['fp']} | {o['tn']} | {fmt(o, 'recall')} | "
                     f"{fmt(o, 'specificity')} | {fmt(o, 'precision')} | {o['f1']:.3f} |")
    honest = r["operating_points"]["max F1 (val)"]["f1"]
    lines += ["", f"Choosing the best-F1 threshold on the test set itself (the original project's protocol) would report "
              f"F1 = {r['optimistic_f1_threshold_on_test']['f1']:.3f} instead of {honest:.3f}: an optimistic bias of "
              f"{r['optimistic_f1_threshold_on_test']['f1'] - honest:+.3f}.", "",
              f"## Subgroups (test, threshold for recall >= {RECALL_TARGET:.2f})", "",
              "| Subgroup | Images | Fractures | Recall [95% CI] | Specificity |", "|---|---|---|---|---|"]
    for s in r["subgroups"]:
        rec = f"{s['recall']:.2f} [{s['recall_ci'][0]:.2f}, {s['recall_ci'][1]:.2f}]" if s["fractures"] else "n/a"
        spec = f"{s['specificity']:.2f}" if s["images"] > s["fractures"] else "n/a"
        lines.append(f"| {s['subgroup']} | {s['images']} | {s['fractures']} | {rec} | {spec} |")
    lines += ["", "Small subgroups have wide intervals: read them as warnings, not as conclusions."]
    path.write_text("\n".join(lines) + "\n")


def plot_evaluation(y: np.ndarray, p: np.ndarray, r: dict, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr)
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set(title=f"ROC (AUC {r['test']['roc_auc']:.3f})", xlabel="1 - specificity", ylabel="recall")
    prec, rec, _ = precision_recall_curve(y, p)
    ax[1].plot(rec, prec)
    ax[1].axhline(y.mean(), ls="--", color="grey", label="random")
    ax[1].set(title=f"Precision-recall (AP {r['test']['avg_precision']:.3f})", xlabel="recall", ylabel="precision")
    for name, o in r["operating_points"].items():
        ax[0].scatter(1 - o["specificity"], o["recall"], zorder=3, label=name)
        ax[1].scatter(o["recall"], o["precision"], zorder=3)
    ax[0].legend(fontsize=8, loc="lower right")
    main = next(iter(r["operating_points"].values()))
    cm = np.array([[main["tn"], main["fp"]], [main["fn"], main["tp"]]])
    ax[2].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[2].text(j, i, str(v), ha="center", va="center", fontsize=14)
    ax[2].set(title=f"Confusion matrix, t = {main['threshold']:.3f}", xticks=[0, 1], yticks=[0, 1],
              xticklabels=["pred normal", "pred fracture"], yticklabels=["normal", "fracture"])
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax[3].plot(mean_pred, frac_pos, "o-")
    ax[3].plot([0, 1], [0, 1], "--", color="grey")
    ax[3].set(title="Calibration (test)", xlabel="predicted probability", ylabel="observed fracture rate")
    fig.tight_layout()
    fig.savefig(out / "evaluation.png", dpi=120)
    plt.close(fig)
