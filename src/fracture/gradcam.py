"""Grad-CAM: which regions pushed the fracture logit up?

For the target layer's activations A (C, h, w) and the gradient of the logit with respect to them:
    weights_c = mean over (h, w) of d logit / d A_c          (how much each channel matters)
    cam       = ReLU( sum_c weights_c * A_c )                 (h, w), upsampled to the image
A heatmap on the bone is reassuring; a heatmap on a metal plate, a ruler or an image border is a shortcut.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import RUNS_DIR
from .data import FractureDataset, build_transforms, load_splits
from .evaluate import load_run


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.acts = output
        output.register_hook(lambda g: setattr(self, "grads", g))   # called during backward

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        """image (3, H, W) -> heatmap (H, W) in [0, 1]."""
        self.model.eval()
        x = image.unsqueeze(0).requires_grad_(True)       # (1, 3, H, W); gradients must flow even if layers are frozen
        logit = self.model(x)                              # (1,)
        self.model.zero_grad()
        logit.sum().backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)              # (1, C, 1, 1)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))     # (1, 1, h, w)
        cam = F.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def gradcam_grid(run: str, per_kind: int = 4) -> None:
    """Heatmaps for the most confident true positives, the missed fractures and the worst false alarms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, model, _ = load_run(run, device=torch.device("cpu"))      # CPU: simpler and deterministic for 12 images
    preds = pd.read_csv(RUNS_DIR / run / "predictions_test.csv")
    kinds = {
        "true positive": preds[(preds.label == 1) & (preds.pred == 1)].nlargest(per_kind, "prob"),
        "missed fracture (FN)": preds[(preds.label == 1) & (preds.pred == 0)].nsmallest(per_kind, "prob"),
        "false alarm (FP)": preds[(preds.label == 0) & (preds.pred == 1)].nlargest(per_kind, "prob"),
    }
    frame = load_splits().set_index("image_id")
    ds_tf = build_transforms(train=False)
    cam = GradCAM(model, model.cam_layer)
    fig, axes = plt.subplots(len(kinds), per_kind, figsize=(3 * per_kind, 3.3 * len(kinds)))
    for ax in axes.ravel():
        ax.axis("off")
    for i, (kind, rows) in enumerate(kinds.items()):
        for j, r in enumerate(rows.itertuples()):
            image, _, _ = FractureDataset(frame.loc[[r.image_id]].reset_index(), ds_tf)[0]
            heat = cam(image)
            gray = image[0].numpy() * 0.229 + 0.485                 # undo the normalization for display
            axes[i, j].imshow(gray, cmap="gray")
            axes[i, j].imshow(heat, cmap="jet", alpha=0.35)
            axes[i, j].set_title(f"{kind}\n{r.image_id}  p={r.prob:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(RUNS_DIR / run / "gradcam.png", dpi=110)
    plt.close(fig)
    print(f"saved {RUNS_DIR / run / 'gradcam.png'}")
