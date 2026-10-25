# Fracture detection on X-rays, in PyTorch

Binary classification of musculoskeletal radiographs (fracture / no fracture) on the public
[FracAtlas](https://doi.org/10.6084/m9.figshare.22363012) dataset, rebuilt from a Keras course project
with a leakage-free protocol: data audit, grouped split, thresholds chosen on validation,
a test set scored once, confidence intervals, subgroup analysis and Grad-CAM.

## Results (test set: 603 images, 108 fractures, scored once)

| Model | Parameters | ROC-AUC [95% CI] | Average precision [95% CI] |
|---|---|---|---|
| ResNet-18, ImageNet weights, fine-tuned | 11.2 M | **0.882** [0.839, 0.922] | **0.751** [0.674, 0.817] |
| Small CNN trained from scratch | 0.39 M | 0.794 [0.744, 0.838] | 0.518 [0.421, 0.614] |
| Random classifier | | 0.5 | 0.179 |

ResNet-18 at a threshold fixed on validation for 90% recall: **97 of 108 fractures detected**
(recall 0.898 [0.827, 0.942]), specificity 0.584 [0.540, 0.626], precision 0.320.
A triage tool, not a diagnostic one. Full report: `runs/resnet18/report.md`.

### How optimistic was the original protocol?

`scripts/07_leaky_protocol.py` replays it (augmented copies written before a random split, early stopping and
threshold on the same "test") while the clean test set stays untouched. 81% of the leaky test images have a twin
in training.

| Same ResNet-18, evaluated on | ROC-AUC | AP | F1 | Recall |
|---|---|---|---|---|
| its leaky test (what the original protocol reports) | 0.988 | 0.966 | 0.912 | 0.918 |
| the clean held-out test | 0.874 | 0.757 | 0.691 | 0.611 |

The protocol inflated F1 by 0.22 and ROC-AUC by 0.11.

## What the data audit found

- 59 truncated JPEGs, all in the non-fractured class (excluded: a class-specific artifact is a shortcut).
- 12 groups of byte-identical files and 113 near-duplicate pairs (thumbnail correlation >= 0.99):
  186 images in 87 groups, each kept within a single split.
- 4 duplicate groups with contradictory labels (excluded).
- 2 images stored in both class folders: labels are read from `dataset.csv`, never from folder names.
- Possible shortcuts: 98% of images with orthopedic implants are fractures; fracture rates differ by image format.

## Protocol

1. `scripts/01_audit.py`: file integrity, duplicates, label conflicts, shortcut variables.
2. `scripts/02_split.py`: `StratifiedGroupKFold`, 70/15/15, groups = duplicate clusters.
3. `scripts/03_prepare.py`: grayscale, letterbox to a square, 256×256 PNG cache.
4. `scripts/04_train.py`: `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)`, AdamW with discriminative learning
   rates (backbone 1e-4, head 1e-3), 3 epochs with a frozen backbone, cosine schedule, early stopping on
   validation average precision.
5. `scripts/05_evaluate.py`: thresholds on validation (recall >= 0.90, max F1, 0.5), one scoring of the test set,
   Wilson and bootstrap intervals, subgroups (body part, implants, format), calibration.
6. `scripts/06_gradcam.py`: heatmaps for true positives, missed fractures and false alarms.
7. `scripts/07_leaky_protocol.py`: replays the original protocol to measure its optimism.

Part 2, `cifar10/train_cifar.py`: four CNNs trained from scratch on CIFAR-10 (basic, VGG-style with BatchNorm,
depthwise separable, small ResNet), with a held-out validation split.

| CIFAR-10 model | Parameters | s / epoch | Test accuracy |
|---|---|---|---|
| basic CNN (original architecture) | 122,570 | 2.8 | 0.805 |
| VGG-style + BatchNorm | 1,148,874 | 24.8 | 0.912 |
| depthwise separable | 140,106 | 19.4 | 0.864 |
| small ResNet (3x3 stem) | 1,084,586 | 40.1 | **0.913** |

## Run it

```bash
make setup    # Python 3.13 virtual environment with uv, pinned requirements
make data     # downloads FracAtlas v7 (338 MB, CC BY 4.0)
make all      # audit -> split -> prepare -> train -> evaluate -> Grad-CAM -> leaky-protocol experiment
make cifar    # part 2
make test     # unit tests: shapes, metrics, no split leakage, a loop that overfits one batch
```

Trained on an Apple M5 GPU (MPS): about 23 s per epoch for ResNet-18.

## Limitations

No patient identifiers (images of the same patient may fall into different splits), 108 test fractures
(wide intervals, small hip and shoulder subgroups), possible shortcuts (implants, casts) measured but not removed,
uncalibrated probabilities (because of `pos_weight`), single source, no external validation.

## Data

Abedeen, I., Rahman, M. A., Prottyasha, F. Z. et al. *FracAtlas: A Dataset for Fracture Classification,
Localization and Segmentation of Musculoskeletal Radiographs.* Scientific Data 10, 521 (2023).
Licensed under CC BY 4.0.
