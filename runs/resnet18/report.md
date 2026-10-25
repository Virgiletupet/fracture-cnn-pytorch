# Evaluation of run `resnet18`

Test set: 603 images, 108 fractures (prevalence 17.9%). Thresholds were fixed on the validation set; the test set was scored once.

## Threshold-free metrics (test, 95% bootstrap CI)

- ROC-AUC: 0.882 [0.839, 0.922] (validation 0.884)
- Average precision (PR-AUC): 0.751 [0.674, 0.817] (random model = 0.179)
- Brier score: 0.089 (probabilities are not calibrated: the loss weights fractures by pos_weight)

## Operating points (test, 95% Wilson CI)

| Threshold rule | t | TP | FN | FP | TN | Recall | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
| recall>=0.90 (val) | 0.010 | 97 | 11 | 206 | 289 | 0.898 [0.827, 0.942] | 0.584 [0.540, 0.626] | 0.320 [0.270, 0.375] | 0.472 |
| max F1 (val) | 0.378 | 71 | 37 | 32 | 463 | 0.657 [0.564, 0.740] | 0.935 [0.910, 0.954] | 0.689 [0.594, 0.771] | 0.673 |
| 0.5 (default) | 0.500 | 67 | 41 | 28 | 467 | 0.620 [0.526, 0.706] | 0.943 [0.919, 0.961] | 0.705 [0.607, 0.788] | 0.660 |

Choosing the best-F1 threshold on the test set itself (the original project's protocol) would report F1 = 0.681 instead of 0.673: an optimistic bias of +0.008.

## Subgroups (test, threshold for recall >= 0.90)

| Subgroup | Images | Fractures | Recall [95% CI] | Specificity |
|---|---|---|---|---|
| hand | 206 | 64 | 0.94 [0.85, 0.98] | 0.41 |
| leg | 344 | 39 | 0.85 [0.70, 0.93] | 0.61 |
| hip | 68 | 14 | 0.79 [0.52, 0.92] | 0.85 |
| shoulder | 44 | 9 | 0.78 [0.45, 0.94] | 0.80 |
| mixed regions | 55 | 17 | 0.76 [0.53, 0.90] | 0.68 |
| with implants | 12 | 12 | 1.00 [0.76, 1.00] | n/a |
| without implants | 591 | 96 | 0.89 [0.81, 0.93] | 0.58 |
| portrait 373x454 | 383 | 75 | 0.92 [0.84, 0.96] | 0.49 |
| landscape 454x373 | 164 | 18 | 0.83 [0.61, 0.94] | 0.81 |
| large format | 56 | 15 | 0.87 [0.62, 0.96] | 0.51 |

Small subgroups have wide intervals: read them as warnings, not as conclusions.
