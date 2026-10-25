# Evaluation of run `simple_cnn`

Test set: 603 images, 108 fractures (prevalence 17.9%). Thresholds were fixed on the validation set; the test set was scored once.

## Threshold-free metrics (test, 95% bootstrap CI)

- ROC-AUC: 0.794 [0.744, 0.838] (validation 0.815)
- Average precision (PR-AUC): 0.518 [0.421, 0.614] (random model = 0.179)
- Brier score: 0.161 (probabilities are not calibrated: the loss weights fractures by pos_weight)

## Operating points (test, 95% Wilson CI)

| Threshold rule | t | TP | FN | FP | TN | Recall | Specificity | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
| recall>=0.90 (val) | 0.215 | 97 | 11 | 273 | 222 | 0.898 [0.827, 0.942] | 0.448 [0.405, 0.493] | 0.262 [0.220, 0.309] | 0.406 |
| max F1 (val) | 0.717 | 39 | 69 | 33 | 462 | 0.361 [0.277, 0.455] | 0.933 [0.908, 0.952] | 0.542 [0.427, 0.652] | 0.433 |
| 0.5 (default) | 0.500 | 68 | 40 | 106 | 389 | 0.630 [0.536, 0.715] | 0.786 [0.748, 0.820] | 0.391 [0.321, 0.465] | 0.482 |

Choosing the best-F1 threshold on the test set itself (the original project's protocol) would report F1 = 0.517 instead of 0.433: an optimistic bias of +0.083.

## Subgroups (test, threshold for recall >= 0.90)

| Subgroup | Images | Fractures | Recall [95% CI] | Specificity |
|---|---|---|---|---|
| hand | 206 | 64 | 0.97 [0.89, 0.99] | 0.11 |
| leg | 344 | 39 | 0.79 [0.64, 0.89] | 0.58 |
| hip | 68 | 14 | 0.86 [0.60, 0.96] | 0.57 |
| shoulder | 44 | 9 | 1.00 [0.70, 1.00] | 0.00 |
| mixed regions | 55 | 17 | 0.94 [0.73, 0.99] | 0.08 |
| with implants | 12 | 12 | 1.00 [0.76, 1.00] | n/a |
| without implants | 591 | 96 | 0.89 [0.81, 0.93] | 0.45 |
| portrait 373x454 | 383 | 75 | 0.93 [0.85, 0.97] | 0.27 |
| landscape 454x373 | 164 | 18 | 0.72 [0.49, 0.88] | 0.82 |
| large format | 56 | 15 | 0.93 [0.70, 0.99] | 0.49 |

Small subgroups have wide intervals: read them as warnings, not as conclusions.
