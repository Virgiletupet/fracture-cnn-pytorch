# Replaying the original protocol

Augmented copies written before a random split: 81% of the leaky test images have their original or augmented twin in the training set. Early stopping and the threshold (t = 0.439, best F1) use that same leaky test.

| Evaluated on | Images | ROC-AUC | AP | F1 | Recall | Precision |
|---|---|---|---|---|---|---|
| leaky test (as the original protocol reports) | 1365 | 0.988 | 0.966 | 0.912 | 0.918 | 0.907 |
| clean held-out test, same model and threshold | 603 | 0.874 | 0.757 | 0.691 | 0.611 | 0.795 |

The gap between the two rows is the optimism created by the protocol, not by the model.
