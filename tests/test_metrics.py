import numpy as np

from fracture.metrics import confusion, rates, threshold_for_recall, threshold_max_f1, wilson


def test_confusion_and_rates():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7, 0.1, 0.1, 0.3, 0.2])
    c = confusion(y, p, 0.5)
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}
    r = rates(c)
    assert np.isclose(r["recall"], 2 / 3) and np.isclose(r["specificity"], 4 / 5) and np.isclose(r["precision"], 2 / 3)


def test_wilson_interval():
    lo, hi = wilson(5, 10)
    assert np.isclose(lo, 0.2366, atol=1e-4) and np.isclose(hi, 0.7634, atol=1e-4)
    assert wilson(0, 20)[0] == 0.0                        # never below 0, unlike the normal approximation


def test_threshold_for_recall_is_the_highest_that_meets_the_target():
    y = np.array([1, 1, 1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.4, 0.3, 0.5, 0.1])
    t = threshold_for_recall(y, p, 0.75)                  # 3 of 4 positives must be caught
    assert t == 0.4
    assert rates(confusion(y, p, t))["recall"] >= 0.75
    assert rates(confusion(y, p, t + 1e-6))["recall"] < 0.75


def test_threshold_max_f1():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.7, 0.6, 0.1])
    assert threshold_max_f1(y, p) == 0.7
