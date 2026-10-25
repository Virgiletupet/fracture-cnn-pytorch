"""Step 5. Thresholds on validation, one scoring of the test set, intervals, subgroups, plots."""
import sys

from fracture.evaluate import evaluate_run

if __name__ == "__main__":
    evaluate_run(sys.argv[1] if len(sys.argv) > 1 else "resnet18")
