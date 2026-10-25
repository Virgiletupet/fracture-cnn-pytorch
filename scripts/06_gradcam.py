"""Step 6. Grad-CAM heatmaps for true positives, missed fractures and false alarms."""
import sys

from fracture.gradcam import gradcam_grid

if __name__ == "__main__":
    gradcam_grid(sys.argv[1] if len(sys.argv) > 1 else "resnet18")
