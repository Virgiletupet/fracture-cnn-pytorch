"""Step 2. Grouped, stratified 70/15/15 split, saved to data/splits.csv."""
from fracture.splits import make_splits

if __name__ == "__main__":
    make_splits(seed=42)
