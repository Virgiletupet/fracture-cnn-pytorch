"""Step 3. Cache every kept image as a 256x256 grayscale PNG (letterboxed)."""
from fracture.data import prepare_images

if __name__ == "__main__":
    prepare_images(size=256)
