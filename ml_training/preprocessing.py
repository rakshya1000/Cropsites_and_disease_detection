"""
Requirements 3, 4, 5: Load images from disk, resize to the chosen input size,
and normalize pixel values.

NOTE ON PILLOW: Decoding JPEG/PNG byte streams into raw pixel arrays is not
something plain NumPy can do — there is no pure-Python/NumPy image codec.
Pillow is used strictly as a file-reading utility (open the file, decode the
bytes, resize the grid) — comparable to using a CSV parser to read tabular
data. No Pillow function here performs any machine-learning computation.
Every numerical operation from this point onward (normalization, array
manipulation, everything in dataset_builder.py and beyond) is pure NumPy.
"""

import numpy as np
from PIL import Image
from config import IMG_SIZE


def load_and_preprocess_image(filepath, img_size=IMG_SIZE):
    """
    Loads one image, converts to RGB, resizes to img_size, and normalizes
    pixel values from [0, 255] integers to [0.0, 1.0] floats.

    Returns:
        np.ndarray of shape (height, width, 3), dtype float32
    """
    with Image.open(filepath) as img:
        img = img.convert("RGB")
        img = img.resize((img_size[1], img_size[0]), Image.BILINEAR)  # PIL takes (width, height)
        array = np.asarray(img, dtype=np.float32)

    # Normalize using NumPy: scale 0-255 integer range to 0.0-1.0 float range
    array = array / 255.0
    return array
