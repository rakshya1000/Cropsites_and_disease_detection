"""
Requirement 1: Scan the dataset folders automatically.

Walks DATASET_ROOT expecting the structure:
    dataset/<Crop>/<Class>/<image files>

and returns a flat list of (filepath, crop_name, class_name) tuples.
No image data is loaded here — this step only discovers what exists on disk.
"""

import os
from config import DATASET_ROOT, VALID_EXTENSIONS


def scan_dataset(root=DATASET_ROOT):
    """
    Returns:
        List[Tuple[str, str, str]]: (filepath, crop_name, class_name) for every
        valid image file found under root.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")

    samples = []
    crops = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )

    for crop in crops:
        crop_path = os.path.join(root, crop)
        classes = sorted(
            d for d in os.listdir(crop_path) if os.path.isdir(os.path.join(crop_path, d))
        )

        for class_name in classes:
            class_path = os.path.join(crop_path, class_name)
            for fname in sorted(os.listdir(class_path)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in VALID_EXTENSIONS:
                    samples.append((os.path.join(class_path, fname), crop, class_name))

    return samples


if __name__ == "__main__":
    found = scan_dataset()
    print(f"Found {len(found)} images across the dataset.")
    print("Example entries:")
    for entry in found[:5]:
        print(f"  {entry}")
