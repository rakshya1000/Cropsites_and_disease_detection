"""
Runs the complete data loading pipeline end-to-end:
    scan -> load+resize+normalize -> shuffle -> one-hot encode -> split -> save

Run this once to produce prepared_dataset.npz and label_map.json.
The CNN training script (next phase) will load those two files directly
instead of re-scanning/re-decoding images every run.
"""

import json
import numpy as np

from dataset_builder import build_dataset, one_hot_encode, shuffle_dataset, split_dataset
from batch_generator import BatchGenerator
from config import BATCH_SIZE


def main():
    print("Scanning and loading dataset (this can take a while for ~14,800 images)...")
    X, y, encoder = build_dataset()
    print(f"Loaded {X.shape[0]} images across {encoder.num_classes} classes.")

    X, y = shuffle_dataset(X, y)
    y_onehot = one_hot_encode(y, encoder.num_classes)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y_onehot)

    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    train_batches = BatchGenerator(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Batch size: {BATCH_SIZE}  ->  {len(train_batches)} batches per epoch")

    # Save the prepared arrays so later phases (CNN training) don't need to
    # re-scan and re-decode every image from disk on every run.
    np.savez_compressed(
        "prepared_dataset.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )
    print("Saved prepared arrays to prepared_dataset.npz")

    with open("label_map.json", "w") as f:
        json.dump(encoder.as_dict(), f, indent=2)
    print("Saved label map to label_map.json")


if __name__ == "__main__":
    main()
