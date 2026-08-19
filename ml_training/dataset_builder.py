"""
Requirements 6, 7, 8: Encode labels (one-hot), shuffle the dataset, and split
into training / validation / testing sets. All pure NumPy.
"""

import numpy as np
from data_loader import scan_dataset
from label_encoder import LabelEncoder
from preprocessing import load_and_preprocess_image
from config import IMG_SIZE, RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT


def build_dataset(root=None, verbose=True):
    """
    Scans the dataset, loads every image into memory, and returns:
        X: np.ndarray of shape (N, H, W, 3), float32, values in [0, 1]
        y: np.ndarray of shape (N,), int64, integer class labels
        encoder: LabelEncoder instance (needed to decode predictions later)
    """
    samples = scan_dataset(root) if root else scan_dataset()
    encoder = LabelEncoder(samples)

    n = len(samples)
    h, w = IMG_SIZE
    X = np.zeros((n, h, w, 3), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)

    for i, (filepath, crop, class_name) in enumerate(samples):
        X[i] = load_and_preprocess_image(filepath)
        y[i] = encoder.encode(crop, class_name)
        if verbose and (i + 1) % 1000 == 0:
            print(f"Loaded {i + 1}/{n} images...")

    return X, y, encoder


def one_hot_encode(y, num_classes):
    """Converts integer labels of shape (N,) to one-hot vectors of shape (N, num_classes)."""
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot


def shuffle_dataset(X, y, seed=RANDOM_SEED):
    """Shuffles X and y together (same permutation applied to both) using NumPy's RNG."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(X.shape[0])
    return X[indices], y[indices]


def split_dataset(X, y, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT, test_split=TEST_SPLIT):
    """
    Splits already-shuffled X, y into train/val/test sets by simple slicing.
    Because shuffle_dataset() is called before this, the slices are random
    samples, not the original on-disk ordering.
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

    n = X.shape[0]
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
