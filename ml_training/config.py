"""
Configuration for the AgroSmart disease detection data pipeline.
Edit DATASET_ROOT to point at your actual dataset folder.
"""

import os

# Path to the root of your finalized dataset (7 crop folders inside).
# Change this to the actual absolute/relative path on your machine.
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# Input size the CNN will expect: (height, width)
IMG_SIZE = (128, 128)

# File extensions considered valid images (matches your verification script)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Train / validation / test split ratios — must sum to 1.0
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Fixed seed so shuffling/splitting is reproducible run to run
RANDOM_SEED = 42

# Default mini-batch size for training
BATCH_SIZE = 32
