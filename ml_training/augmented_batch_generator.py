"""
Augmented batch generator — the training-time replacement for the plain
BatchGenerator from batch_generator.py.

KEY GUARANTEE: the original dataset is never modified. Each batch is copied
out of X (via fancy indexing, which already returns a new array) and, if
augment=True, written into a freshly allocated array via np.empty_like().
self.X itself is only ever read, never written to. Once a batch is yielded
and used for one training step, the augmented copy is discarded — next
epoch, the same original images get re-augmented from scratch with new
random parameters.
"""

import numpy as np
from augmentation import random_augment


class AugmentedBatchGenerator:
    def __init__(self, X, y, batch_size=32, shuffle=True, seed=None, augment=True):
        """
        X, y: the ORIGINAL arrays (e.g. X_train, y_train from dataset_builder).
              These are stored by reference but never written to.
        augment: True for training data, False for validation/test data
                 (val/test must see the real, unaltered images).
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.n = X.shape[0]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, self.n, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]

            # Fancy indexing already returns a fresh copy, not a view —
            # so X_batch_source is independent of self.X from this point on.
            X_batch_source = self.X[batch_idx]
            y_batch = self.y[batch_idx]

            if self.augment:
                X_batch = np.empty_like(X_batch_source)
                for i in range(X_batch_source.shape[0]):
                    X_batch[i] = random_augment(X_batch_source[i], self.rng)
            else:
                X_batch = X_batch_source

            yield X_batch, y_batch
