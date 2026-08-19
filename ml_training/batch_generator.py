"""
Requirement 9: Generate mini-batches for training.
"""

import numpy as np


class BatchGenerator:
    """
    Iterable that yields (X_batch, y_batch) mini-batches.
    Reshuffles the index order at the start of every __iter__ call (i.e.
    every epoch) when shuffle=True, so each epoch sees a different batch order.
    """

    def __init__(self, X, y, batch_size=32, shuffle=True, seed=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = X.shape[0]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        """Number of batches per epoch (last batch may be smaller than batch_size)."""
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, self.n, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]
