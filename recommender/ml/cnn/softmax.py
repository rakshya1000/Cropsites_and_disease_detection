"""Numerically stable softmax function."""

import numpy as np


def softmax(logits):
    """logits: (N, num_classes) -> probs: (N, num_classes), each row sums to 1."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)
