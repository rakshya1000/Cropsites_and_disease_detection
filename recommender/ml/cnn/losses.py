"""
Cross-entropy loss, combined with softmax.

Combining softmax + cross-entropy into one step is standard practice: the
gradient of softmax-cross-entropy with respect to the raw logits simplifies
to (probs - y_true), which is simpler and more numerically stable than
differentiating softmax and cross-entropy separately and chaining them
through the backward pass.
"""

import numpy as np
from softmax import softmax


def cross_entropy_loss(logits, y_true_onehot):
    """
    logits: (N, num_classes) raw scores from the final Dense layer
    y_true_onehot: (N, num_classes) one-hot encoded true labels

    Returns:
        loss: scalar, mean cross-entropy over the batch
        dlogits: (N, num_classes) gradient of loss w.r.t. logits
    """
    N = logits.shape[0]
    probs = softmax(logits)

    eps = 1e-9  # avoids log(0)
    loss = -np.sum(y_true_onehot * np.log(probs + eps)) / N

    dlogits = (probs - y_true_onehot) / N
    return loss, dlogits
