"""
Assembles all layers into one CNN model, and provides forward/backward/
predict methods that chain through every layer in sequence.

Architecture (tuned for 128x128x3 input, 29 output classes):
    Conv2D(3 -> 16, 3x3, pad=1) -> ReLU -> MaxPool(2x2)      : 128x128 -> 64x64
    Conv2D(16 -> 32, 3x3, pad=1) -> ReLU -> MaxPool(2x2)     : 64x64  -> 32x32
    Conv2D(32 -> 64, 3x3, pad=1) -> ReLU -> MaxPool(2x2)     : 32x32  -> 16x16
    Flatten                                                  : 16*16*64 = 16384
    Dense(16384 -> 128) -> ReLU
    Dense(128 -> num_classes)                                : raw logits (no softmax here)
"""

import numpy as np
from conv2d import Conv2D
from activations import ReLU
from pooling import MaxPool2D
from flatten import Flatten
from dense import Dense
from softmax import softmax


def build_model(num_classes, input_shape=(128, 128, 3), seed=42):
    H, W, C = input_shape
    layers = [
        Conv2D(C, 16, kernel_size=3, stride=1, padding=1, seed=seed),
        ReLU(),
        MaxPool2D(pool_size=2),

        Conv2D(16, 32, kernel_size=3, stride=1, padding=1, seed=seed),
        ReLU(),
        MaxPool2D(pool_size=2),

        Conv2D(32, 64, kernel_size=3, stride=1, padding=1, seed=seed),
        ReLU(),
        MaxPool2D(pool_size=2),

        Flatten(),
        Dense((H // 8) * (W // 8) * 64, 128, seed=seed),
        ReLU(),
        Dense(128, num_classes, seed=seed),
    ]
    return CNNModel(layers)


class CNNModel:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        """Runs x through every layer in order; returns raw logits (no softmax)."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dlogits):
        """Runs the gradient backward through every layer, in reverse order."""
        grad = dlogits
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params_and_grads(self):
        """Collects (param, grad) pairs from every layer that has parameters."""
        params_and_grads = []
        for layer in self.layers:
            params_and_grads.extend(layer.get_params_and_grads())
        return params_and_grads

    def predict_proba(self, x):
        """Forward pass + softmax, for inference (no backward pass needed)."""
        logits = self.forward(x)
        return softmax(logits)

    def predict(self, x):
        """Returns the predicted integer class label for each sample in x."""
        return np.argmax(self.predict_proba(x), axis=1)
