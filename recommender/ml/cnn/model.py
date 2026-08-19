"""
Same architecture as before — only imports below are now relative,
since this file lives inside recommender/ml/cnn/ as a package.
"""

import numpy as np
from .conv2d import Conv2D
from .activations import ReLU
from .pooling import MaxPool2D
from .flatten import Flatten
from .dense import Dense
from .softmax import softmax


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
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dlogits):
        grad = dlogits
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_params_and_grads(self):
        params_and_grads = []
        for layer in self.layers:
            params_and_grads.extend(layer.get_params_and_grads())
        return params_and_grads

    def predict_proba(self, x):
        logits = self.forward(x)
        return softmax(logits)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)
