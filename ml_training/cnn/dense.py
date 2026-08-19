"""Fully connected (dense) layer."""

import numpy as np


class Dense:
    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_features)  # He initialization
        self.W = rng.normal(0, scale, size=(in_features, out_features)).astype(np.float32)
        self.b = np.zeros((out_features,), dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._x = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def get_params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]
