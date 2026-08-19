"""
Same logic as before — only the import line changes to relative,
since this file lives inside recommender/ml/cnn/ as a package now.
"""

import numpy as np
from .im2col import im2col, col2im


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = self.kw = kernel_size
        self.stride = stride
        self.pad = padding

        rng = np.random.default_rng(seed)
        fan_in = in_channels * self.kh * self.kw
        scale = np.sqrt(2.0 / fan_in)
        self.W = rng.normal(0, scale, size=(self.kh, self.kw, in_channels, out_channels)).astype(np.float32)
        self.b = np.zeros((out_channels,), dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._cache = None

    def forward(self, x):
        cols, (H_out, W_out) = im2col(x, self.kh, self.kw, self.stride, self.pad)
        W_col = self.W.reshape(-1, self.out_channels)
        out = cols @ W_col + self.b
        out = out.reshape(x.shape[0], H_out, W_out, self.out_channels)
        self._cache = (x.shape, cols)
        return out

    def backward(self, dout):
        x_shape, cols = self._cache
        C_out = dout.shape[-1]
        dout_flat = dout.reshape(-1, C_out)
        self.dW = (cols.T @ dout_flat).reshape(self.W.shape)
        self.db = dout_flat.sum(axis=0)
        W_col = self.W.reshape(-1, self.out_channels)
        dcols = dout_flat @ W_col.T
        return col2im(dcols, x_shape, self.kh, self.kw, self.stride, self.pad)

    def get_params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]
