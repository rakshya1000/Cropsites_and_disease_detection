"""
Max pooling layer — downsamples spatial dimensions by taking the max over
non-overlapping windows (stride == pool_size, the standard configuration
for CNN classifiers). Fully vectorized via reshape/transpose rather than
per-window loops.
"""

import numpy as np


class MaxPool2D:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        assert self.stride == self.pool_size, (
            "This from-scratch MaxPool2D supports only non-overlapping "
            "pooling (stride must equal pool_size)."
        )
        self._cache = None

    def forward(self, x):
        N, H, W, C = x.shape
        p = self.pool_size
        H_out, W_out = H // p, W // p

        # Crop any remainder so H, W divide evenly (not an issue for
        # power-of-two input sizes like 128x128 with p=2).
        x_cropped = x[:, :H_out * p, :W_out * p, :]

        x_reshaped = x_cropped.reshape(N, H_out, p, W_out, p, C)
        x_reshaped = x_reshaped.transpose(0, 1, 3, 2, 4, 5)      # (N, H_out, W_out, p, p, C)
        windows = x_reshaped.reshape(N, H_out, W_out, p * p, C)

        out = windows.max(axis=3)
        mask = windows == out[:, :, :, None, :]                  # True at max location(s)

        self._cache = (x.shape, mask)
        return out

    def backward(self, dout):
        x_shape, mask = self._cache
        N, H, W, C = x_shape
        p = self.pool_size
        H_out, W_out = H // p, W // p

        dout_expanded = dout[:, :, :, None, :]                    # (N, H_out, W_out, 1, C)
        # Ties (multiple equal max values in a window) get the full
        # gradient passed to every tied position — a common simplification.
        dx_windows = mask.astype(dout.dtype) * dout_expanded

        dx_reshaped = dx_windows.reshape(N, H_out, W_out, p, p, C)
        dx_reshaped = dx_reshaped.transpose(0, 1, 3, 2, 4, 5)
        dx_cropped = dx_reshaped.reshape(N, H_out * p, W_out * p, C)

        dx = np.zeros(x_shape, dtype=dout.dtype)
        dx[:, :H_out * p, :W_out * p, :] = dx_cropped
        return dx

    def get_params_and_grads(self):
        return []
