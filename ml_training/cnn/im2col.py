"""
im2col / col2im — turn convolution into matrix multiplication.

This is the standard trick for making from-scratch convolution fast in
NumPy: instead of looping over every output pixel with nested for-loops,
overlapping input patches are rearranged into one big 2D matrix, multiplied
once by the reshaped filters, then rearranged back. The math computed is
identical to a naive nested-loop convolution — this is a vectorization
technique, not a different algorithm, and both functions here are plain
NumPy with no external libraries.
"""

import numpy as np


def get_output_shape(H, W, kh, kw, stride, pad):
    H_out = (H + 2 * pad - kh) // stride + 1
    W_out = (W + 2 * pad - kw) // stride + 1
    return H_out, W_out


def im2col(x, kh, kw, stride, pad):
    """
    x: (N, H, W, C)
    Returns:
        cols: (N * H_out * W_out, kh * kw * C)
        (H_out, W_out): output spatial size, needed to reshape the result later
    """
    N, H, W, C = x.shape
    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="constant")

    H_out, W_out = get_output_shape(H, W, kh, kw, stride, pad)

    cols = np.zeros((N, H_out, W_out, kh, kw, C), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * H_out
        for j in range(kw):
            j_max = j + stride * W_out
            cols[:, :, :, i, j, :] = x[:, i:i_max:stride, j:j_max:stride, :]

    return cols.reshape(N * H_out * W_out, kh * kw * C), (H_out, W_out)


def col2im(cols, x_shape, kh, kw, stride, pad):
    """
    Inverse of im2col — scatters gradient columns back into an image-shaped
    gradient array, accumulating overlapping contributions by addition.
    """
    N, H, W, C = x_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    H_out, W_out = get_output_shape(H, W, kh, kw, stride, pad)

    cols_reshaped = cols.reshape(N, H_out, W_out, kh, kw, C)
    dx_padded = np.zeros((N, H_padded, W_padded, C), dtype=cols.dtype)

    for i in range(kh):
        i_max = i + stride * H_out
        for j in range(kw):
            j_max = j + stride * W_out
            dx_padded[:, i:i_max:stride, j:j_max:stride, :] += cols_reshaped[:, :, :, i, j, :]

    if pad > 0:
        return dx_padded[:, pad:-pad, pad:-pad, :]
    return dx_padded
