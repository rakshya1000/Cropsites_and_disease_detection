"""Flatten layer — converts (N, H, W, C) feature maps into (N, H*W*C) vectors."""


class Flatten:
    def __init__(self):
        self._input_shape = None

    def forward(self, x):
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._input_shape)

    def get_params_and_grads(self):
        return []
