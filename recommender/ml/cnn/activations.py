"""ReLU activation layer — no learnable parameters."""


class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask

    def get_params_and_grads(self):
        return []
