"""SGD optimizer with momentum — updates every layer's parameters in-place."""

import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self._velocity = {}  # id(param) -> velocity array

    def step(self, params_and_grads):
        """
        params_and_grads: list of (param_array, grad_array) tuples gathered
        from every layer in the model. Updates each param IN PLACE (param
        += v), so a layer's own self.W / self.b — which reference the same
        array object — reflect the update automatically. No reassignment,
        so identity-keyed momentum tracking (via id(param)) stays valid
        across every training step.
        """
        for param, grad in params_and_grads:
            key = id(param)
            if key not in self._velocity:
                self._velocity[key] = np.zeros_like(param)

            v = self._velocity[key]
            v[:] = self.momentum * v - self.lr * grad
            param += v
