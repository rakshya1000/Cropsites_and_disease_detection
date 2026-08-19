"""
Model saving/loading. Saves every parameterized layer's weight/bias arrays
into a single .npz file, keyed by layer index. To load, the exact same
architecture is rebuilt via build_model() first, then each layer's saved
W/b arrays are copied into it in place.
"""

import numpy as np
from model import build_model


def save_model(model, filepath):
    """
    Saves every Conv2D/Dense layer's W and b into filepath.
    Layers with no parameters (ReLU, MaxPool2D, Flatten) are skipped.
    """
    save_dict = {}
    for idx, layer in enumerate(model.layers):
        params = layer.get_params_and_grads()
        if not params:
            continue
        # Conv2D and Dense both return [(W, dW), (b, db)] in that order
        save_dict[f"layer{idx}_W"] = params[0][0]
        save_dict[f"layer{idx}_b"] = params[1][0]

    np.savez_compressed(filepath, **save_dict)
    print(f"Model saved to {filepath}")


def load_model(filepath, num_classes, input_shape=(128, 128, 3)):
    """
    Rebuilds the architecture via build_model() (must match what was used
    during training) and loads each layer's saved W/b arrays into it.
    """
    model = build_model(num_classes=num_classes, input_shape=input_shape)
    data = np.load(filepath)

    for idx, layer in enumerate(model.layers):
        w_key, b_key = f"layer{idx}_W", f"layer{idx}_b"
        if w_key in data:
            layer.W[:] = data[w_key]
            layer.b[:] = data[b_key]

    print(f"Model loaded from {filepath}")
    return model
