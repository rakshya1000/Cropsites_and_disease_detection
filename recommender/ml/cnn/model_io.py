"""Only the import below changes to relative — logic is unchanged."""

import numpy as np
from .model import build_model


def save_model(model, filepath):
    save_dict = {}
    for idx, layer in enumerate(model.layers):
        params = layer.get_params_and_grads()
        if not params:
            continue
        save_dict[f"layer{idx}_W"] = params[0][0]
        save_dict[f"layer{idx}_b"] = params[1][0]
    np.savez_compressed(filepath, **save_dict)
    print(f"Model saved to {filepath}")


def load_model(filepath, num_classes, input_shape=(128, 128, 3)):
    model = build_model(num_classes=num_classes, input_shape=input_shape)
    data = np.load(filepath)

    for idx, layer in enumerate(model.layers):
        w_key, b_key = f"layer{idx}_W", f"layer{idx}_b"
        if w_key in data:
            layer.W[:] = data[w_key]
            layer.b[:] = data[b_key]

    return model
