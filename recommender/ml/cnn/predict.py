"""Single-image prediction — loads one image from disk and runs it through the trained model."""

import os
import sys

# cnn/ sits inside ml_training/ — add ml_training/ itself to sys.path so
# this file can import preprocessing.py, which lives one level up.
ML_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML_TRAINING_ROOT)

import json
import numpy as np

from preprocessing import load_and_preprocess_image
from model_io import load_model


def predict_image(filepath, model_path=None,
                   label_map_path=None, num_classes=29, input_shape=(128, 128, 3)):
    if model_path is None:
        model_path = os.path.join(ML_TRAINING_ROOT, "disease_cnn_model.npz")
    if label_map_path is None:
        label_map_path = os.path.join(ML_TRAINING_ROOT, "label_map.json")

    with open(label_map_path) as f:
        label_map = json.load(f)

    model = load_model(model_path, num_classes=num_classes, input_shape=input_shape)

    image = load_and_preprocess_image(filepath, img_size=input_shape[:2])
    x = image[np.newaxis, ...]  # add batch dimension: (1, H, W, C)

    probs = model.predict_proba(x)[0]
    predicted_index = int(np.argmax(probs))
    predicted_label = label_map[str(predicted_index)]
    confidence = float(probs[predicted_index])

    return predicted_label, confidence


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    label, confidence = predict_image(sys.argv[1])
    print(f"Predicted: {label}  (confidence: {confidence:.2%})")
