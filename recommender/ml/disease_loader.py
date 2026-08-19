"""
Loads the trained CNN once (cached) and predicts on uploaded images.
Paths below match your ACTUAL layout: disease_cnn_model.npz and
label_map.json sit directly in recommender/ml/ (no model_weights/
subfolder), and the model code is in recommender/ml/cnn/.
"""

import os
import json
from functools import lru_cache

import numpy as np
from PIL import Image

from .cnn.model_io import load_model

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "disease_cnn_model.npz")
LABEL_MAP_PATH = os.path.join(APP_DIR, "label_map.json")

# Must match IMG_SIZE in ml_training/config.py — what the model was trained on.
IMG_SIZE = (128, 128)


@lru_cache(maxsize=1)
def get_label_map():
    with open(LABEL_MAP_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_model():
    label_map = get_label_map()
    num_classes = len(label_map)
    return load_model(MODEL_PATH, num_classes=num_classes, input_shape=(*IMG_SIZE, 3))


def preprocess_uploaded_image(image_file):
    """image_file: a Django UploadedFile, e.g. request.FILES['image']."""
    img = Image.open(image_file).convert("RGB")
    img = img.resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
    array = np.asarray(img, dtype=np.float32) / 255.0
    return array[np.newaxis, ...]


def predict_disease(image_file):
    """
    Returns:
        {"crop": "Maize", "disease": "Common Rust",
         "is_healthy": False, "confidence": 0.9421}
        confidence is a 0.0-1.0 fraction — multiply by 100 exactly once,
        in the view, when displaying it.
    """
    model = get_model()
    label_map = get_label_map()

    x = preprocess_uploaded_image(image_file)
    probs = model.predict_proba(x)[0]

    predicted_index = int(np.argmax(probs))
    confidence = float(probs[predicted_index])

    label = label_map[str(predicted_index)]  # e.g. "Maize__Common_Rust"
    crop, disease_raw = label.split("__", 1)
    disease = disease_raw.replace("_", " ")

    return {
        "crop": crop,
        "disease": disease,
        "is_healthy": disease_raw.lower() == "healthy",
        "confidence": confidence,
    }
