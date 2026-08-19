import os
import sys
import pickle
from functools import lru_cache

from django.conf import settings

# Ensure the ml/ folder itself is on sys.path, so pickle.load can find
# random_forest.py and decision_tree.py by their bare module names
# (e.g. "random_forest"), matching how they were saved when the model
# was trained/pickled from inside recommender/ml/.
ML_DIR = os.path.join(settings.BASE_DIR, "recommender", "ml")
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)


@lru_cache(maxsize=1)
def load_bundle():
    pkl_path = os.path.join(ML_DIR, "Crop_recommendation_RF.pkl")

    with open(pkl_path, 'rb') as f:
        bundle = pickle.load(f)

    assert "model" in bundle and "feature_cols" in bundle, "Invalid model bundle structure"
    return bundle


def predict_one(feature_dict):
    bundle = load_bundle()

    model = bundle["model"]
    order = bundle["feature_cols"]

    X = [[float(feature_dict[c]) for c in order]]

    pred = model.predict(X)[0]
    return pred


def predict_with_confidence(feature_dict):
    """
    Returns (predicted_label, confidence, top3) using the model's real
    predict_proba output instead of a hardcoded heuristic.
    """
    bundle = load_bundle()
    model = bundle["model"]
    order = bundle["feature_cols"]

    X = [[float(feature_dict[c]) for c in order]]

    probs = model.predict_proba(X)[0]
    classes = model.classes_

    ranked_idx = probs.argsort()[::-1]

    predicted_label = classes[ranked_idx[0]]
    confidence = round(float(probs[ranked_idx[0]]), 2)

    top3 = [
        {"name": classes[i], "confidence": round(float(probs[i]), 2)}
        for i in ranked_idx[:3]
    ]

    return predicted_label, confidence, top3