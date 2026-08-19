"""Evaluation on the held-out test set: overall accuracy + per-class accuracy."""

import os
import sys

# cnn/ sits inside ml_training/ — add ml_training/ itself to sys.path so
# this file can import batch_generator.py, which lives one level up.
ML_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML_TRAINING_ROOT)

import json
import numpy as np
from batch_generator import BatchGenerator
from losses import cross_entropy_loss


def evaluate_test_set(model, X_test, y_test, label_map_path=None, batch_size=32):
    if label_map_path is None:
        label_map_path = os.path.join(ML_TRAINING_ROOT, "label_map.json")
    with open(label_map_path) as f:
        label_map = json.load(f)  # {"0": "Maize__Healthy", ...}

    batches = BatchGenerator(X_test, y_test, batch_size=batch_size, shuffle=False)

    num_classes = y_test.shape[1]
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    total_loss, total_samples = 0.0, 0

    for X_batch, y_batch in batches:
        logits = model.forward(X_batch)
        loss, _ = cross_entropy_loss(logits, y_batch)
        preds = np.argmax(logits, axis=1)
        true = np.argmax(y_batch, axis=1)

        total_loss += loss * X_batch.shape[0]
        total_samples += X_batch.shape[0]

        for c in range(num_classes):
            mask = true == c
            class_total[c] += mask.sum()
            class_correct[c] += (preds[mask] == c).sum()

    overall_acc = class_correct.sum() / class_total.sum()
    print(f"\nTest loss: {total_loss / total_samples:.4f}")
    print(f"Test accuracy: {overall_acc:.4f}\n")

    print(f"{'Class':<35}{'Accuracy':>10}{'Support':>10}")
    print("-" * 55)
    for c in range(num_classes):
        class_name = label_map.get(str(c), f"class_{c}")
        acc = class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0
        print(f"{class_name:<35}{acc:>10.4f}{int(class_total[c]):>10}")

    return overall_acc


if __name__ == "__main__":
    from model_io import load_model

    data = np.load(os.path.join(ML_TRAINING_ROOT, "prepared_dataset.npz"))
    X_test, y_test = data["X_test"], data["y_test"]

    model_path = os.path.join(ML_TRAINING_ROOT, "disease_cnn_model.npz")
    model = load_model(model_path, num_classes=y_test.shape[1], input_shape=X_test.shape[1:])
    evaluate_test_set(model, X_test, y_test)
