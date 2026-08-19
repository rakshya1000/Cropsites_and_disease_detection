"""
Training loop: forward -> loss -> backward -> optimizer step, repeated over
mini-batches for a number of epochs. Tracks train/val loss and accuracy
per epoch, and prints progress.

Run directly (python train.py) to train on prepared_dataset.npz and save
the trained model to disease_cnn_model.npz.
"""

import os
import sys

# cnn/ sits inside ml_training/ — add ml_training/ itself to sys.path so
# this file can import the phase-4 pipeline modules (batch_generator.py,
# augmented_batch_generator.py, preprocessing.py) that live one level up,
# regardless of whether you run this as `python train.py` from inside
# cnn/ or as `python cnn/train.py` from ml_training/.
ML_TRAINING_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML_TRAINING_ROOT)

import numpy as np
import time

from model import build_model
from losses import cross_entropy_loss
from optimizer import SGD
from augmented_batch_generator import AugmentedBatchGenerator
from batch_generator import BatchGenerator


def evaluate_split(model, X, y_onehot, batch_size=32):
    """Forward-only pass over a dataset split — no augmentation, no backward."""
    batches = BatchGenerator(X, y_onehot, batch_size=batch_size, shuffle=False)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for X_batch, y_batch in batches:
        logits = model.forward(X_batch)
        loss, _ = cross_entropy_loss(logits, y_batch)
        total_loss += loss * X_batch.shape[0]
        total_correct += np.sum(np.argmax(logits, axis=1) == np.argmax(y_batch, axis=1))
        total_samples += X_batch.shape[0]

    return total_loss / total_samples, total_correct / total_samples


def train(model, X_train, y_train, X_val, y_val,
          epochs=20, batch_size=32, learning_rate=0.01, momentum=0.9,
          augment=True, seed=42, print_every=20):
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for epoch in range(1, epochs + 1):
        train_batches = AugmentedBatchGenerator(
            X_train, y_train, batch_size=batch_size, shuffle=True,
            seed=seed + epoch, augment=augment,
        )

        epoch_loss, epoch_correct, epoch_samples = 0.0, 0, 0
        epoch_start = time.time()

        for batch_idx, (X_batch, y_batch) in enumerate(train_batches, start=1):
            batch_start = time.time()

            logits = model.forward(X_batch)
            loss, dlogits = cross_entropy_loss(logits, y_batch)
            model.backward(dlogits)
            optimizer.step(model.get_params_and_grads())

            epoch_loss += loss * X_batch.shape[0]
            epoch_correct += np.sum(np.argmax(logits, axis=1) == np.argmax(y_batch, axis=1))
            epoch_samples += X_batch.shape[0]

            if batch_idx == 1 or batch_idx % print_every == 0 or batch_idx == num_batches:
                batch_time = time.time() - batch_start
                print(f"  epoch {epoch}/{epochs}  batch {batch_idx}/{num_batches}  "
                      f"loss: {loss:.4f}  ({batch_time:.2f}s/batch)", flush=True)

        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples
        val_loss, val_acc = evaluate_split(model, X_val, y_val, batch_size=batch_size)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) - "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}", flush=True)

    return history


if __name__ == "__main__":
    from model_io import save_model

    print("Loading prepared_dataset.npz ...", flush=True)
    dataset_path = os.path.join(ML_TRAINING_ROOT, "prepared_dataset.npz")
    data = np.load(dataset_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(f"Loaded. Train: {X_train.shape[0]}  Val: {X_val.shape[0]}", flush=True)

    num_classes = y_train.shape[1]
    model = build_model(num_classes=num_classes, input_shape=X_train.shape[1:])
    print("Model built. Starting training...", flush=True)

    history = train(model, X_train, y_train, X_val, y_val,
                     epochs=20, batch_size=32, learning_rate=0.01)

    model_path = os.path.join(ML_TRAINING_ROOT, "disease_cnn_model.npz")
    save_model(model, model_path)