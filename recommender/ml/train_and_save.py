"""
Train the from-scratch Random Forest and save it as a pickle bundle,
in the exact format loader.py expects: {"model": ..., "feature_cols": [...]}

Run this ONCE (from inside recommender/ml/) to regenerate
Crop_recommendation_RF.pkl with the from-scratch model instead of sklearn's.

    cd recommender/ml
    python3 train_and_save.py
"""

import csv
import pickle
import random

from random_forest import RandomForest

random.seed(42)

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_dataset(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append([
                float(r["N"]), float(r["P"]), float(r["K"]),
                float(r["temperature"]), float(r["humidity"]),
                float(r["ph"]), float(r["rainfall"]),
                r["label"].strip().lower(),  # lowercase to match get_crop_info() lookup keys
            ])
    return rows


def main():
    data = load_dataset("agrosmart_crop_dataset.csv")
    print(f"Loaded {len(data)} training rows")

    forest = RandomForest(n_trees=15, max_depth=10, min_samples_split=5)
    forest.fit(data)
    print(f"Trained Random Forest with {forest.n_trees} trees")
    print(f"Classes learned: {forest.classes_}")

    bundle = {
        "model": forest,
        "feature_cols": FEATURE_COLS,
    }

    with open("Crop_recommendation_RF.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("Saved Crop_recommendation_RF.pkl (from-scratch model)")


if __name__ == "__main__":
    main()
