"""Full evaluation: train Random Forest, then report all metrics from scratch."""

import csv
import random

from random_forest import RandomForest
from metrics import print_classification_report, print_confusion_matrix, confusion_matrix

random.seed(42)


def load_dataset(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append([
                float(r["N"]), float(r["P"]), float(r["K"]),
                float(r["temperature"]), float(r["humidity"]),
                float(r["ph"]), float(r["rainfall"]),
                r["label"],
            ])
    return rows


def train_test_split(rows, test_ratio=0.2):
    shuffled = rows[:]
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_point], shuffled[split_point:]


if __name__ == "__main__":
    data = load_dataset("agrosmart_crop_dataset.csv")
    train_data, test_data = train_test_split(data, test_ratio=0.2)

    test_features = [row[:-1] for row in test_data]
    test_labels = [row[-1] for row in test_data]

    print("Training Random Forest (15 trees)...")
    forest = RandomForest(n_trees=15, max_depth=10, min_samples_split=5)
    forest.fit(train_data)

    predictions = forest.predict(test_features)

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    report = print_classification_report(test_labels, predictions)

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX (rows = actual, columns = predicted)")
    print("=" * 70)
    matrix, labels = confusion_matrix(test_labels, predictions)
    print_confusion_matrix(matrix, labels)
