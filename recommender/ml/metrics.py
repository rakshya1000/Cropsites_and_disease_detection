"""
Evaluation metrics -- built from scratch, no ML libraries.

Implements the standard classification metrics needed for a project
report: accuracy, precision, recall, F1-score (per-class and averaged),
and a confusion matrix. All computed using only Python's built-in data
structures.
"""


def confusion_matrix(actual, predicted, labels=None):
    """Build a confusion matrix as a nested dictionary:
        matrix[true_label][predicted_label] = count

    `labels`: optional list of all class labels to include, even ones
    with zero occurrences (keeps row/column order consistent). If not
    given, labels are inferred from the data.
    """
    if labels is None:
        labels = sorted(set(actual) | set(predicted))

    matrix = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}

    for true_label, pred_label in zip(actual, predicted):
        matrix[true_label][pred_label] += 1

    return matrix, labels


def accuracy(actual, predicted):
    """Overall accuracy: fraction of predictions that match the actual label."""
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual)


def precision_recall_f1_per_class(matrix, labels):
    """Compute precision, recall, and F1-score for EACH class.

    For a given class C:
    - True Positives (TP)  = matrix[C][C] -- predicted C, actually C
    - False Positives (FP) = sum of matrix[other][C] for other != C
                             -- predicted C, but actually something else
    - False Negatives (FN) = sum of matrix[C][other] for other != C
                             -- actually C, but predicted something else

    Precision = TP / (TP + FP)   "of everything I predicted as C, how
                                   much was actually C?"
    Recall    = TP / (TP + FN)   "of everything that was actually C,
                                   how much did I correctly catch?"
    F1        = harmonic mean of precision and recall -- balances both;
                punishes cases where one is high but the other is low.
    """
    results = {}

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,  # total actual samples of this class
        }

    return results


def macro_average(per_class_metrics):
    """Macro-average: simple (unweighted) average across all classes.
    Treats every class equally regardless of how many samples it has."""
    n = len(per_class_metrics)
    avg_precision = sum(m["precision"] for m in per_class_metrics.values()) / n
    avg_recall = sum(m["recall"] for m in per_class_metrics.values()) / n
    avg_f1 = sum(m["f1"] for m in per_class_metrics.values()) / n
    return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}


def weighted_average(per_class_metrics):
    """Weighted average: weights each class's contribution by its
    support (how many actual samples it has). More representative
    when classes are imbalanced (though yours is balanced at 150/crop)."""
    total_support = sum(m["support"] for m in per_class_metrics.values())
    if total_support == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    avg_precision = sum(m["precision"] * m["support"] for m in per_class_metrics.values()) / total_support
    avg_recall = sum(m["recall"] * m["support"] for m in per_class_metrics.values()) / total_support
    avg_f1 = sum(m["f1"] * m["support"] for m in per_class_metrics.values()) / total_support
    return {"precision": avg_precision, "recall": avg_recall, "f1": avg_f1}


def print_confusion_matrix(matrix, labels):
    """Pretty-print the confusion matrix as a text table."""
    col_width = max(len(label) for label in labels) + 2
    col_width = max(col_width, 6)

    header = " " * (col_width) + "".join(f"{label[:6]:>{col_width}}" for label in labels)
    print(header)

    for true_label in labels:
        row = f"{true_label[:col_width-1]:<{col_width}}"
        for pred_label in labels:
            row += f"{matrix[true_label][pred_label]:>{col_width}}"
        print(row)


def print_classification_report(actual, predicted):
    """Print a full classification report: per-class precision/recall/F1,
    plus macro and weighted averages, similar in spirit to sklearn's
    classification_report -- but computed entirely from scratch."""
    matrix, labels = confusion_matrix(actual, predicted)
    per_class = precision_recall_f1_per_class(matrix, labels)
    macro = macro_average(per_class)
    weighted = weighted_average(per_class)
    overall_acc = accuracy(actual, predicted)

    name_width = max(len(label) for label in labels) + 2

    print(f"{'Class':<{name_width}}{'Precision':>10}{'Recall':>10}{'F1-score':>10}{'Support':>10}")
    for label in labels:
        m = per_class[label]
        print(f"{label:<{name_width}}{m['precision']:>10.2f}{m['recall']:>10.2f}{m['f1']:>10.2f}{m['support']:>10}")

    print()
    print(f"{'Accuracy':<{name_width}}{'':>10}{'':>10}{overall_acc:>10.2f}{len(actual):>10}")
    print(f"{'Macro avg':<{name_width}}{macro['precision']:>10.2f}{macro['recall']:>10.2f}{macro['f1']:>10.2f}{len(actual):>10}")
    print(f"{'Weighted avg':<{name_width}}{weighted['precision']:>10.2f}{weighted['recall']:>10.2f}{weighted['f1']:>10.2f}{len(actual):>10}")

    return {
        "per_class": per_class,
        "macro_avg": macro,
        "weighted_avg": weighted,
        "accuracy": overall_acc,
        "confusion_matrix": matrix,
        "labels": labels,
    }
