"""
Decision Tree Classifier -- built from scratch, no ML libraries.

Place this file in: recommender/ml/decision_tree.py
(same folder as loader.py, so the pickle can find it on import)
"""

import random


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self):
        return self.label is not None


def gini_impurity(labels):
    if not labels:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    n = len(labels)
    impurity = 1.0
    for count in counts.values():
        proportion = count / n
        impurity -= proportion ** 2
    return impurity


def weighted_gini(left_labels, right_labels):
    n_left = len(left_labels)
    n_right = len(right_labels)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    weighted = (n_left / n_total) * gini_impurity(left_labels)
    weighted += (n_right / n_total) * gini_impurity(right_labels)
    return weighted


def split_dataset(rows, feature_index, threshold):
    left = [row for row in rows if row[feature_index] <= threshold]
    right = [row for row in rows if row[feature_index] > threshold]
    return left, right


def candidate_thresholds(rows, feature_index, max_thresholds=10):
    values = sorted(set(row[feature_index] for row in rows))
    if len(values) <= max_thresholds:
        return values
    step = len(values) / max_thresholds
    return [values[int(i * step)] for i in range(max_thresholds)]


def best_split(rows, feature_indices, n_features_to_try=None):
    labels = [row[-1] for row in rows]
    current_impurity = gini_impurity(labels)

    best_gain = 0.0
    best_feature = None
    best_threshold = None

    features_to_check = feature_indices
    if n_features_to_try is not None and n_features_to_try < len(feature_indices):
        features_to_check = random.sample(feature_indices, n_features_to_try)

    for feature_index in features_to_check:
        thresholds = candidate_thresholds(rows, feature_index)
        for threshold in thresholds:
            left, right = split_dataset(rows, feature_index, threshold)
            if not left or not right:
                continue

            left_labels = [row[-1] for row in left]
            right_labels = [row[-1] for row in right]
            split_impurity = weighted_gini(left_labels, right_labels)

            gain = current_impurity - split_impurity
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


def majority_label(rows):
    labels = [row[-1] for row in rows]
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


def build_tree(rows, feature_indices, max_depth=10, min_samples_split=5,
               depth=0, n_features_to_try=None):
    labels = [row[-1] for row in rows]

    if depth >= max_depth or len(rows) < min_samples_split or gini_impurity(labels) == 0.0:
        return TreeNode(label=majority_label(rows))

    feature, threshold, gain = best_split(rows, feature_indices, n_features_to_try)

    if feature is None or gain <= 0:
        return TreeNode(label=majority_label(rows))

    left_rows, right_rows = split_dataset(rows, feature, threshold)

    left_child = build_tree(left_rows, feature_indices, max_depth,
                             min_samples_split, depth + 1, n_features_to_try)
    right_child = build_tree(right_rows, feature_indices, max_depth,
                              min_samples_split, depth + 1, n_features_to_try)

    return TreeNode(feature=feature, threshold=threshold,
                     left=left_child, right=right_child)


def predict_one(node, row):
    if node.is_leaf():
        return node.label
    if row[node.feature] <= node.threshold:
        return predict_one(node.left, row)
    else:
        return predict_one(node.right, row)


def predict(tree, rows):
    return [predict_one(tree, row) for row in rows]


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=5, n_features_to_try=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_to_try = n_features_to_try
        self.root = None
        self.feature_indices = None

    def fit(self, rows):
        n_features = len(rows[0]) - 1
        self.feature_indices = list(range(n_features))
        self.root = build_tree(
            rows, self.feature_indices,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features_to_try=self.n_features_to_try,
        )

    def predict(self, rows):
        return predict(self.root, rows)
