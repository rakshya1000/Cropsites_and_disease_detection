"""
Random Forest Classifier -- built from scratch, no ML libraries.

Place this file in: recommender/ml/random_forest.py
(same folder as loader.py, so the pickle can find it on import)

IMPORTANT: this version adds three sklearn-compatible members --
.predict(X), .predict_proba(X), and .classes_ -- so that your existing
loader.py (predict_one / predict_with_confidence) works UNCHANGED. Those
functions only ever call these three things generically; they don't know
or care that the model underneath is no longer a real sklearn object.
"""

import math
import random

from decision_tree import DecisionTree


def bootstrap_sample(rows):
    n = len(rows)
    return [random.choice(rows) for _ in range(n)]


def majority_vote(predictions_from_all_trees):
    counts = {}
    for pred in predictions_from_all_trees:
        counts[pred] = counts.get(pred, 0) + 1
    return max(counts, key=counts.get)


def _collect_labels(node):
    """Walk a tree and collect every leaf label (used to build classes_)."""
    if node.is_leaf():
        return {node.label}
    return _collect_labels(node.left) | _collect_labels(node.right)


class RandomForest:
    def __init__(self, n_trees=15, max_depth=10, min_samples_split=5,
                 n_features_to_try="sqrt"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features_to_try_setting = n_features_to_try
        self.trees = []

    def _resolve_n_features(self, total_features):
        if self.n_features_to_try_setting == "sqrt":
            return max(1, round(math.sqrt(total_features)))
        elif self.n_features_to_try_setting is None:
            return total_features
        else:
            return self.n_features_to_try_setting

    def fit(self, rows):
        total_features = len(rows[0]) - 1
        n_features_to_try = self._resolve_n_features(total_features)

        self.trees = []
        for i in range(self.n_trees):
            sample = bootstrap_sample(rows)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features_to_try=n_features_to_try,
            )
            tree.fit(sample)
            self.trees.append(tree)

    def predict(self, rows):
        """rows: list of feature lists (no label). Returns a list of
        predicted labels, one per row -- matches sklearn's .predict()."""
        all_tree_predictions = [tree.predict(rows) for tree in self.trees]
        n_rows = len(rows)
        final_predictions = []
        for row_index in range(n_rows):
            votes = [all_tree_predictions[tree_index][row_index]
                     for tree_index in range(self.n_trees)]
            final_predictions.append(majority_vote(votes))
        return final_predictions

    @property
    def classes_(self):
        """sklearn-style attribute: sorted list of all labels seen during
        training. Needed because loader.py does `classes = model.classes_`
        then indexes into it with `classes[i]`."""
        labels = set()
        for tree in self.trees:
            labels.update(_collect_labels(tree.root))
        return sorted(labels)

    def predict_proba(self, rows):
        """sklearn-style: for each row, return a list of probabilities in
        the same order as self.classes_ (here: each tree's vote share).
        Needed because loader.py does `probs = model.predict_proba(X)[0]`
        then `probs.argsort()`.

        NOTE: probs is returned as a plain Python list, but loader.py calls
        `.argsort()` on it, which is a NumPy array method, not a list
        method. See the argsort_compatible wrapper below -- we return a
        lightweight object that supports .argsort() without requiring
        NumPy, to keep this fully library-free.
        """
        classes = self.classes_
        class_index = {c: i for i, c in enumerate(classes)}

        all_tree_predictions = [tree.predict(rows) for tree in self.trees]
        n_rows = len(rows)
        result = []
        for row_index in range(n_rows):
            votes = [all_tree_predictions[t][row_index] for t in range(self.n_trees)]
            probs = [0.0] * len(classes)
            for v in votes:
                probs[class_index[v]] += 1
            probs = [p / self.n_trees for p in probs]
            result.append(_ArgsortableList(probs))
        return result

    def predict_with_confidence(self, row):
        """Convenience method: predict ONE sample, return (label, confidence)."""
        votes = [tree.predict([row])[0] for tree in self.trees]
        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        winner = max(counts, key=counts.get)
        confidence = counts[winner] / len(votes)
        return winner, confidence

    def predict_top_n(self, row, n=3):
        """Convenience method: top N predicted crops with vote-share confidence."""
        votes = [tree.predict([row])[0] for tree in self.trees]
        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        total = len(votes)
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [(label, count / total) for label, count in ranked[:n]]


class _ArgsortableList(list):
    """A plain list that also supports .argsort(), mimicking the one
    NumPy array method that loader.py relies on -- WITHOUT requiring
    NumPy as a dependency. argsort() returns indices that would sort
    the list; [::-1] on the result (as loader.py does) reverses it to
    get descending order, same as NumPy's behavior."""

    def argsort(self):
        indexed = list(enumerate(self))
        indexed.sort(key=lambda pair: pair[1])
        return [i for i, _ in indexed]
