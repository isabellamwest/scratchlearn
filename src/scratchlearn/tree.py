"""Decision tree classification with gini or entropy splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import check_array, check_is_fitted, check_X_y
from .base import BaseEstimator, ClassifierMixin


def entropy(p: ArrayLike) -> float:
    """Shannon entropy -sum(p * log2(p)) of a probability vector, in bits."""
    arr = np.asarray(p, dtype=np.float64)
    arr = arr[arr > 0]
    return float(-(arr * np.log2(arr)).sum())


def gini(p: ArrayLike) -> float:
    """Gini impurity 1 - sum(p^2) of a probability vector."""
    arr = np.asarray(p, dtype=np.float64)
    return float(1.0 - (arr**2).sum())


def information_gain(parent: ArrayLike, left: ArrayLike, right: ArrayLike) -> float:
    """Entropy reduction from splitting the `parent` labels into `left` and `right`."""

    def h(labels: np.ndarray) -> float:
        counts = np.unique(labels, return_counts=True)[1]
        return entropy(counts / len(labels))

    parent_arr = np.asarray(parent)
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    n = len(parent_arr)
    return (
        h(parent_arr) - (len(left_arr) / n) * h(left_arr) - (len(right_arr) / n) * h(right_arr)
    )


@dataclass
class _Node:
    """A single tree node; leaves keep feature=None and class counts in value."""

    feature: int | None = None
    threshold: float = 0.0
    left: _Node | None = None
    right: _Node | None = None
    value: np.ndarray | None = None


class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """Greedy binary CART tree.

    Each node takes the split with the largest impurity decrease, scanning
    midpoints between consecutive distinct feature values. With no depth
    limit the tree memorises the training set, which is exactly the
    overfitting behaviour explored in examples/03_trees_vs_knn.ipynb.
    """

    def __init__(
        self, criterion: str = "gini", max_depth: int | None = None, min_samples_split: int = 2
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X: ArrayLike, y: ArrayLike) -> DecisionTreeClassifier:
        if self.criterion not in ("gini", "entropy"):
            raise ValueError(f"unknown criterion {self.criterion!r}")
        X_arr, y_arr = check_X_y(X, y)
        self.classes_, encoded = np.unique(y_arr, return_inverse=True)
        self.n_features_ = X_arr.shape[1]
        self.tree_ = self._grow(X_arr, encoded, depth=0)
        return self

    def _impurity(self, counts: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """Impurity of several count vectors at once; counts is (m, K), sizes (m,)."""
        p = counts / sizes[:, None]
        if self.criterion == "gini":
            return 1.0 - (p**2).sum(axis=1)
        logp = np.zeros_like(p)
        np.log2(p, out=logp, where=p > 0)  # leaves log(0) terms at zero
        return -(p * logp).sum(axis=1)

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        counts = np.bincount(y, minlength=len(self.classes_))
        at_limit = self.max_depth is not None and depth >= self.max_depth
        if len(y) < self.min_samples_split or at_limit or counts.max() == len(y):
            return _Node(value=counts)
        feature, threshold = self._best_split(X, y, counts)
        if feature is None:
            return _Node(value=counts)
        mask = X[:, feature] <= threshold
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._grow(X[mask], y[mask], depth + 1),
            right=self._grow(X[~mask], y[~mask], depth + 1),
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, parent_counts: np.ndarray
    ) -> tuple[int | None, float]:
        n = len(y)
        parent_imp = self._impurity(parent_counts[None, :].astype(float), np.array([n]))[0]
        onehot = np.eye(len(self.classes_))[y]
        best_gain = 1e-12
        best: tuple[int | None, float] = (None, 0.0)
        for j in range(self.n_features_):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            cut = np.flatnonzero(xs[1:] > xs[:-1])  # split between distinct values
            if cut.size == 0:
                continue
            left_counts = np.cumsum(onehot[order], axis=0)[cut]
            nl = (cut + 1).astype(np.float64)
            nr = n - nl
            weighted = (
                nl * self._impurity(left_counts, nl)
                + nr * self._impurity(parent_counts - left_counts, nr)
            ) / n
            gains = parent_imp - weighted
            i = int(gains.argmax())
            if gains[i] > best_gain:
                best_gain = float(gains[i])
                best = (j, float((xs[cut[i]] + xs[cut[i] + 1]) / 2.0))
        return best

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "tree_")
        X_arr = check_array(X)
        out = np.empty(len(X_arr), dtype=int)
        for i, x in enumerate(X_arr):
            node = self.tree_
            while node.feature is not None:
                child = node.left if x[node.feature] <= node.threshold else node.right
                assert child is not None
                node = child
            assert node.value is not None
            out[i] = int(node.value.argmax())
        return self.classes_[out]
