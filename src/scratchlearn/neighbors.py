"""k-nearest-neighbour classification."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._validation import check_array, check_is_fitted, check_X_y
from .base import BaseEstimator, ClassifierMixin


class KNNClassifier(ClassifierMixin, BaseEstimator):
    """Classify by (optionally distance-weighted) vote among the k nearest neighbours.

    All pairwise distances are computed in one shot with the expansion
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, so there is no Python loop
    over samples.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X: ArrayLike, y: ArrayLike) -> KNNClassifier:
        if self.weights not in ("uniform", "distance"):
            raise ValueError(f"unknown weights {self.weights!r}")
        X_arr, y_arr = check_X_y(X, y)
        if self.n_neighbors > len(X_arr):
            raise ValueError(
                f"n_neighbors={self.n_neighbors} but only {len(X_arr)} training samples"
            )
        self.classes_, self._y = np.unique(y_arr, return_inverse=True)
        self._X = X_arr
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "_X")
        X_arr = check_array(X)
        d2 = (
            (X_arr**2).sum(axis=1)[:, None]
            + (self._X**2).sum(axis=1)[None, :]
            - 2.0 * X_arr @ self._X.T
        )
        np.maximum(d2, 0.0, out=d2)  # rounding can produce tiny negatives
        nearest = np.argpartition(d2, self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
        labels = self._y[nearest]
        if self.weights == "uniform":
            w = np.ones(nearest.shape)
        else:
            rows = np.arange(len(X_arr))[:, None]
            # a floor on the distance keeps the weight finite for exact matches
            w = 1.0 / np.maximum(np.sqrt(d2[rows, nearest]), 1e-12)
        votes = np.zeros((len(X_arr), len(self.classes_)))
        for k in range(len(self.classes_)):
            votes[:, k] = np.where(labels == k, w, 0.0).sum(axis=1)
        return self.classes_[votes.argmax(axis=1)]
