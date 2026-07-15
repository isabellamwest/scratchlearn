"""Data splitting, cross-validation and grid search."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import metrics
from ._validation import check_is_fitted

_SCORERS = {
    "accuracy": metrics.accuracy_score,
    "f1": metrics.f1_score,
    "r2": metrics.r2_score,
    "rmse": metrics.rmse,
    "mae": metrics.mae,
}
_LOWER_IS_BETTER = {"rmse", "mae"}


def clone(estimator: Any) -> Any:
    """Return an unfitted copy built from the estimator's constructor params."""
    return type(estimator)(**estimator.get_params())


def train_test_split(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.25,
    stratify: ArrayLike | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split into train and test sets.

    Pass the labels as `stratify` to keep class proportions the same
    in both halves.
    """
    X_arr, y_arr = np.asarray(X), np.asarray(y)
    n = len(X_arr)
    if len(y_arr) != n:
        raise ValueError(f"X has {n} rows but y has {len(y_arr)}")
    rng = np.random.default_rng(random_state)
    n_test = int(np.ceil(n * test_size))
    if not 0 < n_test < n:
        raise ValueError(f"test_size={test_size} leaves an empty train or test set")

    if stratify is None:
        order = rng.permutation(n)
        test_idx, train_idx = order[:n_test], order[n_test:]
    else:
        strat = np.asarray(stratify)
        classes, counts = np.unique(strat, return_counts=True)
        # floor of each class quota, then hand the remainder to the largest fractions
        quotas = counts * n_test / n
        take = np.floor(quotas).astype(int)
        for i in np.argsort(-(quotas - take))[: n_test - take.sum()]:
            take[i] += 1
        test_parts, train_parts = [], []
        for cls, k in zip(classes, take):
            members = rng.permutation(np.flatnonzero(strat == cls))
            test_parts.append(members[:k])
            train_parts.append(members[k:])
        test_idx = rng.permutation(np.concatenate(test_parts))
        train_idx = rng.permutation(np.concatenate(train_parts))

    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


class KFold:
    """Split indices into k consecutive (or shuffled) folds."""

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int | None = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(np.asarray(X))
        if not 2 <= self.n_splits <= n:
            raise ValueError(f"n_splits={self.n_splits} is invalid for {n} samples")
        indices = np.arange(n)
        if self.shuffle:
            np.random.default_rng(self.random_state).shuffle(indices)
        sizes = np.full(self.n_splits, n // self.n_splits)
        sizes[: n % self.n_splits] += 1
        start = 0
        for size in sizes:
            test = indices[start : start + size]
            train = np.concatenate([indices[:start], indices[start + size :]])
            yield train, test
            start += size


class StratifiedKFold:
    """KFold that keeps class proportions roughly equal in every fold."""

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int | None = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: ArrayLike, y: ArrayLike) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        y_arr = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        fold_of = np.empty(len(y_arr), dtype=int)
        for cls in np.unique(y_arr):
            members = np.flatnonzero(y_arr == cls)
            if members.size < self.n_splits:
                raise ValueError(f"class {cls!r} has fewer samples than n_splits")
            if self.shuffle:
                rng.shuffle(members)
            fold_of[members] = np.arange(members.size) % self.n_splits
        for fold in range(self.n_splits):
            yield np.flatnonzero(fold_of != fold), np.flatnonzero(fold_of == fold)


def cross_val_score(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
    cv: int | KFold | StratifiedKFold = 5,
    scoring: str | None = None,
) -> np.ndarray:
    """Fit a fresh clone on each fold and return the array of test scores.

    scoring=None uses the estimator's own score method; otherwise pass one
    of 'accuracy', 'f1', 'r2', 'rmse' or 'mae'.
    """
    X_arr, y_arr = np.asarray(X), np.asarray(y)
    splitter = KFold(n_splits=cv) if isinstance(cv, int) else cv
    scores = []
    for train, test in splitter.split(X_arr, y_arr):
        fitted = clone(estimator).fit(X_arr[train], y_arr[train])
        if scoring is None:
            scores.append(fitted.score(X_arr[test], y_arr[test]))
        else:
            scores.append(_SCORERS[scoring](y_arr[test], fitted.predict(X_arr[test])))
    return np.asarray(scores)


class GridSearchCV:
    """Exhaustive search over a dict-of-lists parameter grid, scored by cross-validation."""

    def __init__(
        self,
        estimator: Any,
        param_grid: dict[str, list],
        cv: int | KFold | StratifiedKFold = 5,
        scoring: str | None = None,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def fit(self, X: ArrayLike, y: ArrayLike) -> GridSearchCV:
        names = list(self.param_grid)
        results: dict[str, list] = {"params": [], "mean_score": [], "std_score": []}
        for combo in itertools.product(*self.param_grid.values()):
            params = dict(zip(names, combo))
            candidate = type(self.estimator)(**{**self.estimator.get_params(), **params})
            scores = cross_val_score(candidate, X, y, cv=self.cv, scoring=self.scoring)
            results["params"].append(params)
            results["mean_score"].append(float(scores.mean()))
            results["std_score"].append(float(scores.std()))

        means = np.asarray(results["mean_score"])
        best = int(means.argmin() if self.scoring in _LOWER_IS_BETTER else means.argmax())
        self.cv_results_ = results
        self.best_index_ = best
        self.best_params_ = results["params"][best]
        self.best_score_ = results["mean_score"][best]
        self.best_estimator_ = type(self.estimator)(
            **{**self.estimator.get_params(), **self.best_params_}
        ).fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.score(X, y)
