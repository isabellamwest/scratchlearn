"""Evaluation metrics for classification and regression."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _column(y: ArrayLike) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-d array, got shape {arr.shape}")
    return arr


def _paired(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a, b = _column(y_true), _column(y_pred)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return a, b


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of predictions that match the true labels."""
    a, b = _paired(y_true, y_pred)
    return float(np.mean(a == b))


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Matrix C where C[i, j] counts samples of class i predicted as class j."""
    a, b = _paired(y_true, y_pred)
    labels = np.unique(np.concatenate([a, b]))
    ti = np.searchsorted(labels, a)
    pi = np.searchsorted(labels, b)
    cm = np.zeros((labels.size, labels.size), dtype=np.int64)
    np.add.at(cm, (ti, pi), 1)
    return cm


def _per_class_scores(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall and F1 for every label, with 0 where a denominator is 0."""
    a, b = _paired(y_true, y_pred)
    labels = np.unique(np.concatenate([a, b]))
    cm = confusion_matrix(a, b).astype(np.float64)
    tp = np.diag(cm)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return labels, precision, recall, f1


def _averaged(values: np.ndarray, labels: np.ndarray, average: str, pos_label: object) -> float:
    if average == "binary":
        where = np.flatnonzero(labels == pos_label)
        if where.size == 0:
            raise ValueError(f"pos_label {pos_label!r} does not appear in the data")
        return float(values[where[0]])
    if average == "macro":
        return float(values.mean())
    raise ValueError("average must be 'binary' or 'macro'")


def precision_score(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary", pos_label: object = 1
) -> float:
    """Precision tp / (tp + fp) for the positive class, or macro-averaged."""
    labels, precision, _, _ = _per_class_scores(y_true, y_pred)
    return _averaged(precision, labels, average, pos_label)


def recall_score(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary", pos_label: object = 1
) -> float:
    """Recall tp / (tp + fn) for the positive class, or macro-averaged."""
    labels, _, recall, _ = _per_class_scores(y_true, y_pred)
    return _averaged(recall, labels, average, pos_label)


def f1_score(
    y_true: ArrayLike, y_pred: ArrayLike, average: str = "binary", pos_label: object = 1
) -> float:
    """Harmonic mean of precision and recall."""
    labels, _, _, f1 = _per_class_scores(y_true, y_pred)
    return _averaged(f1, labels, average, pos_label)


def roc_curve(
    y_true: ArrayLike, y_score: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """False and true positive rates as the decision threshold sweeps downwards.

    Label 1 is treated as the positive class. Returns (fpr, tpr, thresholds),
    starting from the (0, 0) point at threshold +inf.
    """
    y, s = _paired(y_true, y_score)
    pos = y == 1
    order = np.argsort(-s, kind="stable")
    pos, s = pos[order], s[order]
    # one curve point per distinct score: index of the last sample in each run
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    tps = np.cumsum(pos)[last].astype(np.float64)
    fps = (last + 1) - tps
    if tps[-1] == 0 or fps[-1] == 0:
        raise ValueError("roc_curve needs at least one positive and one negative sample")
    fpr = np.r_[0.0, fps / fps[-1]]
    tpr = np.r_[0.0, tps / tps[-1]]
    thresholds = np.r_[np.inf, s[last]]
    return fpr, tpr, thresholds


def roc_auc_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve, by trapezoidal integration."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1])) / 2.0)


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error."""
    a, b = _paired(y_true, y_pred)
    return float(np.sqrt(np.mean((a.astype(float) - b.astype(float)) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error."""
    a, b = _paired(y_true, y_pred)
    return float(np.mean(np.abs(a.astype(float) - b.astype(float))))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    a, b = _paired(y_true, y_pred)
    a, b = a.astype(float), b.astype(float)
    ss_res = float(((a - b) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot
