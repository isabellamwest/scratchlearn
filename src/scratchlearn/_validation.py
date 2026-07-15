"""Input checking helpers shared by every estimator."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .exceptions import NotFittedError


def check_array(X: ArrayLike) -> np.ndarray:
    """Convert X to a 2-d float64 array, rejecting NaN and infinity."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-d array, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("input contains NaN or infinity")
    return arr


def check_X_y(X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate a feature matrix and a target vector of matching length."""
    arr = check_array(X)
    target = np.asarray(y)
    if target.ndim != 1:
        raise ValueError(f"expected a 1-d target, got shape {target.shape}")
    if len(arr) != len(target):
        raise ValueError(f"X has {len(arr)} rows but y has {len(target)}")
    return arr, target


def check_is_fitted(estimator: object, attribute: str) -> None:
    """Raise NotFittedError unless `attribute` exists on the estimator."""
    if not hasattr(estimator, attribute):
        raise NotFittedError(
            f"this {type(estimator).__name__} instance is not fitted yet; call fit first"
        )
