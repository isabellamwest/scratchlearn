"""Base classes shared by every estimator in the library."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from . import metrics


class BaseEstimator:
    """Provides get_params and a readable repr, mirroring the scikit-learn contract."""

    def get_params(self) -> dict[str, Any]:
        """Return the constructor arguments and their current values."""
        init = inspect.signature(type(self).__init__)
        names = [name for name in init.parameters if name != "self"]
        return {name: getattr(self, name) for name in names}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class ClassifierMixin:
    """Adds accuracy scoring to classifiers."""

    predict: Callable[..., np.ndarray]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        return metrics.accuracy_score(y, self.predict(X))


class RegressorMixin:
    """Adds R^2 scoring to regressors."""

    predict: Callable[..., np.ndarray]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        return metrics.r2_score(y, self.predict(X))
