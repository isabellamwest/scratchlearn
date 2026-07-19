"""Principal component analysis via eigendecomposition or SVD."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._validation import check_array, check_is_fitted
from .base import BaseEstimator


class PCA(BaseEstimator):
    """Project data onto the directions of maximum variance.

    method="eigen" diagonalises the sample covariance X'X/(n-1);
    method="svd" takes the right singular vectors of the centred data.
    The two agree up to sign (docs/derivations/pca.md shows why), so each
    component's sign is fixed to make its largest entry positive.
    """

    def __init__(self, n_components: int | None = None, method: str = "svd"):
        self.n_components = n_components
        self.method = method

    def fit(self, X: ArrayLike) -> PCA:
        X_arr = check_array(X)
        n = len(X_arr)
        self.mean_ = X_arr.mean(axis=0)
        Xc = X_arr - self.mean_

        if self.method == "eigen":
            cov = Xc.T @ Xc / (n - 1)
            values, vectors = np.linalg.eigh(cov)
            order = np.argsort(values)[::-1]
            variances = values[order]
            components = vectors[:, order].T
        elif self.method == "svd":
            _, s, vt = np.linalg.svd(Xc, full_matrices=False)
            variances = s**2 / (n - 1)
            components = vt
        else:
            raise ValueError(f"unknown method {self.method!r}")

        variances = np.clip(variances, 0.0, None)  # eigh can return tiny negative values
        flip = components[np.arange(len(components)), np.abs(components).argmax(axis=1)] < 0
        components[flip] *= -1.0

        k = components.shape[0] if self.n_components is None else self.n_components
        self.components_ = components[:k]
        self.explained_variance_ = variances[:k]
        total = variances.sum()
        # A constant input has zero total variance; report each ratio as 0
        # rather than 0/0 = NaN, since no direction explains any variance.
        self.explained_variance_ratio_ = (
            variances[:k] / total if total > 0.0 else np.zeros_like(variances[:k])
        )
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Project X onto the fitted components."""
        check_is_fitted(self, "components_")
        return (check_array(X) - self.mean_) @ self.components_.T

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> np.ndarray:
        """Map projected data back to the original feature space."""
        check_is_fitted(self, "components_")
        return np.asarray(Z) @ self.components_ + self.mean_
