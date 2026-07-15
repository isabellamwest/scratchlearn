"""Clustering: k-means and Gaussian mixture models."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

from ._validation import check_array, check_is_fitted
from .base import BaseEstimator

logger = logging.getLogger(__name__)


def _squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared distances via ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b."""
    d2 = (A**2).sum(axis=1)[:, None] + (B**2).sum(axis=1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d2, 0.0)  # rounding can push tiny values just below zero


def _log_sum_exp(a: np.ndarray, axis: int) -> np.ndarray:
    """log(sum(exp(a))) with the maximum factored out first.

    exp overflows for arguments above ~709, and Gaussian log-densities in
    the EM E-step easily reach that. Since logsumexp(a) = m + logsumexp(a - m)
    for any m, subtracting the row maximum makes every exponent <= 0.
    """
    m = a.max(axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True)), axis=axis)


class KMeans(BaseEstimator):
    """Lloyd's algorithm with k-means++ initialisation.

    Runs n_init times from different starting centres and keeps the run
    with the lowest inertia (total squared distance to the closest centre).
    """

    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-6,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: ArrayLike) -> KMeans:
        X_arr = check_array(X)
        if len(X_arr) < self.n_clusters:
            raise ValueError(f"{len(X_arr)} samples cannot form {self.n_clusters} clusters")
        rng = np.random.default_rng(self.random_state)
        best_inertia = np.inf
        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._single_run(X_arr, rng)
            if inertia < best_inertia:
                best = (centers, labels, inertia, n_iter)
                best_inertia = inertia
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        logger.debug("best of %d runs: inertia=%.4f", self.n_init, self.inertia_)
        return self

    def _single_run(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        centers = self._init_centers(X, rng)
        for it in range(self.max_iter):
            d2 = _squared_distances(X, centers)
            labels = d2.argmin(axis=1)
            new_centers = np.empty_like(centers)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) == 0:
                    # re-seed an emptied cluster on the point furthest from its centre
                    new_centers[k] = X[d2.min(axis=1).argmax()]
                else:
                    new_centers[k] = members.mean(axis=0)
            shift = float(((new_centers - centers) ** 2).sum())
            centers = new_centers
            if shift < self.tol:
                break
        d2 = _squared_distances(X, centers)
        labels = d2.argmin(axis=1)
        inertia = float(d2[np.arange(len(X)), labels].sum())
        return centers, labels, inertia, it + 1

    def _init_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """k-means++: sample each new centre with probability proportional to D^2."""
        centers = np.empty((self.n_clusters, X.shape[1]))
        centers[0] = X[rng.integers(len(X))]
        closest = _squared_distances(X, centers[:1])[:, 0]
        for k in range(1, self.n_clusters):
            total = closest.sum()
            if total == 0.0:  # every point already sits on a centre
                idx = int(rng.integers(len(X)))
            else:
                idx = int(rng.choice(len(X), p=closest / total))
            centers[k] = X[idx]
            closest = np.minimum(closest, _squared_distances(X, centers[k : k + 1])[:, 0])
        return centers

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Assign each sample to its nearest fitted centre."""
        check_is_fitted(self, "cluster_centers_")
        return _squared_distances(check_array(X), self.cluster_centers_).argmin(axis=1)


class GaussianMixture(BaseEstimator):
    """Mixture of full-covariance Gaussians fitted by expectation-maximisation.

    The E-step computes responsibilities in log space via _log_sum_exp; the
    M-step re-estimates weights, means and covariances from them. The mean
    log-likelihood is recorded each iteration in log_likelihood_history_ and,
    as EM guarantees, never decreases. Derivation (including the ELBO/KL
    view) in docs/derivations/gmm_em.md.
    """

    def __init__(
        self,
        n_components: int = 1,
        max_iter: int = 100,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X: ArrayLike) -> GaussianMixture:
        X_arr = check_array(X)
        n = len(X_arr)

        # start from a cheap k-means run so EM begins near a sensible solution
        km = KMeans(n_clusters=self.n_components, n_init=1, random_state=self.random_state)
        km.fit(X_arr)
        resp = np.zeros((n, self.n_components))
        resp[np.arange(n), km.labels_] = 1.0
        self._m_step(X_arr, resp)

        history: list[float] = []
        prev = -np.inf
        self.converged_ = False
        for it in range(self.max_iter):
            log_resp, mean_ll = self._e_step(X_arr)
            history.append(mean_ll)
            self._m_step(X_arr, np.exp(log_resp))
            if abs(mean_ll - prev) < self.tol:
                self.converged_ = True
                break
            prev = mean_ll
        logger.debug("EM ran %d iterations, mean log-likelihood %.4f", it + 1, history[-1])
        self.log_likelihood_history_ = history
        self.n_iter_ = it + 1
        return self

    def _log_prob(self, X: np.ndarray) -> np.ndarray:
        """log N(x | mu_k, Sigma_k) for every sample and component, via Cholesky factors."""
        n, d = X.shape
        out = np.empty((n, self.n_components))
        for k in range(self.n_components):
            L = np.linalg.cholesky(self.covariances_[k])
            z = np.linalg.solve(L, (X - self.means_[k]).T)
            maha = (z**2).sum(axis=0)
            log_det = 2.0 * float(np.log(np.diag(L)).sum())
            out[:, k] = -0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)
        return out

    def _e_step(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        weighted = self._log_prob(X) + np.log(self.weights_)
        norm = _log_sum_exp(weighted, axis=1)
        return weighted - norm[:, None], float(norm.mean())

    def _m_step(self, X: np.ndarray, resp: np.ndarray) -> None:
        n, d = X.shape
        nk = resp.sum(axis=0) + 10.0 * np.finfo(np.float64).eps
        self.weights_ = nk / n
        self.means_ = (resp.T @ X) / nk[:, None]
        covariances = np.empty((self.n_components, d, d))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            covariances[k] = (resp[:, k] * diff.T) @ diff / nk[k]
            covariances[k][np.diag_indices(d)] += self.reg_covar
        self.covariances_ = covariances

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Responsibility of each component for each sample."""
        check_is_fitted(self, "means_")
        weighted = self._log_prob(check_array(X)) + np.log(self.weights_)
        return np.exp(weighted - _log_sum_exp(weighted, axis=1)[:, None])

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Most responsible component for each sample."""
        check_is_fitted(self, "means_")
        return (self._log_prob(check_array(X)) + np.log(self.weights_)).argmax(axis=1)
