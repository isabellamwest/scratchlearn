"""Linear models: least squares, ridge, lasso and logistic regression."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

from ._validation import check_array, check_is_fitted, check_X_y
from .base import BaseEstimator, ClassifierMixin, RegressorMixin

logger = logging.getLogger(__name__)


def _soft_threshold(z: float, gamma: float) -> float:
    """S(z, gamma) = sign(z) * max(|z| - gamma, 0), the proximal step for the L1 penalty."""
    return float(np.sign(z) * max(abs(z) - gamma, 0.0))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function with inputs clipped to +-500 so exp cannot overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


class LinearRegression(RegressorMixin, BaseEstimator):
    """Ordinary least squares, solved exactly or by gradient descent.

    Minimises the mean squared error ||y - Xw - b||^2 / n. solver="normal"
    uses np.linalg.lstsq on the design matrix (the normal equations, solved
    stably); solver="gd" runs full-batch gradient descent and records the
    loss at every step in loss_history_.
    """

    def __init__(
        self, solver: str = "normal", lr: float = 0.05, max_iter: int = 1000, tol: float = 1e-10
    ):
        self.solver = solver
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegression:
        X_arr, y_arr = check_X_y(X, y)
        y_arr = y_arr.astype(np.float64)
        if self.solver == "normal":
            design = np.c_[np.ones(len(X_arr)), X_arr]
            beta, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
            self.loss_history_: list[float] = []
        elif self.solver == "gd":
            self._fit_gd(X_arr, y_arr)
        else:
            raise ValueError(f"unknown solver {self.solver!r}")
        return self

    def _fit_gd(self, X: np.ndarray, y: np.ndarray) -> None:
        n, p = X.shape
        w = np.zeros(p)
        b = 0.0
        losses = []
        prev = np.inf
        for it in range(self.max_iter):
            resid = X @ w + b - y
            loss = float(resid @ resid / n)
            losses.append(loss)
            w -= self.lr * (2.0 * X.T @ resid / n)
            b -= self.lr * (2.0 * resid.mean())
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        logger.debug("gd stopped after %d iterations, mse=%.3e", it + 1, loss)
        self.coef_ = w
        self.intercept_ = float(b)
        self.loss_history_ = losses

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class Ridge(RegressorMixin, BaseEstimator):
    """Linear regression with an L2 penalty on the weights.

    Minimises ||y - Xw - b||^2 + alpha * ||w||^2, leaving the intercept
    unpenalised: X and y are centred first, then (Xc'Xc + alpha*I) w = Xc'y
    is solved with np.linalg.solve. An explicit inverse would be slower and
    less accurate on ill-conditioned problems.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: ArrayLike, y: ArrayLike) -> Ridge:
        X_arr, y_arr = check_X_y(X, y)
        y_arr = y_arr.astype(np.float64)
        x_mean = X_arr.mean(axis=0)
        y_mean = float(y_arr.mean())
        Xc = X_arr - x_mean
        A = Xc.T @ Xc + self.alpha * np.eye(X_arr.shape[1])
        self.coef_ = np.linalg.solve(A, Xc.T @ (y_arr - y_mean))
        self.intercept_ = y_mean - float(x_mean @ self.coef_)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class Lasso(RegressorMixin, BaseEstimator):
    """Linear regression with an L1 penalty, fitted by cyclic coordinate descent.

    Minimises ||y - Xw - b||^2 / (2n) + alpha * ||w||_1, the same objective
    as scikit-learn. Each coordinate update is a soft-thresholding step, which
    is what drives small coefficients to exactly zero.
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-8):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> Lasso:
        X_arr, y_arr = check_X_y(X, y)
        y_arr = y_arr.astype(np.float64)
        n, p = X_arr.shape
        x_mean = X_arr.mean(axis=0)
        y_mean = float(y_arr.mean())
        Xc = X_arr - x_mean
        col_norm = (Xc**2).mean(axis=0)  # ||x_j||^2 / n for each column

        w = np.zeros(p)
        resid = y_arr - y_mean
        for sweep in range(self.max_iter):
            largest = 0.0
            for j in range(p):
                if col_norm[j] == 0.0:
                    continue
                old = w[j]
                rho = float(Xc[:, j] @ resid) / n + col_norm[j] * old
                w[j] = _soft_threshold(rho, self.alpha) / col_norm[j]
                if w[j] != old:
                    resid -= Xc[:, j] * (w[j] - old)
                    largest = max(largest, abs(w[j] - old))
            if largest < self.tol:
                break
        logger.debug("coordinate descent stopped after %d sweeps", sweep + 1)
        self.coef_ = w
        self.intercept_ = y_mean - float(x_mean @ w)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "coef_")
        return check_array(X) @ self.coef_ + self.intercept_


class LogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary logistic regression fitted by gradient descent.

    Minimises mean binary cross-entropy plus (alpha/2) * ||w||^2. The
    gradient X'(sigmoid(Xw + b) - y) / n + alpha * w is derived step by
    step in docs/derivations/logistic_regression.md.
    """

    def __init__(
        self, lr: float = 0.1, alpha: float = 0.0, max_iter: int = 1000, tol: float = 1e-8
    ):
        self.lr = lr
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegression:
        X_arr, y_arr = check_X_y(X, y)
        self.classes_ = np.unique(y_arr)
        if self.classes_.size != 2:
            raise ValueError("only binary targets are supported")
        t = (y_arr == self.classes_[1]).astype(np.float64)

        n, p = X_arr.shape
        w = np.zeros(p)
        b = 0.0
        losses = []
        prev = np.inf
        for it in range(self.max_iter):
            z = X_arr @ w + b
            # log(1 + e^z) - t*z is the cross-entropy in a form that never overflows
            loss = float(np.mean(np.logaddexp(0.0, z) - t * z) + 0.5 * self.alpha * (w @ w))
            losses.append(loss)
            err = _sigmoid(z) - t
            w -= self.lr * (X_arr.T @ err / n + self.alpha * w)
            b -= self.lr * float(err.mean())
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        logger.debug("gd stopped after %d iterations, loss=%.4f", it + 1, loss)
        self.coef_ = w
        self.intercept_ = float(b)
        self.loss_history_ = losses
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Class probabilities, one column per entry of classes_."""
        check_is_fitted(self, "coef_")
        p = _sigmoid(check_array(X) @ self.coef_ + self.intercept_)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
