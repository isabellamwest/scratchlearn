"""A multilayer perceptron trained with hand-written backpropagation."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike

from ._validation import check_array, check_is_fitted, check_X_y
from .base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


def _softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax; the row maximum is subtracted first so exp cannot overflow."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class MLPClassifier(ClassifierMixin, BaseEstimator):
    """Fully connected net: ReLU hidden layers, softmax output, cross-entropy loss.

    Trained with mini-batch SGD plus momentum. The four backprop equations
    are implemented directly in _gradients and verified against central
    finite differences in tests/test_neural.py; the full chain-rule
    derivation lives in docs/derivations/mlp_backprop.md.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64,),
        lr: float = 0.1,
        batch_size: int = 32,
        epochs: int = 100,
        momentum: float = 0.9,
        random_state: int | None = None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.random_state = random_state

    def _initialise(self, n_in: int, n_out: int, rng: np.random.Generator) -> None:
        """He initialisation: std sqrt(2 / fan_in) keeps ReLU activations from dying out."""
        sizes = [n_in, *self.hidden_layer_sizes, n_out]
        self.weights_ = [
            rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b))
            for a, b in zip(sizes[:-1], sizes[1:])
        ]
        self.biases_ = [np.zeros(b) for b in sizes[1:]]

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return pre-activations z and activations a for every layer, inputs included."""
        zs: list[np.ndarray] = []
        activations = [X]
        a = X
        last = len(self.weights_) - 1
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = a @ W + b
            a = _softmax(z) if i == last else np.maximum(z, 0.0)
            zs.append(z)
            activations.append(a)
        return zs, activations

    def _loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Mean cross-entropy against one-hot targets Y."""
        probs = self._forward(X)[1][-1]
        true_p = (probs * Y).sum(axis=1)
        return float(-np.mean(np.log(np.clip(true_p, 1e-12, None))))

    def _gradients(
        self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """One backward pass: (loss, dL/dW per layer, dL/db per layer)."""
        zs, activations = self._forward(X)
        probs = activations[-1]
        true_p = (probs * Y).sum(axis=1)
        loss = float(-np.mean(np.log(np.clip(true_p, 1e-12, None))))

        n_layers = len(self.weights_)
        grads_w = [np.empty(0)] * n_layers
        grads_b = [np.empty(0)] * n_layers
        delta = (probs - Y) / len(X)  # dL/dz at the output: softmax with cross-entropy
        for layer in range(n_layers - 1, -1, -1):
            grads_w[layer] = activations[layer].T @ delta
            grads_b[layer] = delta.sum(axis=0)
            if layer > 0:
                delta = (delta @ self.weights_[layer].T) * (zs[layer - 1] > 0)  # ReLU'
        return loss, grads_w, grads_b

    def fit(self, X: ArrayLike, y: ArrayLike) -> MLPClassifier:
        X_arr, y_arr = check_X_y(X, y)
        rng = np.random.default_rng(self.random_state)
        self.classes_, encoded = np.unique(y_arr, return_inverse=True)
        Y = np.eye(len(self.classes_))[encoded]
        self._initialise(X_arr.shape[1], len(self.classes_), rng)

        vel_w = [np.zeros_like(W) for W in self.weights_]
        vel_b = [np.zeros_like(b) for b in self.biases_]
        n = len(X_arr)
        self.loss_history_: list[float] = []
        for _ in range(self.epochs):
            order = rng.permutation(n)
            batch_losses = []
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                loss, grads_w, grads_b = self._gradients(X_arr[batch], Y[batch])
                batch_losses.append(loss)
                for layer in range(len(self.weights_)):
                    vel_w[layer] = self.momentum * vel_w[layer] - self.lr * grads_w[layer]
                    vel_b[layer] = self.momentum * vel_b[layer] - self.lr * grads_b[layer]
                    self.weights_[layer] += vel_w[layer]
                    self.biases_[layer] += vel_b[layer]
            self.loss_history_.append(float(np.mean(batch_losses)))
        logger.debug("trained %d epochs, final loss %.4f", self.epochs, self.loss_history_[-1])
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Softmax class probabilities, one column per entry of classes_."""
        check_is_fitted(self, "weights_")
        return self._forward(check_array(X))[1][-1]

    def predict(self, X: ArrayLike) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]
