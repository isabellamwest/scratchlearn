import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_digits, make_blobs

from scratchlearn.exceptions import NotFittedError
from scratchlearn.model_selection import train_test_split
from scratchlearn.neural import MLPClassifier


def test_gradients_match_finite_differences():
    """Every analytic gradient must agree with central differences to 1e-6."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 2))
    Y = np.eye(2)[rng.integers(0, 2, 5)]
    net = MLPClassifier(hidden_layer_sizes=(3,), random_state=0)
    net._initialise(2, 2, np.random.default_rng(1))

    _, grads_w, grads_b = net._gradients(X, Y)
    eps = 1e-5
    for params, grads in [(net.weights_, grads_w), (net.biases_, grads_b)]:
        for layer, grad in zip(params, grads):
            numeric = np.empty_like(layer)
            for idx in np.ndindex(layer.shape):
                original = layer[idx]
                layer[idx] = original + eps
                up = net._loss(X, Y)
                layer[idx] = original - eps
                down = net._loss(X, Y)
                layer[idx] = original
                numeric[idx] = (up - down) / (2 * eps)
            scale = np.linalg.norm(grad) + np.linalg.norm(numeric)
            assert np.linalg.norm(grad - numeric) / max(scale, 1e-12) < 1e-6


def test_loss_decreases_and_blobs_are_learnable():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=0)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    net = MLPClassifier(hidden_layer_sizes=(16,), epochs=30, random_state=0).fit(X, y)
    assert net.loss_history_[-1] < net.loss_history_[0]
    assert net.score(X, y) > 0.9


def test_digits_accuracy():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    net = MLPClassifier(hidden_layer_sizes=(64,), epochs=50, random_state=0).fit(X_tr, y_tr)
    assert net.score(X_te, y_te) >= 0.95


def test_predict_proba_rows_sum_to_one():
    X, y = make_blobs(n_samples=90, centers=3, random_state=1)
    net = MLPClassifier(hidden_layer_sizes=(8,), epochs=10, random_state=0).fit(X, y)
    proba = net.predict_proba(X)
    assert proba.shape == (90, 3)
    assert_allclose(proba.sum(axis=1), 1.0)
    assert set(net.predict(X)) <= set(net.classes_)


def test_unfitted_predict_raises():
    with pytest.raises(NotFittedError):
        MLPClassifier().predict(np.zeros((3, 2)))
