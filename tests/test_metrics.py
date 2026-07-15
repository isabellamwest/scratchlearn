import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import metrics as skm

from scratchlearn import metrics


@pytest.fixture
def binary():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    y_score = rng.random(200)
    return y_true, y_pred, y_score


@pytest.fixture
def multiclass():
    rng = np.random.default_rng(1)
    return rng.integers(0, 4, 200), rng.integers(0, 4, 200)


def test_accuracy(binary):
    y_true, y_pred, _ = binary
    assert_allclose(metrics.accuracy_score(y_true, y_pred), skm.accuracy_score(y_true, y_pred))


def test_precision_recall_f1_binary(binary):
    y_true, y_pred, _ = binary
    assert_allclose(
        metrics.precision_score(y_true, y_pred), skm.precision_score(y_true, y_pred), rtol=1e-6
    )
    assert_allclose(
        metrics.recall_score(y_true, y_pred), skm.recall_score(y_true, y_pred), rtol=1e-6
    )
    assert_allclose(metrics.f1_score(y_true, y_pred), skm.f1_score(y_true, y_pred), rtol=1e-6)


def test_precision_recall_f1_macro(multiclass):
    y_true, y_pred = multiclass
    for ours, theirs in [
        (metrics.precision_score, skm.precision_score),
        (metrics.recall_score, skm.recall_score),
        (metrics.f1_score, skm.f1_score),
    ]:
        assert_allclose(
            ours(y_true, y_pred, average="macro"),
            theirs(y_true, y_pred, average="macro"),
            rtol=1e-6,
        )


def test_confusion_matrix(multiclass):
    y_true, y_pred = multiclass
    assert_array_equal(
        metrics.confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred)
    )


def test_roc_curve_shape_and_monotonicity(binary):
    y_true, _, y_score = binary
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == 1.0 and tpr[-1] == 1.0
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)
    assert np.all(np.diff(thresholds) < 0)


def test_roc_curve_matches_sklearn(binary):
    y_true, _, y_score = binary
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    sk_fpr, sk_tpr, sk_thr = skm.roc_curve(y_true, y_score, drop_intermediate=False)
    assert_allclose(fpr, sk_fpr)
    assert_allclose(tpr, sk_tpr)
    assert_allclose(thresholds[1:], sk_thr[1:])  # conventions for the first threshold vary


def test_roc_auc(binary):
    y_true, _, y_score = binary
    assert_allclose(
        metrics.roc_auc_score(y_true, y_score), skm.roc_auc_score(y_true, y_score), rtol=1e-6
    )


def test_roc_curve_rejects_single_class():
    with pytest.raises(ValueError):
        metrics.roc_curve(np.ones(10), np.linspace(0, 1, 10))


def test_regression_metrics():
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.5, size=200)
    assert_allclose(
        metrics.rmse(y_true, y_pred), np.sqrt(skm.mean_squared_error(y_true, y_pred)), rtol=1e-6
    )
    assert_allclose(
        metrics.mae(y_true, y_pred), skm.mean_absolute_error(y_true, y_pred), rtol=1e-6
    )
    assert_allclose(metrics.r2_score(y_true, y_pred), skm.r2_score(y_true, y_pred), rtol=1e-6)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.accuracy_score([0, 1], [0, 1, 0])
