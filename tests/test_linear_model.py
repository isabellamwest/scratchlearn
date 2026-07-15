import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import linear_model as sklm
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import roc_auc_score

from scratchlearn.exceptions import NotFittedError
from scratchlearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from scratchlearn.model_selection import train_test_split


@pytest.fixture
def regression_data():
    return make_regression(n_samples=120, n_features=6, noise=5.0, random_state=0)


def test_linear_regression_normal_matches_sklearn(regression_data):
    X, y = regression_data
    ours = LinearRegression().fit(X, y)
    theirs = sklm.LinearRegression().fit(X, y)
    assert_allclose(ours.coef_, theirs.coef_, atol=1e-6)
    assert_allclose(ours.intercept_, theirs.intercept_, atol=1e-6)


def test_linear_regression_gd_agrees_with_normal(regression_data):
    X, y = regression_data
    exact = LinearRegression().fit(X, y)
    gd = LinearRegression(solver="gd", lr=0.05, max_iter=20000, tol=1e-14).fit(X, y)
    assert_allclose(gd.coef_, exact.coef_, atol=1e-3)
    assert_allclose(gd.intercept_, exact.intercept_, atol=1e-3)


def test_linear_regression_gd_loss_decreases(regression_data):
    X, y = regression_data
    gd = LinearRegression(solver="gd", lr=0.05, max_iter=500).fit(X, y)
    diffs = np.diff(gd.loss_history_)
    assert np.all(diffs <= 1e-9)


def test_ridge_matches_sklearn(regression_data):
    X, y = regression_data
    ours = Ridge(alpha=3.7).fit(X, y)
    theirs = sklm.Ridge(alpha=3.7).fit(X, y)
    assert_allclose(ours.coef_, theirs.coef_, atol=1e-6)
    assert_allclose(ours.intercept_, theirs.intercept_, atol=1e-6)


def test_lasso_matches_sklearn():
    X, y = make_regression(
        n_samples=150, n_features=8, n_informative=3, noise=1.0, random_state=1
    )
    ours = Lasso(alpha=0.5, max_iter=5000, tol=1e-12).fit(X, y)
    theirs = sklm.Lasso(alpha=0.5, max_iter=50000, tol=1e-12).fit(X, y)
    assert_allclose(ours.coef_, theirs.coef_, atol=1e-3)
    assert_allclose(ours.intercept_, theirs.intercept_, atol=1e-3)


def test_lasso_produces_exact_zeros():
    X, y = make_regression(
        n_samples=150, n_features=8, n_informative=3, noise=1.0, random_state=1
    )
    coef = Lasso(alpha=5.0, max_iter=5000).fit(X, y).coef_
    assert np.sum(coef == 0.0) >= 4  # only 3 informative features
    huge = Lasso(alpha=1e6, max_iter=5000).fit(X, y).coef_
    assert np.all(huge == 0.0)


def test_logistic_regression_close_to_sklearn():
    X, y = make_classification(n_samples=400, n_features=10, random_state=3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    ours = LogisticRegression(lr=0.5, max_iter=5000).fit(X_tr, y_tr)
    theirs = sklm.LogisticRegression(C=1e6, max_iter=5000).fit(X_tr, y_tr)
    ours_auc = roc_auc_score(y_te, ours.predict_proba(X_te)[:, 1])
    theirs_auc = roc_auc_score(y_te, theirs.predict_proba(X_te)[:, 1])
    assert abs(ours_auc - theirs_auc) < 0.01


def test_logistic_regression_interface():
    X, y = make_classification(n_samples=100, n_features=4, random_state=4)
    clf = LogisticRegression(lr=0.5, max_iter=500).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (100, 2)
    assert_allclose(proba.sum(axis=1), 1.0)
    assert set(clf.predict(X)) <= set(clf.classes_)
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, np.zeros(100))  # single class


def test_unfitted_predict_raises():
    with pytest.raises(NotFittedError):
        Ridge().predict(np.zeros((3, 2)))


def test_repr_and_get_params():
    est = Ridge(alpha=2.0)
    assert repr(est) == "Ridge(alpha=2.0)"
    assert est.get_params() == {"alpha": 2.0}
