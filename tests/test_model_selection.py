import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression

from scratchlearn.exceptions import NotFittedError
from scratchlearn.linear_model import Ridge
from scratchlearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    clone,
    cross_val_score,
    train_test_split,
)


def test_kfold_partitions_all_indices():
    X = np.arange(46).reshape(23, 2)
    seen = []
    for train, test in KFold(n_splits=5).split(X):
        assert np.intersect1d(train, test).size == 0
        assert len(train) + len(test) == 23
        seen.append(test)
    assert_allclose(np.sort(np.concatenate(seen)), np.arange(23))


def test_kfold_shuffle_is_reproducible():
    X = np.zeros((20, 1))
    a = [t.tolist() for _, t in KFold(5, shuffle=True, random_state=3).split(X)]
    b = [t.tolist() for _, t in KFold(5, shuffle=True, random_state=3).split(X)]
    assert a == b


def test_stratified_kfold_keeps_proportions():
    y = np.array([0] * 60 + [1] * 30 + [2] * 10)
    X = np.zeros((100, 1))
    for train, test in StratifiedKFold(n_splits=5).split(X, y):
        assert np.sort(np.concatenate([train, test])).tolist() == list(range(100))
        _, counts = np.unique(y[test], return_counts=True)
        assert counts.tolist() == [12, 6, 2]


def test_train_test_split_basic():
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
    assert X_te.shape == (25, 2) and X_tr.shape == (75, 2)
    assert np.intersect1d(y_tr, y_te).size == 0
    again = train_test_split(X, y, test_size=0.25, random_state=0)
    assert_allclose(again[1], X_te)


def test_train_test_split_stratified():
    y = np.array([0] * 80 + [1] * 20)
    X = np.zeros((100, 3))
    _, _, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    _, counts = np.unique(y_te, return_counts=True)
    assert counts.tolist() == [16, 4]


def test_clone_returns_unfitted_copy():
    est = Ridge(alpha=2.5)
    copy = clone(est)
    assert copy is not est
    assert copy.get_params() == {"alpha": 2.5}


def test_cross_val_score_matches_manual_loop():
    X, y = make_regression(n_samples=80, n_features=5, noise=2.0, random_state=0)
    scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=4)
    manual = []
    for train, test in KFold(n_splits=4).split(X):
        manual.append(Ridge(alpha=1.0).fit(X[train], y[train]).score(X[test], y[test]))
    assert_allclose(scores, manual)
    assert len(scores) == 4


def test_cross_val_score_with_named_scorer():
    X, y = make_regression(n_samples=80, n_features=5, noise=2.0, random_state=0)
    scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=4, scoring="rmse")
    assert np.all(scores > 0)


def test_grid_search_finds_reasonable_alpha():
    X, y = make_regression(n_samples=120, n_features=8, noise=10.0, random_state=2)
    search = GridSearchCV(Ridge(), {"alpha": [0.01, 1.0, 100.0]}, cv=4)
    search.fit(X, y)
    assert search.best_params_["alpha"] in (0.01, 1.0, 100.0)
    assert len(search.cv_results_["mean_score"]) == 3
    assert search.best_score_ == max(search.cv_results_["mean_score"])
    assert search.predict(X).shape == y.shape


def test_grid_search_unfitted_raises():
    with pytest.raises(NotFittedError):
        GridSearchCV(Ridge(), {"alpha": [1.0]}).predict(np.zeros((3, 2)))
