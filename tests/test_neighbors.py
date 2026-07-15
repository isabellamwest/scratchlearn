import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from scratchlearn.exceptions import NotFittedError
from scratchlearn.metrics import accuracy_score
from scratchlearn.model_selection import train_test_split
from scratchlearn.neighbors import KNNClassifier


@pytest.fixture
def iris_split():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_exact_match_with_sklearn_uniform(iris_split, k):
    X_tr, X_te, y_tr, y_te = iris_split
    ours = KNNClassifier(n_neighbors=k).fit(X_tr, y_tr).predict(X_te)
    theirs = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr).predict(X_te)
    assert_array_equal(ours, theirs)


@pytest.mark.parametrize("k", [3, 5])
def test_exact_match_with_sklearn_distance_weights(iris_split, k):
    X_tr, X_te, y_tr, y_te = iris_split
    ours = KNNClassifier(n_neighbors=k, weights="distance").fit(X_tr, y_tr).predict(X_te)
    theirs = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_tr, y_tr)
    assert_array_equal(ours, theirs.predict(X_te))


def test_score_is_accuracy(iris_split):
    X_tr, X_te, y_tr, y_te = iris_split
    clf = KNNClassifier(n_neighbors=5).fit(X_tr, y_tr)
    assert clf.score(X_te, y_te) == accuracy_score(y_te, clf.predict(X_te))


def test_too_many_neighbours_raises():
    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=10).fit(np.zeros((5, 2)), np.arange(5))


def test_invalid_weights_raises():
    with pytest.raises(ValueError):
        KNNClassifier(weights="gaussian").fit(np.zeros((5, 2)), np.arange(5))


def test_unfitted_predict_raises():
    with pytest.raises(NotFittedError):
        KNNClassifier().predict(np.zeros((3, 2)))
