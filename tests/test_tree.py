import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.tree import DecisionTreeClassifier as SkTree

from scratchlearn.exceptions import NotFittedError
from scratchlearn.model_selection import train_test_split
from scratchlearn.tree import DecisionTreeClassifier, entropy, gini, information_gain


def test_entropy_values():
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy([1.0]) == pytest.approx(0.0)
    assert entropy([0.25] * 4) == pytest.approx(2.0)


def test_gini_values():
    assert gini([0.5, 0.5]) == pytest.approx(0.5)
    assert gini([1.0]) == pytest.approx(0.0)


def test_information_gain_of_perfect_split():
    parent = np.array([0] * 10 + [1] * 10)
    gain = information_gain(parent, parent[:10], parent[10:])
    assert gain == pytest.approx(1.0)  # parent entropy is 1 bit, children are pure


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_unlimited_tree_memorises_training_set(criterion):
    X, y = make_classification(n_samples=150, n_features=5, random_state=0)
    tree = DecisionTreeClassifier(criterion=criterion).fit(X, y)
    assert tree.score(X, y) == 1.0


def test_iris_accuracy_close_to_sklearn():
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    ours = DecisionTreeClassifier(max_depth=3).fit(X_tr, y_tr).score(X_te, y_te)
    theirs = SkTree(max_depth=3, random_state=0).fit(X_tr, y_tr).score(X_te, y_te)
    assert abs(ours - theirs) <= 0.03


def test_max_depth_limits_fit():
    X, y = make_classification(n_samples=200, n_features=6, random_state=1)
    stump = DecisionTreeClassifier(max_depth=1).fit(X, y)
    full = DecisionTreeClassifier().fit(X, y)
    assert stump.score(X, y) < full.score(X, y)
    root = stump.tree_
    assert root.left is not None and root.left.feature is None  # one split only


def test_invalid_criterion_raises():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="misclassification").fit(np.zeros((4, 2)), [0, 1, 0, 1])


def test_unfitted_predict_raises():
    with pytest.raises(NotFittedError):
        DecisionTreeClassifier().predict(np.zeros((3, 2)))
