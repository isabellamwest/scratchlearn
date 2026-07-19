# scratchlearn

[![CI](https://github.com/isabellamwest/scratchlearn/actions/workflows/ci.yml/badge.svg)](https://github.com/isabellamwest/scratchlearn/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![licence](https://img.shields.io/badge/licence-MIT-lightgrey)](LICENSE)

scratchlearn is a small collection of machine learning algorithms implemented from scratch in NumPy. Each estimator is a single class written for readability, with the mathematical derivation documented alongside the code and tests that check the results against scikit-learn.

The project is a study of how the standard algorithms work. It prioritises clarity over performance and is not intended for production use.

## Contents

| Algorithm | Code | Derivation | Parity test |
| --- | --- | --- | --- |
| Linear regression (normal equations + gradient descent) | [`linear_model.py`](src/scratchlearn/linear_model.py) | [notes](docs/derivations/linear_regression.md) | [tests](tests/test_linear_model.py) |
| Ridge (closed form) | [`linear_model.py`](src/scratchlearn/linear_model.py) | [notes](docs/derivations/linear_regression.md) | [tests](tests/test_linear_model.py) |
| Lasso (coordinate descent, soft-thresholding) | [`linear_model.py`](src/scratchlearn/linear_model.py) | [notes](docs/derivations/linear_regression.md) | [tests](tests/test_linear_model.py) |
| Logistic regression (gradient descent, L2) | [`linear_model.py`](src/scratchlearn/linear_model.py) | [notes](docs/derivations/logistic_regression.md) | [tests](tests/test_linear_model.py) |
| PCA (eigendecomposition and SVD) | [`decomposition.py`](src/scratchlearn/decomposition.py) | [notes](docs/derivations/pca.md) | [tests](tests/test_decomposition.py) |
| k-means (k-means++ initialisation) | [`cluster.py`](src/scratchlearn/cluster.py) | [notes](docs/derivations/kmeans.md) | [tests](tests/test_cluster.py) |
| Gaussian mixture (full EM) | [`cluster.py`](src/scratchlearn/cluster.py) | [notes](docs/derivations/gmm_em.md) | [tests](tests/test_cluster.py) |
| Decision tree (gini and entropy) | [`tree.py`](src/scratchlearn/tree.py) | [notes](docs/derivations/decision_tree.md) | [tests](tests/test_tree.py) |
| k-nearest neighbours | [`neighbors.py`](src/scratchlearn/neighbors.py) | n/a | [tests](tests/test_neighbors.py) |
| MLP with hand-written backprop | [`neural.py`](src/scratchlearn/neural.py) | [notes](docs/derivations/mlp_backprop.md) | [tests](tests/test_neural.py) |
| Metrics (accuracy, P/R/F1, confusion matrix, ROC/AUC, RMSE, MAE, R²) | [`metrics.py`](src/scratchlearn/metrics.py) | n/a | [tests](tests/test_metrics.py) |
| Splitting, k-fold CV, grid search | [`model_selection.py`](src/scratchlearn/model_selection.py) | n/a | [tests](tests/test_model_selection.py) |

The [examples/](examples/) notebooks apply the estimators to real datasets. The MLP notebook reaches 96% test accuracy on the scikit-learn digits dataset after a few seconds of CPU training.

## Installation

```bash
git clone https://github.com/isabellamwest/scratchlearn.git
cd scratchlearn
pip install -e .
```

## Usage

The estimator interface follows scikit-learn: `fit`, `predict`, `score`, `get_params`.

```python
from scratchlearn.linear_model import Ridge
from scratchlearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = Ridge(alpha=1.0).fit(X_train, y_train)
print(model.score(X_test, y_test))            # R^2
print(cross_val_score(model, X_train, y_train, cv=5))
```

## Running the tests

scikit-learn is required for the parity comparisons.

```bash
pip install -r requirements-dev.txt
pytest
```

## Design notes

- All estimators share a common base. `base.py` defines `BaseEstimator` with `get_params` and a readable `repr`, plus scoring mixins. Hyperparameters are set in `__init__`, learned attributes end with a trailing underscore, `fit` returns `self`, and calling `predict` before `fit` raises `NotFittedError`.
- Linear systems are solved without forming matrix inverses. Ridge uses `np.linalg.solve` and ordinary least squares uses `lstsq`, which avoids squaring the condition number of the input.
- The GMM E-step runs in log space through a hand-written log-sum-exp, which prevents underflow in high-dimensional Gaussian densities. The same approach is used for softmax and the logistic loss, as shown in the derivation notes.
- The MLP backpropagation is checked against central finite differences to a relative error of 1e-6. The GMM test asserts that the EM log-likelihood is non-decreasing across iterations.
- [`tests/test_properties.py`](tests/test_properties.py) uses [Hypothesis](https://hypothesis.readthedocs.io/) to test invariants over generated inputs rather than fixed examples. This covers the finite-difference gradient check across architectures and batch sizes, the Lasso KKT optimality conditions, and PCA round-trips. See [tests/README.md](tests/README.md) for the full list.
- The implementations are vectorised, with the exception of tree recursion and the EM and Lloyd outer loops.

## Limitations

This is an educational library and is not a replacement for scikit-learn. There is no sparse-matrix support, no multiclass logistic regression, and no tree pruning. Numerical parity with scikit-learn is tested on dense, well-behaved data of moderate size.
