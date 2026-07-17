# scratchlearn

[![CI](https://github.com/isabellamwest/scratchlearn/actions/workflows/ci.yml/badge.svg)](https://github.com/isabellamwest/scratchlearn/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/coverage-97%25-brightgreen)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![licence](https://img.shields.io/badge/licence-MIT-lightgrey)](LICENSE)

I built this library to understand how scikit-learn algorithms work from first principles. Each algorithm is implemented using only NumPy, with the mathematical derivation documented alongside the code and unit tests verifying agreement with scikit-learn.
I kept the library small so each implementation stays readable. Each estimator is implemented as a single class, prioritising clarity over optimisation, with the underlying theory explained in the accompanying notes.

## What's inside

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
| k-nearest neighbours | [`neighbors.py`](src/scratchlearn/neighbors.py) | — | [tests](tests/test_neighbors.py) |
| MLP with hand-written backprop | [`neural.py`](src/scratchlearn/neural.py) | [notes](docs/derivations/mlp_backprop.md) | [tests](tests/test_neural.py) |
| Metrics (accuracy, P/R/F1, confusion matrix, ROC/AUC, RMSE, MAE, R²) | [`metrics.py`](src/scratchlearn/metrics.py) | — | [tests](tests/test_metrics.py) |
| Splitting, k-fold CV, grid search | [`model_selection.py`](src/scratchlearn/model_selection.py) | — | [tests](tests/test_model_selection.py) |

The [examples/](examples/) notebooks show the estimators on real problems, including the MLP
reaching 96% test accuracy on the scikit-learn digits after a few seconds of CPU training.

## Quickstart

```bash
git clone https://github.com/isabellamwest/scratchlearn.git
cd scratchlearn
pip install -e .
```

The API mirrors scikit-learn's estimator interface (fit, predict, score, get_params, etc.).

```python
from scratchlearn.linear_model import Ridge
from scratchlearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = Ridge(alpha=1.0).fit(X_train, y_train)
print(model.score(X_test, y_test))            # R^2
print(cross_val_score(model, X_train, y_train, cv=5))
```

To run the test suite (scikit-learn is needed for the parity comparisons):

```bash
pip install -r requirements-dev.txt
pytest
```

## Design notes

- **One contract for every estimator.** `base.py` defines `BaseEstimator` (introspective
  `get_params`, readable `repr`) plus scoring mixins, mirroring scikit-learn's
  template-method design: hyperparameters in `__init__`, learned attributes get a trailing
  underscore, `fit` returns `self`, and `predict` before `fit` raises `NotFittedError`.
- **No explicit matrix inverses.** Ridge uses `np.linalg.solve` and OLS uses `lstsq`;
  forming `inv(X'X)` squares the condition number for no benefit.
- **Log-space where it matters.** The GMM E-step works entirely with log-densities through a
  hand-written log-sum-exp, because 64-dimensional Gaussian densities underflow long before
  clustering gets hard. The derivation notes show the same trick stabilising softmax and the
  logistic loss.
- **Gradients you can trust.** The MLP's backprop is verified against central finite
  differences to a relative error of 1e-6, and the GMM test asserts the EM log-likelihood is
  monotonically non-decreasing — the two I check first when a refactor breaks something.
- **Vectorised NumPy throughout.** A Python loop over samples is treated as a bug, with the
  documented exceptions of tree recursion and the EM/Lloyd outer iterations.

## Limitations

This is an educational library and does not try to compete with scikit-learn: there is no
sparse-matrix support, no multiclass logistic regression, no tree pruning, and the
implementations favour clarity over speed. Numerical parity with scikit-learn is tested on
dense, well-behaved data of moderate size — which is exactly the regime the library is
meant for.
