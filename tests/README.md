# Tests

Run everything with `python -m pytest` from the repository root. Test dependencies are
listed in `requirements-dev.txt` (pytest, scikit-learn for datasets and comparison
oracles, hypothesis).

Each `test_<module>.py` file holds example-based unit tests for one module.
`test_properties.py` holds property-based tests written with
[Hypothesis](https://hypothesis.readthedocs.io/): each test states a mathematical
invariant that must hold for *every* valid input, and Hypothesis searches for a
counterexample, shrinking any failure to a minimal input. Failing examples are cached
in `.hypothesis/` (gitignored) and retried first on subsequent runs. Tolerances and
Hypothesis settings are justified in comments next to each test.

## Properties

| Invariant | Protects against |
| --- | --- |
| MLP backprop gradients match central finite differences for generated architectures, weights and batches | Any wrong term in the chain rule: a bad ReLU mask, a transposed matmul, a missing 1/n |
| Fitted Lasso coefficients satisfy the KKT (subgradient) conditions | Coordinate descent converging to a non-optimum: broken soft-thresholding, wrong column norm or 1/n scaling |
| Gradient descent on least squares decreases the loss at every step when lr < 1/L | A sign or scale error in the MSE gradient (monotone descent on a convex quadratic is a theorem) |
| Softmax is invariant to per-row shifts, rows sum to 1, outputs stay finite up to inputs of 1e5 | Overflow in exp, a broken row-max guard, normalising over the wrong axis |
| Information gain is non-negative for every true partition, including empty sides | Wrong split weighting or label counts (entropy concavity forbids negative gain) |
| PCA inverse_transform(transform(X)) reconstructs X exactly at full rank, degenerate inputs included | Non-orthonormal components, wrong centring in either direction |
| PCA eigen and SVD methods agree on the explained variances | Either algorithm computing a wrong spectrum: missed centring, wrong 1/(n-1) |
| train_test_split returns an exact partition with (X, y) pairs kept aligned, at the requested sizes | Lost or duplicated rows, X/y misalignment after shuffling |
| Stratified splits give every class floor or ceil of its exact test-set quota | Broken largest-remainder apportionment |

## Findings

All nine properties pass. The degenerate inputs Hypothesis generates still surfaced
one real issue: fitting `PCA` on zero-variance data silently produces NaN —
`PCA().fit(np.zeros((2, 1))).explained_variance_ratio_` is `[nan]`, from the 0/0 in
`variances[:k] / variances.sum()` (decomposition.py, with a RuntimeWarning as the only
symptom). A constant matrix passes input validation, so the NaN propagates silently.
Unfixed for now; the options are returning 0 for the ratio or raising a clear error.

An earlier suspicion that `information_gain` would divide by zero on an empty
partition turned out to be unfounded: NumPy's elementwise division of an empty array
is a no-op, so an empty side contributes exactly zero entropy and the property holds.
