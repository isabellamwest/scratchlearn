"""Property-based tests: mathematical invariants checked with Hypothesis.

Each test states an invariant the implementation must satisfy for every
valid input, then lets Hypothesis search for a counterexample. Expected
values are never produced by reimplementing the code under test: they come
from mathematical identities (KKT conditions, convexity guarantees,
orthogonality, concavity of entropy), from central finite differences, or
from cross-checking two independent algorithms against each other.

deadline=None throughout: these tests do real linear algebra (and, for the
gradient check, a full finite-difference sweep over every parameter), so
wall time varies with the drawn problem size and with BLAS threading.
Hypothesis's default 200 ms per-example deadline turns slow-but-correct
examples into flaky DeadlineExceeded errors; total runtime is bounded by
max_examples instead.
"""

import numpy as np
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

from scratchlearn.decomposition import PCA
from scratchlearn.linear_model import Lasso, LinearRegression
from scratchlearn.model_selection import train_test_split
from scratchlearn.neural import MLPClassifier, _softmax
from scratchlearn.tree import information_gain


def _floats(bound: float) -> st.SearchStrategy[float]:
    """Finite float64 values in [-bound, bound]."""
    return st.floats(
        min_value=-bound, max_value=bound, allow_nan=False, allow_infinity=False, width=64
    )


# ---------------------------------------------------------------------------
# MLPClassifier: analytic gradients vs central finite differences
# ---------------------------------------------------------------------------


@st.composite
def _mlp_cases(draw):
    n = draw(st.integers(min_value=2, max_value=5))
    p = draw(st.integers(min_value=1, max_value=3))
    k = draw(st.integers(min_value=2, max_value=3))
    hidden = draw(st.integers(min_value=1, max_value=4))
    X = draw(hnp.arrays(np.float64, (n, p), elements=_floats(2.0)))
    labels = draw(hnp.arrays(np.int64, (n,), elements=st.integers(0, k - 1)))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return X, labels, k, hidden, seed


@settings(max_examples=50, deadline=None)
@given(_mlp_cases())
def test_mlp_gradients_match_central_differences(case):
    """Backprop must agree with finite differences for every architecture and input.

    Generalises the fixed-shape check in test_neural.py to generated batch
    sizes, widths, class counts, weights and inputs.
    """
    X, labels, k, hidden, seed = case
    net = MLPClassifier(hidden_layer_sizes=(hidden,))
    net._initialise(X.shape[1], k, np.random.default_rng(seed))
    Y = np.eye(k)[labels]

    zs, activations = net._forward(X)
    # Central differences are invalid within eps of a ReLU kink (the loss is
    # not differentiable there) and uninformative inside the log-clip region
    # of the loss, so such draws are discarded. This constrains where the
    # check runs, not what it asserts.
    assume(all(np.abs(z).min() > 1e-3 for z in zs[:-1]))
    assume((activations[-1] * Y).sum(axis=1).min() > 1e-6)

    _, grads_w, grads_b = net._gradients(X, Y)
    eps = 1e-5
    for params, grads in ((net.weights_, grads_w), (net.biases_, grads_b)):
        for layer, grad in zip(params, grads):
            numeric = np.empty_like(layer)
            for idx in np.ndindex(layer.shape):
                original = layer[idx]
                layer[idx] = original + eps
                up = net._loss(X, Y)
                layer[idx] = original - eps
                down = net._loss(X, Y)
                layer[idx] = original
                numeric[idx] = (up - down) / (2.0 * eps)
            # Central differences with eps=1e-5 carry O(eps^2) ~ 1e-10
            # truncation error plus ~1e-11 cancellation error (machine epsilon
            # over 2*eps at loss scale ~1), so the numeric estimate is good to
            # ~1e-9. rtol 1e-5 / atol 1e-8 sit two orders above that noise,
            # while a wrong term in the chain rule shows up at the scale of
            # the gradient itself.
            assert_allclose(grad, numeric, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Lasso: subgradient (KKT) optimality of the fitted coefficients
# ---------------------------------------------------------------------------


@st.composite
def _lasso_cases(draw):
    n = draw(st.integers(min_value=5, max_value=25))
    p = draw(st.integers(min_value=1, max_value=5))
    X = draw(hnp.arrays(np.float64, (n, p), elements=_floats(5.0)))
    y = draw(hnp.arrays(np.float64, (n,), elements=_floats(5.0)))
    alpha = draw(st.floats(0.05, 2.0, allow_nan=False, allow_infinity=False, width=64))
    return X, y, alpha


@settings(max_examples=25, deadline=None)
@given(_lasso_cases())
def test_lasso_solution_satisfies_kkt_conditions(case):
    """The fitted coefficients satisfy lasso subgradient optimality.

    For ||y - Xw - b||^2 / (2n) + alpha * ||w||_1 the minimiser must have
    correlation c_j = x_j . resid / n equal to alpha * sign(w_j) on every
    active coefficient and |c_j| <= alpha on every zero one. This pins down
    the answer without reimplementing coordinate descent.
    """
    X, y, alpha = case
    model = Lasso(alpha=alpha, max_iter=20_000, tol=1e-12).fit(X, y)
    resid = y - model.predict(X)
    corr = X.T @ resid / len(y)
    # The fit stops once the largest coefficient update in a sweep is below
    # 1e-12, which bounds the stationarity residual by roughly
    # max_j ||x_j||^2 / n * 1e-12 ~ 1e-10 for these bounded inputs. 1e-6 sits
    # four orders above that slack and four orders below the O(alpha) >= 0.05
    # violation a broken soft-threshold or column-norm term would produce.
    tol = 1e-6
    active = model.coef_ != 0.0
    assert_allclose(corr[active], alpha * np.sign(model.coef_[active]), rtol=0.0, atol=tol)
    assert np.all(np.abs(corr[~active]) <= alpha + tol)


# ---------------------------------------------------------------------------
# LinearRegression(solver="gd"): monotone descent on a convex quadratic
# ---------------------------------------------------------------------------


@st.composite
def _regression_cases(draw):
    n = draw(st.integers(min_value=3, max_value=20))
    p = draw(st.integers(min_value=1, max_value=4))
    X = draw(hnp.arrays(np.float64, (n, p), elements=_floats(10.0)))
    y = draw(hnp.arrays(np.float64, (n,), elements=_floats(10.0)))
    return X, y


@settings(max_examples=50, deadline=None)
@given(_regression_cases())
def test_gradient_descent_loss_is_monotone_on_least_squares(case):
    """Every gradient-descent step on the (convex, quadratic) MSE lowers the loss."""
    X, y = case
    design = np.c_[np.ones(len(X)), X]
    # For a convex quadratic, gradient descent with step size below 1/L
    # (L = largest Hessian eigenvalue, here 2 * lambda_max(D'D) / n) decreases
    # the loss at every iteration -- a theorem, not a heuristic. Computing L
    # from the data is choosing a valid hyperparameter, not reimplementing
    # the solver. The intercept column keeps L >= 2, so lr is always finite.
    lipschitz = 2.0 * np.linalg.eigvalsh(design.T @ design)[-1] / len(X)
    model = LinearRegression(solver="gd", lr=0.9 / lipschitz, max_iter=60, tol=0.0).fit(X, y)
    history = np.asarray(model.loss_history_)
    # Exact arithmetic guarantees a non-increasing sequence; each recorded
    # loss carries only ~1e-14 relative rounding, so 1e-9 * scale allows a
    # 10^5 margin for noise while still catching real divergence, which grows
    # the loss geometrically.
    assert np.all(np.diff(history) <= 1e-9 * max(1.0, history[0]))


# ---------------------------------------------------------------------------
# _softmax: shift invariance, normalisation, overflow safety
# ---------------------------------------------------------------------------


@st.composite
def _softmax_cases(draw):
    n = draw(st.integers(min_value=1, max_value=6))
    k = draw(st.integers(min_value=2, max_value=6))
    z = draw(hnp.arrays(np.float64, (n, k), elements=_floats(1e5)))
    shift = draw(hnp.arrays(np.float64, (n, 1), elements=_floats(1e5)))
    return z, shift


@settings(deadline=None)
@given(_softmax_cases())
def test_softmax_shift_invariant_and_normalised(case):
    """softmax(z + c) == softmax(z) per row; rows sum to 1; values stay in [0, 1].

    Inputs deliberately range up to |1e5|, far beyond exp's overflow point at
    ~709 -- an exception to the bounded-magnitude rule, because the property
    under test *is* the numerical guard (row-max subtraction). A naive exp
    would return inf or nan here.
    """
    z, shift = case
    probs = _softmax(z)
    assert np.isfinite(probs).all()
    assert probs.min() >= 0.0
    # k <= 6 normalised terms each carry ~1e-16 relative rounding, so row
    # sums are exact to ~1e-15; atol 1e-12 leaves a 1000x margin while a
    # genuine normalisation bug is off by O(1).
    assert_allclose(probs.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    # Rounding z + c perturbs each exponent by at most ~eps * 2e5 ~ 4e-11,
    # which moves probabilities by the same relative amount; rtol 1e-8 gives
    # a 1000x margin. atol absorbs values driven into (sub)normal underflow.
    assert_allclose(_softmax(z + shift), probs, rtol=1e-8, atol=1e-12)


# ---------------------------------------------------------------------------
# information_gain: non-negativity for any true partition
# ---------------------------------------------------------------------------


@st.composite
def _split_cases(draw):
    labels = draw(st.lists(st.integers(0, 3), min_size=1, max_size=40))
    mask = draw(st.lists(st.booleans(), min_size=len(labels), max_size=len(labels)))
    parent = np.asarray(labels)
    mask_arr = np.asarray(mask, dtype=bool)
    return parent, parent[mask_arr], parent[~mask_arr]


@settings(deadline=None)
@given(_split_cases())
def test_information_gain_is_nonnegative(case):
    """Splitting can never increase weighted entropy (concavity of entropy).

    Empty left or right partitions are deliberately included: the function
    documents no precondition, and a tree search will eventually evaluate a
    degenerate split.
    """
    parent, left, right = case
    ig = information_gain(parent, left, right)
    assert np.isfinite(ig)
    # Entropies of <= 40 integer-counted labels carry ~1e-15 rounding, so
    # only rounding-scale negativity is tolerated; a wrong weighting or
    # count produces violations at O(0.1) or worse.
    assert ig >= -1e-12


# ---------------------------------------------------------------------------
# PCA: round-trip at full rank, and eigen vs SVD as mutual oracles
# ---------------------------------------------------------------------------


@st.composite
def _pca_cases(draw):
    n = draw(st.integers(min_value=2, max_value=12))
    p = draw(st.integers(min_value=1, max_value=min(4, n)))
    return draw(hnp.arrays(np.float64, (n, p), elements=_floats(100.0)))


@settings(max_examples=75, deadline=None)
@given(_pca_cases())
def test_pca_full_rank_round_trip(X):
    """With all components kept (n >= p), inverse_transform undoes transform.

    The reconstruction (X - mu) V'V + mu is exact whenever V is a full p x p
    orthonormal basis, which n >= p guarantees -- including for degenerate
    (constant or rank-deficient) inputs.
    """
    pca = PCA().fit(X)
    recon = pca.inverse_transform(pca.transform(X))
    # Two orthogonal matmuls plus centring accumulate rounding of order
    # p * eps * |X| ~ 1e-13 at this input scale (|X| <= 100); rtol 1e-9 /
    # atol 1e-7 leave ~10^4 margin above that noise and sit nine orders
    # below the O(|X|) distortion a real projection bug causes.
    assert_allclose(recon, X, rtol=1e-9, atol=1e-7)


@settings(max_examples=75, deadline=None)
@given(_pca_cases())
def test_pca_eigen_and_svd_agree_on_the_spectrum(X):
    """Two independent algorithms must find the same explained variances.

    The spectrum is compared rather than the components because repeated
    eigenvalues have no unique eigenbasis, so components may legitimately
    differ between methods; the eigenvalues may not.
    """
    ev_eigen = PCA(method="eigen").fit(X).explained_variance_
    ev_svd = PCA(method="svd").fit(X).explained_variance_
    # eigh and svd are both backward stable: each eigenvalue is accurate to
    # ~eps * ||cov|| ~ 1e-10 at this input scale. The tolerance scales with
    # total variance, sitting >= 10^3 above that noise but far below the
    # spectrum-scale shift caused by a wrong centring or 1/(n-1) factor.
    scale = 1.0 + ev_svd.sum()
    assert_allclose(ev_eigen, ev_svd, rtol=1e-7, atol=1e-8 * scale)


@settings(max_examples=75, deadline=None)
@given(_pca_cases())
@example(np.zeros((3, 2)))  # constant input: zero total variance, the 0/0 = NaN case
def test_pca_variance_ratio_is_finite_and_normalised(X):
    """explained_variance_ratio_ stays finite, non-negative and totals 1 (or 0).

    A constant input has zero total variance, so the ratio must degrade to 0
    rather than 0/0 = NaN; otherwise the kept components' ratios sum to 1.
    """
    ratio = PCA().fit(X).explained_variance_ratio_
    assert np.isfinite(ratio).all()
    assert (ratio >= 0.0).all()
    # exactly 0 when there is no variance to explain, else normalised to 1;
    # k <= 4 normalised terms carry ~1e-16 rounding, so 1e-9 is ample slack.
    total = float(ratio.sum())
    assert total == 0.0 or abs(total - 1.0) <= 1e-9


# ---------------------------------------------------------------------------
# train_test_split: exact aligned partition, exact stratified quotas
# ---------------------------------------------------------------------------


@st.composite
def _split_requests(draw):
    n = draw(st.integers(min_value=3, max_value=40))
    test_size = draw(st.floats(0.1, 0.6, allow_nan=False, allow_infinity=False, width=64))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    assume(0 < int(np.ceil(n * test_size)) < n)
    return n, test_size, seed


@settings(deadline=None)
@given(_split_requests())
def test_train_test_split_is_an_aligned_partition(case):
    """Train + test is exactly a permutation of the input, with pairs intact.

    X is an index-tagged array rather than a Hypothesis-generated float
    matrix because the property under test is pure index bookkeeping:
    tagging row i with value i makes lost, duplicated or misaligned rows
    detectable exactly, with no tolerance needed.
    """
    n, test_size, seed = case
    X = np.arange(n, dtype=np.float64).reshape(-1, 1)
    y = np.arange(n)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    assert len(X_te) == int(np.ceil(n * test_size))
    assert len(X_tr) == n - len(X_te)
    # every (X_i, y_i) pair survives the shuffle together
    assert np.array_equal(X_tr[:, 0], y_tr.astype(np.float64))
    assert np.array_equal(X_te[:, 0], y_te.astype(np.float64))
    # and the two halves partition the original rows exactly
    assert np.array_equal(np.sort(np.concatenate([y_tr, y_te])), np.arange(n))


@st.composite
def _stratified_requests(draw):
    labels = draw(st.lists(st.integers(0, 2), min_size=4, max_size=40))
    test_size = draw(st.floats(0.1, 0.6, allow_nan=False, allow_infinity=False, width=64))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    assume(0 < int(np.ceil(len(labels) * test_size)) < len(labels))
    return np.asarray(labels), test_size, seed


@settings(deadline=None)
@given(_stratified_requests())
def test_stratified_split_preserves_class_proportions(case):
    """Each class's test-set count is within one sample of exact proportionality."""
    y, test_size, seed = case
    n = len(y)
    X = np.arange(n, dtype=np.float64).reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    n_test = int(np.ceil(n * test_size))
    assert len(y_te) == n_test
    assert np.array_equal(
        np.sort(np.concatenate([X_tr[:, 0], X_te[:, 0]])), np.arange(n, dtype=np.float64)
    )
    for cls in np.unique(y):
        exact = (y == cls).sum() * n_test / n
        # largest-remainder apportionment can only floor or ceil each exact
        # quota, so drift of a full sample or more means the quota logic broke
        assert abs((y_te == cls).sum() - exact) < 1.0
