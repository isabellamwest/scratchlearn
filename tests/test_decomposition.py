import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.decomposition import PCA as SkPCA

from scratchlearn.decomposition import PCA
from scratchlearn.exceptions import NotFittedError


@pytest.fixture
def correlated_data():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(200, 3))
    mixing = rng.normal(size=(3, 6))
    return latent @ mixing + rng.normal(scale=0.1, size=(200, 6))


def test_eigen_and_svd_agree(correlated_data):
    X = correlated_data
    eig = PCA(n_components=4, method="eigen").fit(X)
    svd = PCA(n_components=4, method="svd").fit(X)
    assert_allclose(eig.components_, svd.components_, atol=1e-8)
    assert_allclose(eig.explained_variance_, svd.explained_variance_, atol=1e-8)


def test_matches_sklearn_up_to_sign(correlated_data):
    X = correlated_data
    ours = PCA(n_components=3).fit(X)
    theirs = SkPCA(n_components=3).fit(X)
    assert_allclose(np.abs(ours.components_), np.abs(theirs.components_), atol=1e-6)
    assert_allclose(ours.explained_variance_ratio_, theirs.explained_variance_ratio_, atol=1e-6)


def test_reconstruction_error_decreases(correlated_data):
    X = correlated_data
    errors = []
    for k in range(1, X.shape[1] + 1):
        pca = PCA(n_components=k).fit(X)
        recon = pca.inverse_transform(pca.transform(X))
        errors.append(float(((X - recon) ** 2).sum()))
    assert np.all(np.diff(errors) <= 1e-8)
    assert errors[-1] < 1e-18  # all components: exact reconstruction


def test_transform_shape_and_fit_transform(correlated_data):
    X = correlated_data
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    assert Z.shape == (200, 2)
    assert_allclose(Z, pca.transform(X))


def test_unfitted_transform_raises():
    with pytest.raises(NotFittedError):
        PCA().transform(np.zeros((4, 2)))
