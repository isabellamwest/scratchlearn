import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from scratchlearn.cluster import GaussianMixture, KMeans
from scratchlearn.exceptions import NotFittedError


@pytest.fixture
def blobs():
    return make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=1)


def test_kmeans_recovers_blobs(blobs):
    X, y = blobs
    km = KMeans(n_clusters=4, random_state=0).fit(X)
    assert adjusted_rand_score(y, km.labels_) > 0.9


def test_kmeans_inertia_close_to_sklearn(blobs):
    X, _ = blobs
    ours = KMeans(n_clusters=4, random_state=0).fit(X)
    theirs = SkKMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    assert ours.inertia_ <= theirs.inertia_ * 1.05


def test_kmeans_predict_consistent_with_labels(blobs):
    X, _ = blobs
    km = KMeans(n_clusters=4, random_state=0).fit(X)
    assert_allclose(km.predict(X), km.labels_)
    assert km.cluster_centers_.shape == (4, 2)


def test_kmeans_unfitted_raises():
    with pytest.raises(NotFittedError):
        KMeans().predict(np.zeros((3, 2)))


@pytest.fixture
def separated_blobs():
    return make_blobs(
        n_samples=600, centers=[[-5, 0], [5, 0], [0, 8]], cluster_std=0.5, random_state=2
    )


def test_gmm_log_likelihood_never_decreases(separated_blobs):
    X, _ = separated_blobs
    gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
    history = np.asarray(gmm.log_likelihood_history_)
    assert history.size >= 2
    assert np.all(np.diff(history) >= -1e-9)


def test_gmm_recovers_means(separated_blobs):
    X, y = separated_blobs
    gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
    for cls in np.unique(y):
        sample_mean = X[y == cls].mean(axis=0)
        nearest = np.linalg.norm(gmm.means_ - sample_mean, axis=1).min()
        assert nearest < 0.1


def test_gmm_probabilities_are_valid(separated_blobs):
    X, y = separated_blobs
    gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
    resp = gmm.predict_proba(X)
    assert_allclose(resp.sum(axis=1), 1.0)
    assert_allclose(gmm.weights_.sum(), 1.0)
    assert adjusted_rand_score(y, gmm.predict(X)) > 0.95
    assert gmm.converged_
