import numpy as np
import pytest

from src.models.kmeans import KMeans


def test_kmeans_finds_two_well_separated_clusters() -> None:
    X = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [8.0, 8.0],
            [8.1, 7.9],
            [7.9, 8.1],
        ]
    )

    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    labels = model.fit_predict(X)

    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_kmeans_computes_expected_centroids() -> None:
    X = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [8.0, 8.0],
            [8.1, 7.9],
            [7.9, 8.1],
        ]
    )

    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model.fit(X)

    assert model.centroids_ is not None

    centroids = model.centroids_

    centroids_sorted = centroids[
        np.argsort(centroids[:, 0])
    ]

    expected_centroids = np.array(
        [
            [1.0, 1.0],
            [8.0, 8.0],
        ]
    )

    np.testing.assert_allclose(
        centroids_sorted,
        expected_centroids,
        atol=1e-8,
    )


def test_kmeans_converges() -> None:
    X = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [0.9, 1.1],
            [8.0, 8.0],
            [8.1, 7.9],
            [7.9, 8.1],
        ]
    )

    model = KMeans(
        n_clusters=2,
        max_iterations=100,
        tolerance=1e-4,
        random_state=42,
    )

    model.fit(X)

    assert model.converged_ is True
    assert model.n_iterations_ is not None
    assert model.n_iterations_ <= 100
    assert len(model.centroid_shift_history_) == model.n_iterations_


def test_kmeans_inertia_is_non_negative() -> None:
    X = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [8.0, 8.0],
            [8.1, 7.9],
        ]
    )

    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model.fit(X)

    assert model.inertia_ is not None
    assert model.inertia_ >= 0.0


def test_predict_before_fit_raises_error() -> None:
    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    X = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        model.predict(X)


def test_invalid_number_of_clusters_raises_error() -> None:
    with pytest.raises(
        ValueError,
        match="n_clusters",
    ):
        KMeans(
            n_clusters=0
        )


def test_more_clusters_than_samples_raises_error() -> None:
    X = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
        ]
    )

    model = KMeans(
        n_clusters=3,
        random_state=42,
    )

    with pytest.raises(
        ValueError,
        match="number of samples",
    ):
        model.fit(X)


def test_invalid_input_dimension_raises_error() -> None:
    X = np.array(
        [
            1.0,
            2.0,
            3.0,
        ]
    )

    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    with pytest.raises(
        ValueError,
        match="2-dimensional",
    ):
        model.fit(X)


def test_non_finite_values_raise_error() -> None:
    X = np.array(
        [
            [1.0, 1.0],
            [np.nan, 2.0],
        ]
    )

    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        model.fit(X)


def test_predict_returns_cluster_for_each_sample() -> None:
    X_train = np.array(
        [
            [1.0, 1.0],
            [1.1, 0.9],
            [8.0, 8.0],
            [8.1, 7.9],
        ]
    )

    X_new = np.array(
        [
            [1.05, 1.0],
            [8.05, 8.0],
        ]
    )

    model = KMeans(
        n_clusters=2,
        random_state=42,
    )

    model.fit(X_train)

    predictions = model.predict(
        X_new
    )

    assert predictions.shape == (2,)
    assert predictions[0] != predictions[1]