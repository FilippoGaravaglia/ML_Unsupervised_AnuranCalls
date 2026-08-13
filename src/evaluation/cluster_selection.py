from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_score

from src.models.kmeans import KMeans


@dataclass(frozen=True)
class ClusterSelectionResult:
    n_clusters: int
    inertia: float
    silhouette: float
    iterations: int
    best_seed: int


def fit_best_kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    max_iterations: int = 300,
    tolerance: float = 1e-4,
) -> tuple[KMeans, int]:
    if n_init <= 0:
        raise ValueError(
            "n_init must be greater than 0"
        )

    best_model: KMeans | None = None
    best_seed: int | None = None
    best_inertia = float("inf")

    for seed in range(n_init):
        model = KMeans(
            n_clusters=n_clusters,
            max_iterations=max_iterations,
            tolerance=tolerance,
            random_state=seed,
        )

        model.fit(X)

        assert model.inertia_ is not None

        if model.inertia_ < best_inertia:
            best_model = model
            best_seed = seed
            best_inertia = model.inertia_

    assert best_model is not None
    assert best_seed is not None

    return best_model, best_seed


def evaluate_k_values(
    X: np.ndarray,
    k_values: range,
    n_init: int = 10,
    silhouette_sample_size: int = 2000,
    silhouette_random_state: int = 42,
) -> list[ClusterSelectionResult]:
    results = []

    for n_clusters in k_values:
        model, best_seed = fit_best_kmeans(
            X=X,
            n_clusters=n_clusters,
            n_init=n_init,
        )

        assert model.labels_ is not None
        assert model.inertia_ is not None
        assert model.n_iterations_ is not None

        sample_size = min(
            silhouette_sample_size,
            X.shape[0],
        )

        silhouette = silhouette_score(
            X,
            model.labels_,
            sample_size=sample_size,
            random_state=silhouette_random_state,
        )

        results.append(
            ClusterSelectionResult(
                n_clusters=n_clusters,
                inertia=model.inertia_,
                silhouette=float(silhouette),
                iterations=model.n_iterations_,
                best_seed=best_seed,
            )
        )

    return results