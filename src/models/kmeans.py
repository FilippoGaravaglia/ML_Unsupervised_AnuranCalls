from __future__ import annotations

import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int,
        max_iterations: int = 300,
        tolerance: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")

        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")

        if tolerance < 0:
            raise ValueError("tolerance must be greater than or equal to 0")

        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iterations_: int | None = None

    def fit(self, X: np.ndarray) -> KMeans:
        X = self._validate_input(X)

        if self.n_clusters > X.shape[0]:
            raise ValueError(
                "n_clusters cannot be greater than the number of samples"
            )

        rng = np.random.default_rng(self.random_state)

        self.centroids_ = self._initialize_centroids(
            X,
            rng,
        )

        for iteration in range(1, self.max_iterations + 1):
            labels = self._assign_clusters(X)

            new_centroids = self._update_centroids(
                X,
                labels,
                rng,
            )

            centroid_shift = np.linalg.norm(
                new_centroids - self.centroids_,
                axis=1,
            )

            self.centroids_ = new_centroids
            self.labels_ = labels
            self.n_iterations_ = iteration

            if np.all(centroid_shift <= self.tolerance):
                break

        self.labels_ = self._assign_clusters(X)
        self.inertia_ = self._calculate_inertia(
            X,
            self.labels_,
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError(
                "The model must be fitted before calling predict"
            )

        X = self._validate_input(X)

        if X.shape[1] != self.centroids_.shape[1]:
            raise ValueError(
                "Input feature count does not match fitted centroids"
            )

        return self._assign_clusters(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)

        assert self.labels_ is not None

        return self.labels_.copy()

    def _initialize_centroids(
        self,
        X: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        indices = rng.choice(
            X.shape[0],
            size=self.n_clusters,
            replace=False,
        )

        return X[indices].copy()

    def _assign_clusters(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        assert self.centroids_ is not None

        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :],
            axis=2,
        )

        return np.argmin(
            distances,
            axis=1,
        )

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        new_centroids = np.empty(
            (
                self.n_clusters,
                X.shape[1],
            ),
            dtype=float,
        )

        for cluster_index in range(self.n_clusters):
            cluster_points = X[
                labels == cluster_index
            ]

            if cluster_points.shape[0] == 0:
                random_index = rng.integers(
                    0,
                    X.shape[0],
                )

                new_centroids[cluster_index] = X[
                    random_index
                ]
            else:
                new_centroids[cluster_index] = (
                    cluster_points.mean(axis=0)
                )

        return new_centroids

    def _calculate_inertia(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        assert self.centroids_ is not None

        assigned_centroids = self.centroids_[
            labels
        ]

        squared_distances = np.sum(
            (X - assigned_centroids) ** 2,
            axis=1,
        )

        return float(
            np.sum(squared_distances)
        )

    @staticmethod
    def _validate_input(
        X: np.ndarray,
    ) -> np.ndarray:
        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array"
            )

        if X.shape[0] == 0:
            raise ValueError(
                "X must contain at least one sample"
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one feature"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "X contains non-finite values"
            )

        return X