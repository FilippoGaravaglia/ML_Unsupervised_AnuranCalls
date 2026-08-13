from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


FEATURES_PATH = Path(
    "data/processed/features_standardized.csv"
)

LABELS_PATH = Path(
    "data/processed/labels.csv"
)

CUSTOM_ASSIGNMENTS_PATH = Path(
    "results/final_clustering/cluster_assignments.csv"
)

RESULTS_DIR = Path(
    "results/sklearn"
)

N_CLUSTERS = 3
N_INIT = 10
MAX_ITERATIONS = 300
TOLERANCE = 1e-4


def fit_best_sklearn_kmeans(
    X: np.ndarray,
) -> tuple[KMeans, int]:
    best_model: KMeans | None = None
    best_seed: int | None = None
    best_inertia = float("inf")

    for seed in range(N_INIT):
        model = KMeans(
            n_clusters=N_CLUSTERS,
            init="random",
            n_init=1,
            max_iter=MAX_ITERATIONS,
            tol=TOLERANCE,
            random_state=seed,
            algorithm="lloyd",
        )

        model.fit(X)

        if model.inertia_ < best_inertia:
            best_model = model
            best_seed = seed
            best_inertia = model.inertia_

    assert best_model is not None
    assert best_seed is not None

    return best_model, best_seed


def main() -> None:
    features_df = pd.read_csv(
        FEATURES_PATH
    )

    labels_df = pd.read_csv(
        LABELS_PATH
    )

    custom_assignments_df = pd.read_csv(
        CUSTOM_ASSIGNMENTS_PATH
    )

    X = features_df.to_numpy()

    model, best_seed = fit_best_sklearn_kmeans(
        X
    )

    cluster_labels = model.labels_

    silhouette = silhouette_score(
        X,
        cluster_labels,
    )

    cluster_sizes = np.bincount(
        cluster_labels,
        minlength=N_CLUSTERS,
    )

    family_ari = adjusted_rand_score(
        labels_df["Family"],
        cluster_labels,
    )

    family_nmi = normalized_mutual_info_score(
        labels_df["Family"],
        cluster_labels,
    )

    custom_vs_sklearn_ari = adjusted_rand_score(
        custom_assignments_df["Cluster"],
        cluster_labels,
    )

    results_df = labels_df.copy()

    results_df.insert(
        0,
        "Cluster",
        cluster_labels,
    )

    centroids_df = pd.DataFrame(
        model.cluster_centers_,
        columns=features_df.columns,
    )

    centroids_df.insert(
        0,
        "Cluster",
        range(N_CLUSTERS),
    )

    family_distribution = pd.crosstab(
        results_df["Cluster"],
        results_df["Family"],
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    assignments_path = (
        RESULTS_DIR
        / "cluster_assignments.csv"
    )

    centroids_path = (
        RESULTS_DIR
        / "centroids.csv"
    )

    family_path = (
        RESULTS_DIR
        / "cluster_vs_family.csv"
    )

    metrics_path = (
        RESULTS_DIR
        / "metrics.csv"
    )

    results_df.to_csv(
        assignments_path,
        index=False,
    )

    centroids_df.to_csv(
        centroids_path,
        index=False,
    )

    family_distribution.to_csv(
        family_path,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "implementation": "scikit-learn",
                "k": N_CLUSTERS,
                "best_seed": best_seed,
                "iterations": model.n_iter_,
                "inertia": model.inertia_,
                "silhouette": silhouette,
                "family_ari": family_ari,
                "family_nmi": family_nmi,
                "custom_vs_sklearn_ari": custom_vs_sklearn_ari,
            }
        ]
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print("=== SCIKIT-LEARN K-MEANS ===")

    print(
        f"\nObservations: {X.shape[0]}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Selected K: {N_CLUSTERS}"
    )

    print(
        f"Initializations: {N_INIT}"
    )

    print(
        f"Best seed: {best_seed}"
    )

    print(
        f"Iterations: {model.n_iter_}"
    )

    print(
        f"Inertia: {model.inertia_:.6f}"
    )

    print(
        f"Silhouette score: {silhouette:.6f}"
    )

    print("\n=== CLUSTER SIZES ===")

    for cluster_index, size in enumerate(
        cluster_sizes
    ):
        print(
            f"Cluster {cluster_index}: "
            f"{size} observations"
        )

    print(
        "\n=== POST-HOC FAMILY COMPARISON ==="
    )

    print(
        f"Adjusted Rand Index: "
        f"{family_ari:.6f}"
    )

    print(
        f"Normalized Mutual Information: "
        f"{family_nmi:.6f}"
    )

    print(
        "\n=== CUSTOM VS SCIKIT-LEARN ==="
    )

    print(
        f"Adjusted Rand Index: "
        f"{custom_vs_sklearn_ari:.6f}"
    )

    print(
        "\n=== CLUSTER VS FAMILY ==="
    )

    print(
        family_distribution
    )

    print(
        "\n=== OUTPUT FILES ==="
    )

    print(assignments_path)
    print(centroids_path)
    print(family_path)
    print(metrics_path)


if __name__ == "__main__":
    main()