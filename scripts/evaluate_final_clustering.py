from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.evaluation.cluster_selection import fit_best_kmeans


FEATURES_PATH = Path(
    "data/processed/features_standardized.csv"
)

LABELS_PATH = Path(
    "data/processed/labels.csv"
)

RESULTS_DIR = Path(
    "results/final_clustering"
)

N_CLUSTERS = 3
N_INIT = 10


def main() -> None:
    features_df = pd.read_csv(
        FEATURES_PATH
    )

    labels_df = pd.read_csv(
        LABELS_PATH
    )

    X = features_df.to_numpy()

    model, best_seed = fit_best_kmeans(
        X=X,
        n_clusters=N_CLUSTERS,
        n_init=N_INIT,
    )

    assert model.labels_ is not None
    assert model.centroids_ is not None
    assert model.inertia_ is not None
    assert model.n_iterations_ is not None

    cluster_labels = model.labels_

    silhouette = silhouette_score(
        X,
        cluster_labels,
    )

    cluster_sizes = np.bincount(
        cluster_labels,
        minlength=N_CLUSTERS,
    )

    results_df = labels_df.copy()

    results_df.insert(
        0,
        "Cluster",
        cluster_labels,
    )

    centroids_df = pd.DataFrame(
        model.centroids_,
        columns=features_df.columns,
    )

    centroids_df.insert(
        0,
        "Cluster",
        range(N_CLUSTERS),
    )

    family_ari = adjusted_rand_score(
        labels_df["Family"],
        cluster_labels,
    )

    family_nmi = normalized_mutual_info_score(
        labels_df["Family"],
        cluster_labels,
    )

    family_distribution = pd.crosstab(
        results_df["Cluster"],
        results_df["Family"],
    )

    genus_distribution = pd.crosstab(
        results_df["Cluster"],
        results_df["Genus"],
    )

    species_distribution = pd.crosstab(
        results_df["Cluster"],
        results_df["Species"],
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

    genus_path = (
        RESULTS_DIR
        / "cluster_vs_genus.csv"
    )

    species_path = (
        RESULTS_DIR
        / "cluster_vs_species.csv"
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

    genus_distribution.to_csv(
        genus_path,
    )

    species_distribution.to_csv(
        species_path,
    )

    print("=== FINAL CUSTOM K-MEANS CLUSTERING ===")

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
        f"Iterations: {model.n_iterations_}"
    )

    print(
        f"Converged: {model.converged_}"
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
    print(genus_path)
    print(species_path)


if __name__ == "__main__":
    main()