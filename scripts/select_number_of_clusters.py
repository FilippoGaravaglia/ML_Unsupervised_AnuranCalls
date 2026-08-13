from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.cluster_selection import (
    evaluate_k_values,
)


FEATURES_PATH = Path(
    "data/processed/features_standardized.csv"
)

RESULTS_DIR = Path(
    "results/cluster_selection"
)

K_VALUES = range(2, 11)

N_INIT = 10


def main() -> None:
    X = pd.read_csv(
        FEATURES_PATH
    ).to_numpy()

    print("=== K-MEANS CLUSTER SELECTION ===")

    print(
        f"Observations: {X.shape[0]}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"K values: {list(K_VALUES)}"
    )

    print(
        f"Initializations per K: {N_INIT}"
    )

    print(
        "\nRunning custom K-Means experiments..."
    )

    results = evaluate_k_values(
        X=X,
        k_values=K_VALUES,
        n_init=N_INIT,
    )

    results_df = pd.DataFrame(
        [
            {
                "k": result.n_clusters,
                "inertia": result.inertia,
                "silhouette": result.silhouette,
                "iterations": result.iterations,
                "best_seed": result.best_seed,
            }
            for result in results
        ]
    )

    print("\n=== RESULTS ===")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        RESULTS_DIR
        / "k_selection_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    plot_inertia(results_df)
    plot_silhouette(results_df)

    print("\n=== OUTPUT FILES ===")
    print(results_path)

    print(
        RESULTS_DIR
        / "elbow_curve.png"
    )

    print(
        RESULTS_DIR
        / "silhouette_curve.png"
    )


def plot_inertia(
    results_df: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        results_df["k"],
        results_df["inertia"],
        marker="o",
    )

    plt.xlabel(
        "Number of clusters (K)"
    )

    plt.ylabel(
        "Inertia"
    )

    plt.title(
        "Elbow Method — Custom K-Means"
    )

    plt.xticks(
        results_df["k"]
    )

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "elbow_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def plot_silhouette(
    results_df: pd.DataFrame,
) -> None:
    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        results_df["k"],
        results_df["silhouette"],
        marker="o",
    )

    plt.xlabel(
        "Number of clusters (K)"
    )

    plt.ylabel(
        "Silhouette score"
    )

    plt.title(
        "Silhouette Analysis — Custom K-Means"
    )

    plt.xticks(
        results_df["k"]
    )

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "silhouette_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


if __name__ == "__main__":
    main()