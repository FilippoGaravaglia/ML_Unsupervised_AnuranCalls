import numpy as np

from src.models.kmeans import KMeans


def main() -> None:
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

    print("=== K-MEANS CONVERGENCE ANALYSIS ===")

    print(f"\nConverged: {model.converged_}")
    print(f"Iterations: {model.n_iterations_}")

    print("\n=== CENTROID SHIFT HISTORY ===")

    for iteration, shift in enumerate(
        model.centroid_shift_history_,
        start=1,
    ):
        print(
            f"Iteration {iteration}: "
            f"{shift:.8f}"
        )

    print("\n=== FINAL CENTROIDS ===")
    print(model.centroids_)

    print("\n=== FINAL INERTIA ===")
    print(model.inertia_)


if __name__ == "__main__":
    main()