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
        random_state=42,
    )

    labels = model.fit_predict(X)

    print("=== K-MEANS SMOKE TEST ===")

    print("\nInput data:")
    print(X)

    print("\nCluster labels:")
    print(labels)

    print("\nCentroids:")
    print(model.centroids_)

    print("\nInertia:")
    print(model.inertia_)

    print("\nIterations:")
    print(model.n_iterations_)


if __name__ == "__main__":
    main()