from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.loader import load_dataset
from src.data.schema import MFCC_COLUMNS


RESULTS_DIR = Path("results/eda")


def main() -> None:
    df = load_dataset()

    features = df[MFCC_COLUMNS]

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_feature_variances(features)
    plot_feature_boxplots(features)
    plot_correlation_matrix(features)


def plot_feature_variances(features) -> None:
    variances = features.var()

    plt.figure(figsize=(12, 6))

    plt.bar(
        MFCC_COLUMNS,
        variances,
    )

    plt.xticks(
        rotation=90
    )

    plt.ylabel("Variance")
    plt.title("MFCC Feature Variance")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "mfcc_feature_variance.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_feature_boxplots(features) -> None:
    plt.figure(figsize=(14, 7))

    plt.boxplot(
        [
            features[column]
            for column in MFCC_COLUMNS
        ],
        tick_labels=MFCC_COLUMNS,
    )

    plt.xticks(
        rotation=90
    )

    plt.ylabel("Feature value")
    plt.title("MFCC Feature Distributions")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "mfcc_feature_boxplots.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_correlation_matrix(features) -> None:
    correlation_matrix = (
        features.corr().to_numpy()
    )

    plt.figure(figsize=(12, 10))

    image = plt.imshow(
        correlation_matrix,
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Pearson correlation",
    )

    positions = np.arange(
        len(MFCC_COLUMNS)
    )

    plt.xticks(
        positions,
        MFCC_COLUMNS,
        rotation=90,
    )

    plt.yticks(
        positions,
        MFCC_COLUMNS,
    )

    plt.title(
        "MFCC Feature Correlation Matrix"
    )

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "mfcc_correlation_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()