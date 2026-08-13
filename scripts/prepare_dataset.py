import json
from pathlib import Path

import pandas as pd

from src.data.loader import load_dataset
from src.data.preprocessing import prepare_clustering_data
from src.data.schema import (
    ID_COLUMN,
    LABEL_COLUMNS,
    MFCC_COLUMNS,
)


OUTPUT_DIR = Path("data/processed")


def main() -> None:
    df = load_dataset()

    (
        standardized_features,
        labels,
        scaler,
    ) = prepare_clustering_data(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    features_path = (
        OUTPUT_DIR
        / "features_standardized.csv"
    )

    labels_path = (
        OUTPUT_DIR
        / "labels.csv"
    )

    metadata_path = (
        OUTPUT_DIR
        / "preprocessing_metadata.json"
    )

    standardized_features.to_csv(
        features_path,
        index=False,
    )

    labels.to_csv(
        labels_path,
        index=False,
    )

    metadata = {
        "n_observations": len(df),
        "n_features": len(MFCC_COLUMNS),
        "feature_columns": MFCC_COLUMNS,
        "excluded_from_clustering": (
            LABEL_COLUMNS
            + [ID_COLUMN]
        ),
        "scaler_mean": {
            feature: float(mean)
            for feature, mean
            in zip(
                MFCC_COLUMNS,
                scaler.mean_,
            )
        },
        "scaler_scale": {
            feature: float(scale)
            for feature, scale
            in zip(
                MFCC_COLUMNS,
                scaler.scale_,
            )
        },
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("=== PREPROCESSING COMPLETE ===")

    print(
        f"Observations: "
        f"{standardized_features.shape[0]}"
    )

    print(
        f"Clustering features: "
        f"{standardized_features.shape[1]}"
    )

    print(
        "\n=== EXCLUDED FROM CLUSTERING ==="
    )

    print(
        LABEL_COLUMNS
        + [ID_COLUMN]
    )

    print(
        "\n=== STANDARDIZED FEATURE MEANS ==="
    )

    print(
        standardized_features
        .mean()
        .round(6)
    )

    print(
        "\n=== STANDARDIZED FEATURE STANDARD DEVIATIONS ==="
    )

    print(
        standardized_features
        .std(ddof=0)
        .round(6)
    )

    print(
        "\n=== OUTPUT FILES ==="
    )

    print(features_path)
    print(labels_path)
    print(metadata_path)


if __name__ == "__main__":
    main()