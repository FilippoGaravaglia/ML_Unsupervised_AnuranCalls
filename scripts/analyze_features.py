import numpy as np

from src.data.loader import load_dataset
from src.data.schema import MFCC_COLUMNS


CORRELATION_THRESHOLD = 0.90


def main() -> None:
    df = load_dataset()

    features = df[MFCC_COLUMNS]

    print("=== FEATURE MATRIX ===")
    print(f"Observations: {features.shape[0]}")
    print(f"Features: {features.shape[1]}")

    print("\n=== FEATURE STATISTICS ===")

    statistics = features.agg([
        "mean",
        "std",
        "min",
        "max",
    ]).T

    statistics["variance"] = features.var()

    print(
        statistics[
            ["mean", "std", "variance", "min", "max"]
        ].round(6)
    )

    print("\n=== FEATURES SORTED BY VARIANCE ===")

    variances = (
        features.var()
        .sort_values()
    )

    print(variances.round(6))

    print("\n=== CONSTANT FEATURES ===")

    constant_features = variances[
        np.isclose(variances, 0.0)
    ]

    if constant_features.empty:
        print("None")
    else:
        print(constant_features)

    print("\n=== HIGHLY CORRELATED FEATURE PAIRS ===")

    correlation_matrix = features.corr()

    highly_correlated_pairs = []

    for i in range(len(MFCC_COLUMNS)):
        for j in range(i + 1, len(MFCC_COLUMNS)):
            feature_a = MFCC_COLUMNS[i]
            feature_b = MFCC_COLUMNS[j]

            correlation = correlation_matrix.loc[
                feature_a,
                feature_b,
            ]

            if abs(correlation) >= CORRELATION_THRESHOLD:
                highly_correlated_pairs.append(
                    (
                        feature_a,
                        feature_b,
                        correlation,
                    )
                )

    if highly_correlated_pairs:
        highly_correlated_pairs.sort(
            key=lambda item: abs(item[2]),
            reverse=True,
        )

        for feature_a, feature_b, correlation in highly_correlated_pairs:
            print(
                f"{feature_a} <-> {feature_b}: "
                f"{correlation:.4f}"
            )
    else:
        print(
            f"No feature pairs with "
            f"|correlation| >= {CORRELATION_THRESHOLD}"
        )


if __name__ == "__main__":
    main()