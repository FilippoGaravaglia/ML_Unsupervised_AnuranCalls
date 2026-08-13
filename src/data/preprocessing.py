import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.schema import (
    ID_COLUMN,
    LABEL_COLUMNS,
    MFCC_COLUMNS,
)


def prepare_clustering_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:

    required_columns = (
        MFCC_COLUMNS
        + LABEL_COLUMNS
        + [ID_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    features = df[MFCC_COLUMNS].copy()

    metadata_columns = (
        LABEL_COLUMNS
        + [ID_COLUMN]
    )

    labels = df[metadata_columns].copy()

    scaler = StandardScaler()

    standardized_values = scaler.fit_transform(
        features
    )

    standardized_features = pd.DataFrame(
        standardized_values,
        columns=MFCC_COLUMNS,
        index=features.index,
    )

    return (
        standardized_features,
        labels,
        scaler,
    )