from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/raw/Frogs_MFCCs.csv")


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)