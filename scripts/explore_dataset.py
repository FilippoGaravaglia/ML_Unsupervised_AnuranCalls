from src.data.loader import load_dataset


def main() -> None:
    df = load_dataset()

    print("=== DATASET SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== DUPLICATED ROWS ===")
    print(df.duplicated().sum())

    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe())

    print("\n=== UNIQUE LABEL VALUES ===")

    for column in ["Family", "Genus", "Species"]:
        if column in df.columns:
            print(f"\n{column}:")
            print(df[column].value_counts())


if __name__ == "__main__":
    main()