import pandas as pd
from src.data_utils import load_raw_data, find_target
from src.config import CLEAN_DATA_PATH

def main():
    df = load_raw_data()
    target = find_target(df.columns)

    print("\\n========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)
    print("\\nFirst five rows:\\n", df.head())
    print("\\nColumn names:\\n", df.columns.tolist())
    print("\\nData types:\\n", df.dtypes)
    print("\\nMissing values:\\n", df.isna().sum().sort_values(ascending=False))
    print("\\nDuplicate rows:", df.duplicated().sum())
    print("\\nNumeric summary:\\n", df.describe(include="number").T)
    print(f"\\nTarget column: {target}")
    print("\\nTarget summary:\\n", pd.to_numeric(df[target], errors="coerce").describe())

    # Remove exact duplicates and records without a valid positive target.
    df = df.drop_duplicates().copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df = df[df[target] > 0].copy()

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print("\\nCleaned data saved to:", CLEAN_DATA_PATH)
    print("Cleaned shape:", df.shape)
    print("Note: feature missing values are imputed inside the training pipeline.")

if __name__ == "__main__":
    main()
