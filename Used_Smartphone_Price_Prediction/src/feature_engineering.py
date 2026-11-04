from datetime import date
import pandas as pd

from src.config import CLEAN_DATA_PATH, DATA_PATH, FEATURE_DATA_PATH
from src.data_utils import find_target


def main():
    source = CLEAN_DATA_PATH if CLEAN_DATA_PATH.exists() else DATA_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"Dataset not found. Add data/used_phones.csv and run python -m src.data_overview first."
        )

    df = pd.read_csv(source)
    target = find_target(df.columns)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df = df[df[target] > 0].copy()

    # Derive phone age from release year using the current calendar year.
    # Drop release_year afterward to avoid presenting two redundant inputs in the app.
    if "release_year" in df.columns:
        reference_year = date.today().year
        release_year = pd.to_numeric(df["release_year"], errors="coerce")
        df["phone_age_years"] = (reference_year - release_year).clip(lower=0)
        df = df.drop(columns=["release_year"])
        print(f"Created phone_age_years using reference year {reference_year}.")
    else:
        print("release_year not found; no phone-age feature was created.")

    FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURE_DATA_PATH, index=False)
    print("Feature dataset saved to:", FEATURE_DATA_PATH)
    print("Feature dataset shape:", df.shape)


if __name__ == "__main__":
    main()
