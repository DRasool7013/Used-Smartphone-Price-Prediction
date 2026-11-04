import pandas as pd
from src.config import DATA_PATH, CLEAN_DATA_PATH, TARGET_COLUMN, TARGET_CANDIDATES

def find_target(columns):
    if TARGET_COLUMN:
        if TARGET_COLUMN not in columns:
            raise ValueError(
                f"TARGET_COLUMN='{TARGET_COLUMN}' was not found. "
                f"Available columns: {list(columns)}"
            )
        return TARGET_COLUMN
    normalized = {str(col).strip().lower(): col for col in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        "Could not detect the price target. Set TARGET_COLUMN in src/config.py. "
        f"Available columns: {list(columns)}"
    )

def load_raw_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download the CSV and save it as "
            "data/used_phones.csv. See data/README.md."
        )
    return pd.read_csv(DATA_PATH)

def load_project_data(prefer_clean=True):
    path = CLEAN_DATA_PATH if prefer_clean and CLEAN_DATA_PATH.exists() else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Add data/used_phones.csv first."
        )
    return pd.read_csv(path)
