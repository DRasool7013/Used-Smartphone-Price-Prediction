from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "used_phones.csv"
CLEAN_DATA_PATH = ROOT / "data" / "used_phones_cleaned.csv"
FEATURE_DATA_PATH = ROOT / "data" / "used_phones_features.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
METRIC_DIR = REPORT_DIR / "metrics"

# Set to the exact target column if automatic detection does not find it.
TARGET_COLUMN = None
TARGET_CANDIDATES = [
    "normalized_used_price", "used_price", "selling_price",
    "resale_price", "price"
]
RANDOM_STATE = 42
TEST_SIZE = 0.20
