import json
import joblib
import pandas as pd
from src.config import MODEL_DIR

def predict_from_dict(phone_details):
    model_path = MODEL_DIR / "used_smartphone_price_model.joblib"
    meta_path = MODEL_DIR / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Train the model first by running: python -m src.train")

    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [c for c in metadata["feature_columns"] if c not in phone_details]
    if missing:
        raise ValueError(f"Missing required feature values: {missing}")
    row = pd.DataFrame(
        [{c: phone_details[c] for c in metadata["feature_columns"]}],
        columns=metadata["feature_columns"]
    )
    return float(model.predict(row)[0]), metadata

if __name__ == "__main__":
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Train the model first by running: python -m src.train")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print("Enter smartphone details. Press Enter to use a blank value where allowed.")
    details = {}
    numeric_features = set(metadata.get("numeric_features", []))

    for feature in metadata["feature_columns"]:
        label = feature.replace("_", " ")
        raw = input(f"{label} ({'number' if feature in numeric_features else 'text'}): ").strip()
        if feature in numeric_features:
            if raw == "":
                raise ValueError(f"A numeric value is required for {feature}.")
            details[feature] = float(raw)
        else:
            details[feature] = raw if raw else None

    estimate, saved_metadata = predict_from_dict(details)
    print(f"\nEstimated {saved_metadata['target_column']}: {estimate:,.4f}")
    print("The estimate is in the target column's original unit; it is not necessarily INR.")
