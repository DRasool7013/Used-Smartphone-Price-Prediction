import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "used_smartphone_price_model.joblib"
META_FILE = ROOT / "models" / "metadata.json"

st.set_page_config(page_title="Used Smartphone Price Prediction", page_icon="📱", layout="centered")
st.title("📱 Used Smartphone Price Prediction")
st.write("Enter the smartphone details to estimate its used price.")

if not MODEL_FILE.exists() or not META_FILE.exists():
    st.warning("Trained model files are missing. Add the dataset and run `python -m src.train` first.")
    st.stop()

model = joblib.load(MODEL_FILE)
meta = json.loads(META_FILE.read_text(encoding="utf-8"))
feature_cols = meta["feature_columns"]
numeric_cols = set(meta["numeric_features"])
cat_cols = set(meta["categorical_features"])

st.caption(f"Selected model: {meta['model_name']} | Target: {meta['target_column']}")
st.info("The prediction uses the target's original unit. It is not necessarily INR.")

# Read raw data to create sensible categorical choices and numeric defaults.
data_file = ROOT / "data" / "used_phones_features.csv"
if not data_file.exists():
    data_file = ROOT / "data" / "used_phones_cleaned.csv"
if not data_file.exists():
    data_file = ROOT / "data" / "used_phones.csv"
training_df = pd.read_csv(data_file) if data_file.exists() else pd.DataFrame()

inputs = {}
with st.form("phone_form"):
    st.subheader("Smartphone specifications")
    for col in feature_cols:
        label = col.replace("_", " ").replace("-", " ").title()
        if col in numeric_cols:
            if col in training_df.columns:
                vals = pd.to_numeric(training_df[col], errors="coerce").dropna()
                default = float(vals.median()) if not vals.empty else 0.0
                min_value = float(vals.min()) if not vals.empty else 0.0
                max_value = float(vals.max()) if not vals.empty else max(default * 2, 1.0)
                if min_value == max_value:
                    max_value = min_value + 1.0
                inputs[col] = st.number_input(
                    label, min_value=min_value, max_value=max_value,
                    value=min(max(default, min_value), max_value), key=col
                )
            else:
                inputs[col] = st.number_input(label, value=0.0, key=col)
        else:
            options = []
            if col in training_df.columns:
                options = sorted(training_df[col].dropna().astype(str).unique().tolist())
            if options and len(options) <= 100:
                inputs[col] = st.selectbox(label, options, key=col)
            else:
                inputs[col] = st.text_input(label, key=col)

    submitted = st.form_submit_button("Predict used price", type="primary")

if submitted:
    row = pd.DataFrame([{c: inputs[c] for c in feature_cols}], columns=feature_cols)
    try:
        estimate = float(model.predict(row)[0])
        st.success(f"Estimated used price: {estimate:,.3f}")
        st.caption("This is a statistical estimate, not a guaranteed market price.")
    except Exception as exc:
        st.error(f"Prediction error: {exc}")

with st.expander("View final test metrics"):
    st.json(meta.get("test_metrics", {}))
    st.caption("Metrics are from the held-out test split and are in the target's unit.")
