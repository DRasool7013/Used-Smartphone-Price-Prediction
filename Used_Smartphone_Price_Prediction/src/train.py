import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.config import (
    MODEL_DIR, METRIC_DIR, RANDOM_STATE, TEST_SIZE, FEATURE_DATA_PATH
)
from src.data_utils import load_project_data, find_target

warnings.filterwarnings("ignore", category=FutureWarning)

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    # Dense output keeps all candidate estimators, including GradientBoostingRegressor, compatible.
    transformer = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ], sparse_threshold=0)
    return transformer, numeric_cols, categorical_cols

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(actual, predicted))
    }

def main():
    # Use engineered features when the feature-engineering stage has been run.
    df = pd.read_csv(FEATURE_DATA_PATH) if FEATURE_DATA_PATH.exists() else load_project_data()
    target = find_target(df.columns)
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    df = df[df[target] > 0].copy()

    # Exclude IDs and the target from input features.
    id_cols = [c for c in df.columns if c.strip().lower() in
               {"id", "phone_id", "listing_id", "record_id", "url"}]
    X = df.drop(columns=[target] + id_cols)
    y = df[target]

    # Remove constant columns; they contain no learning signal.
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    X = X.drop(columns=constant_cols)
    if X.empty:
        raise ValueError("No usable feature columns remain after removing target/IDs/constants.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE)
    }

    # Compare candidates using CV on training data only. Test data remains untouched.
    cv_rows = []
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        scores = cross_val_score(
            pipe, X_train, y_train, cv=5,
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        cv_rows.append({
            "Model": name,
            "CV_RMSE_Mean": float(-scores.mean()),
            "CV_RMSE_Std": float(scores.std())
        })
        print(f"{name}: CV RMSE = {-scores.mean():.4f} (+/- {scores.std():.4f})")

    cv_df = pd.DataFrame(cv_rows).sort_values("CV_RMSE_Mean")
    best_cv_name = cv_df.iloc[0]["Model"]
    print("\\nCV model comparison:\\n", cv_df.to_string(index=False))

    # Tune Random Forest using training data only. It is a robust tabular-data candidate.
    rf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    params = {
        "model__n_estimators": [150, 250, 350],
        "model__max_depth": [None, 10, 18, 25],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": [1.0, "sqrt", 0.8]
    }
    search = RandomizedSearchCV(
        rf_pipe, params, n_iter=10, cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    # Select the lower-CV-RMSE option between the best baseline and tuned RF.
    if best_cv_name == "Random Forest":
        baseline_estimator = candidates["Random Forest"]
    else:
        baseline_estimator = candidates[best_cv_name]
    baseline_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", baseline_estimator)
    ])
    baseline_pipe.fit(X_train, y_train)

    tuned_cv_rmse = float(-search.best_score_)
    baseline_cv_rmse = float(cv_df.iloc[0]["CV_RMSE_Mean"])
    # The tuned RF is selected only if its CV score improves over the best baseline CV score.
    if tuned_cv_rmse <= baseline_cv_rmse:
        final_model = search.best_estimator_
        final_name = "Tuned Random Forest"
    else:
        final_model = baseline_pipe
        final_name = best_cv_name

    # Final test evaluation is performed after model selection.
    test_predictions = final_model.predict(X_test)
    final_metrics = calculate_metrics(y_test, test_predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "used_smartphone_price_model.joblib")

    cv_df.to_csv(METRIC_DIR / "cross_validation_comparison.csv", index=False)
    pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": pd.Series(test_predictions)
    }).to_csv(METRIC_DIR / "test_actual_vs_predicted.csv", index=False)

    metadata = {
        "target_column": target,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "model_name": final_name,
        "test_metrics": final_metrics,
        "target_note": "Prediction is in the original target column's unit. Verify dataset documentation.",
        "best_random_forest_parameters": search.best_params_,
        "excluded_id_columns": id_cols,
        "excluded_constant_columns": constant_cols
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    print("\\nSelected model:", final_name)
    print("Held-out test metrics:", final_metrics)
    print("Saved model to:", MODEL_DIR / "used_smartphone_price_model.joblib")
    print("Saved reports to:", METRIC_DIR)

if __name__ == "__main__":
    main()
