"""Clean data, train candidate ETA models, compare them, and persist the best."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import DATA_PATH, MODEL_METADATA_PATH, MODEL_PATH, RAW_NUMERICAL_FEATURES, TARGET
from src.features import add_engineered_features
from src.preprocessing import build_pipeline


def inspect_data(data: pd.DataFrame) -> dict[str, Any]:
    """Return transparent cleaning diagnostics without changing any records."""
    numerical = data[RAW_NUMERICAL_FEATURES + [TARGET]].apply(pd.to_numeric, errors="coerce")
    invalid = {
        "non_positive_distance": int((numerical["distance_km"] <= 0).sum()),
        "negative_experience": int((numerical["driver_experience_years"] < 0).sum()),
        "rating_outside_1_to_5": int(((numerical["restaurant_rating"] < 1) | (numerical["restaurant_rating"] > 5)).sum()),
        "non_positive_items": int((numerical["order_items"] < 1).sum()),
        "non_positive_preparation_time": int((numerical["preparation_time_min"] <= 0).sum()),
        "non_positive_target": int((numerical[TARGET] <= 0).sum()),
    }
    outlier_counts: dict[str, int] = {}
    for column in RAW_NUMERICAL_FEATURES + [TARGET]:
        values = numerical[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_counts[column] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
    return {
        "rows_before": len(data),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_values": data.isna().sum().to_dict(),
        "invalid_value_counts": invalid,
        "iqr_outlier_flags_preserved": outlier_counts,
    }


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply only defensible cleaning rules and return a report.

    Duplicate rows are removed.  Rows missing the target or containing
    physically impossible numeric values are removed; feature gaps remain for
    the pipeline's imputers.  IQR outliers are *reported but preserved* because
    long trips and long preparation times can be genuine operations.
    """
    report = inspect_data(data)
    clean = data.drop_duplicates().copy()
    duplicate_rows_removed = len(data) - len(clean)

    for column in RAW_NUMERICAL_FEATURES + [TARGET]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    valid = (
        clean[TARGET].notna()
        & (clean[TARGET] > 0)
        & (clean["distance_km"].isna() | (clean["distance_km"] > 0))
        & (clean["driver_experience_years"].isna() | (clean["driver_experience_years"] >= 0))
        & (clean["restaurant_rating"].isna() | clean["restaurant_rating"].between(1, 5))
        & (clean["order_items"].isna() | (clean["order_items"] >= 1))
        & (clean["preparation_time_min"].isna() | (clean["preparation_time_min"] > 0))
    )
    cleaned = clean.loc[valid].reset_index(drop=True)
    report.update(
        {
            "duplicate_rows_removed": duplicate_rows_removed,
            "rows_removed_for_missing_or_invalid_target_or_values": int(len(clean) - len(cleaned)),
            "rows_after_cleaning": len(cleaned),
        }
    )
    return cleaned, report


def candidate_models() -> dict[str, Any]:
    """Return the requested regressors with reproducible settings."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def train_and_save(data_path=DATA_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train the four candidate models and save the model with lowest MAE."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run data/generate_dataset.py first.")

    raw_data = pd.read_csv(data_path)
    cleaned_data, cleaning_report = clean_dataset(raw_data)
    X = cleaned_data.drop(columns=[TARGET])
    y = cleaned_data[TARGET]

    # Feature engineering happens before each pipeline fit, but preprocessing
    # (including imputing and encoding) occurs after this split to avoid leakage.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train = add_engineered_features(X_train_raw)
    X_test = add_engineered_features(X_test_raw)

    results: list[dict[str, float | str]] = []
    trained_pipelines: dict[str, Any] = {}
    for name, estimator in candidate_models().items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results.append(
            {
                "Model": name,
                "MAE": float(mean_absolute_error(y_test, predictions)),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "R2": float(r2_score(y_test, predictions)),
            }
        )
        trained_pipelines[name] = pipeline

    comparison = pd.DataFrame(results).sort_values(["MAE", "RMSE"], ascending=True).reset_index(drop=True)
    best_name = str(comparison.iloc[0]["Model"])
    best_pipeline = trained_pipelines[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    metadata = {
        "best_model": best_name,
        "selection_rule": "Lowest validation MAE; RMSE used as tie-breaker.",
        "metrics": comparison.round(4).to_dict(orient="records"),
        "cleaning_report": cleaning_report,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return comparison, metadata


def main() -> None:
    comparison, metadata = train_and_save()
    print("\nValidation comparison (lower MAE is better):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved {metadata['best_model']} pipeline to {MODEL_PATH}")
    print(f"Saved diagnostics and metrics to {MODEL_METADATA_PATH}")


if __name__ == "__main__":
    main()
