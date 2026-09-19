"""Load the saved pipeline and make validated ETA predictions."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, RAW_FEATURES, RAW_NUMERICAL_FEATURES, VALID_RANGES
from src.features import add_engineered_features


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load and cache the complete preprocessing-plus-estimator pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_PATH}. Generate data and run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


def validate_delivery_info(delivery_info: Mapping[str, Any]) -> dict[str, Any]:
    """Check required fields and broad numeric bounds for direct Python use."""
    missing = [feature for feature in RAW_FEATURES if feature not in delivery_info]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    cleaned = {feature: delivery_info[feature] for feature in RAW_FEATURES}
    for feature in RAW_NUMERICAL_FEATURES:
        try:
            value = float(cleaned[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{feature} must be numeric.") from error
        minimum, maximum = VALID_RANGES[feature]
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{feature} must be between {minimum} and {maximum}.")
        cleaned[feature] = value

    for feature in set(RAW_FEATURES) - set(RAW_NUMERICAL_FEATURES):
        if not isinstance(cleaned[feature], str) or not cleaned[feature].strip():
            raise ValueError(f"{feature} must be a non-empty string.")
        cleaned[feature] = cleaned[feature].strip()
    return cleaned


def prepared_features(delivery_info: Mapping[str, Any]) -> pd.DataFrame:
    """Validate a single request and apply the shared feature engineering."""
    valid_info = validate_delivery_info(delivery_info)
    return add_engineered_features(pd.DataFrame([valid_info]))


def predict_eta(delivery_info: Mapping[str, Any]) -> float:
    """Predict one delivery ETA in minutes as a native Python float."""
    features = prepared_features(delivery_info)
    prediction = load_model().predict(features)[0]
    return float(prediction)


def _humanize_feature(feature_name: str) -> str:
    """Turn encoded sklearn feature names into labels suitable for the UI."""
    feature_name = feature_name.replace("numeric__", "").replace("categorical__", "")
    return feature_name.replace("_", " ").replace("ratio", "ratio").title()


def explain_prediction(delivery_info: Mapping[str, Any], top_n: int = 4) -> dict[str, Any]:
    """Return genuine SHAP contributions for a supported tree-based pipeline.

    The function never guesses a narrative.  If SHAP or the selected estimator
    cannot provide tree explanations, it returns an explanatory availability
    message instead.
    """
    pipeline = load_model()
    estimator = pipeline.named_steps["model"]
    model_name = estimator.__class__.__name__
    if not any(token in model_name for token in ("Forest", "GradientBoosting", "XGB")):
        return {"available": False, "message": f"SHAP is not configured for {model_name}."}

    try:
        import shap

        features = prepared_features(delivery_info)
        transformer = pipeline.named_steps["preprocessor"]
        transformed = transformer.transform(features)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        values = np.asarray(shap_values).reshape(-1)
        names = transformer.get_feature_names_out()
        transformed_values = (
            transformed.toarray().reshape(-1)
            if hasattr(transformed, "toarray")
            else np.asarray(transformed).reshape(-1)
        )
        contributions = [
            {"feature": _humanize_feature(str(name)), "shap_value": float(value)}
            for name, value, encoded_value in zip(names, values, transformed_values, strict=True)
            # Explain the selected one-hot category, not an absent category's
            # reference-level contribution. Numeric fields are always shown.
            if abs(value) > 1e-8 and (not str(name).startswith("categorical__") or encoded_value > 0.5)
        ]
        positives = sorted((item for item in contributions if item["shap_value"] > 0), key=lambda item: item["shap_value"], reverse=True)
        negatives = sorted((item for item in contributions if item["shap_value"] < 0), key=lambda item: item["shap_value"])
        return {
            "available": True,
            "increasing": positives[:top_n],
            "decreasing": negatives[:top_n],
            "message": "Positive values pushed this estimate upward; negative values pushed it downward.",
        }
    except ImportError:
        return {"available": False, "message": "Install the optional SHAP dependency to view explanations."}
    except Exception as error:  # UI should still provide the ETA if explanation fails.
        return {"available": False, "message": f"SHAP explanation is unavailable: {error}"}
