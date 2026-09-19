"""Build the sklearn preprocessing and modelling pipeline."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import RAW_CATEGORICAL_FEATURES, RAW_NUMERICAL_FEATURES


ENGINEERED_NUMERICAL_FEATURES = ["rush_hour", "weekend", "prep_distance_ratio", "traffic_score", "traffic_distance"]
ENGINEERED_CATEGORICAL_FEATURES = ["distance_category"]
NUMERICAL_FEATURES = RAW_NUMERICAL_FEATURES + ENGINEERED_NUMERICAL_FEATURES
CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Return a trainable transformer that also handles unseen categories."""
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Keep this exact setting: an API caller may send a new category.
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(estimator: Any) -> Pipeline:
    """Pair preprocessing with an estimator so inference mirrors training."""
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])
