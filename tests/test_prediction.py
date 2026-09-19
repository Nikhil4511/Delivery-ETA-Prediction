"""Basic validation and inference tests for the ETA project."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api import PredictionRequest
from src.predict import load_model, predict_eta


VALID_PAYLOAD = {
    "distance_km": 5.5,
    "weather": "Rain",
    "traffic_level": "High",
    "time_of_day": "Evening",
    "vehicle_type": "Bike",
    "driver_experience_years": 2.5,
    "restaurant_rating": 4.3,
    "order_items": 4,
    "preparation_time_min": 18,
    "delivery_area": "Urban",
    "day_of_week": "Friday",
}


def test_model_loading() -> None:
    """The saved artifact must be a complete sklearn prediction pipeline."""
    model = load_model()
    assert hasattr(model, "predict")
    assert "preprocessor" in model.named_steps


def test_valid_prediction_returns_float() -> None:
    prediction = predict_eta(VALID_PAYLOAD)
    assert isinstance(prediction, float)
    assert prediction > 0


def test_invalid_distance_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**VALID_PAYLOAD, "distance_km": 0})


def test_invalid_rating_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**VALID_PAYLOAD, "restaurant_rating": 5.5})


def test_invalid_order_count_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(**{**VALID_PAYLOAD, "order_items": 0})
