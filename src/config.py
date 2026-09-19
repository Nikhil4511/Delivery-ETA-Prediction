"""Central project configuration and input validation constants."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "delivery_eta.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "eta_model.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

RAW_NUMERICAL_FEATURES = [
    "distance_km",
    "driver_experience_years",
    "restaurant_rating",
    "order_items",
    "preparation_time_min",
]
RAW_CATEGORICAL_FEATURES = [
    "weather",
    "traffic_level",
    "time_of_day",
    "vehicle_type",
    "delivery_area",
    "day_of_week",
]
RAW_FEATURES = RAW_NUMERICAL_FEATURES + RAW_CATEGORICAL_FEATURES
TARGET = "delivery_time_min"

# These broad operational limits reject impossible API/UI input without
# treating ordinary, rare observations as bad data.
VALID_RANGES = {
    "distance_km": (0.01, 100.0),
    "driver_experience_years": (0.0, 60.0),
    "restaurant_rating": (1.0, 5.0),
    "order_items": (1, 100),
    "preparation_time_min": (1.0, 240.0),
}

WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Fog", "Storm"]
TRAFFIC_OPTIONS = ["Low", "Medium", "High"]
TIME_OF_DAY_OPTIONS = ["Morning", "Afternoon", "Evening", "Night"]
VEHICLE_OPTIONS = ["Bike", "Scooter", "Car", "Bicycle"]
AREA_OPTIONS = ["Urban", "Suburban", "Rural"]
DAY_OPTIONS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
