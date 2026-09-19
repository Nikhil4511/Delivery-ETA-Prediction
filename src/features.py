"""Leakage-safe feature engineering shared by training and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRAFFIC_SCORES = {"Low": 1, "Medium": 2, "High": 3}


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create model features using only information known before delivery.

    The function returns a copy and deliberately never reads the target.  It
    can therefore be safely called before inference as well as during fitting.
    Missing values are left for the sklearn pipeline's imputers to handle.
    """
    frame = data.copy()
    frame["rush_hour"] = frame["time_of_day"].isin(["Morning", "Evening"]).astype(int)
    frame["weekend"] = frame["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    frame["distance_category"] = pd.cut(
        frame["distance_km"],
        bins=[-np.inf, 3, 8, np.inf],
        labels=["Short", "Medium", "Long"],
    ).astype("object")
    safe_distance = frame["distance_km"].clip(lower=0.1)
    frame["prep_distance_ratio"] = frame["preparation_time_min"] / safe_distance
    frame["traffic_score"] = frame["traffic_level"].map(TRAFFIC_SCORES)
    frame["traffic_distance"] = frame["traffic_score"] * frame["distance_km"]
    return frame
