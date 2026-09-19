"""Generate an educational synthetic dataset for the Delivery ETA project.

The generated records are deliberately fictional.  The relationships are
plausible enough to demonstrate an ML workflow, but they are not a proxy for
any delivery company's operational data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).with_name("delivery_eta.csv")


def generate_dataset(rows: int = 6_000, random_state: int = 42) -> pd.DataFrame:
    """Return a realistic-looking, intentionally synthetic ETA dataset.

    A small amount of missing data and a few duplicate rows are included so
    that the cleaning stage has meaningful examples to handle.
    """
    if rows < 5_000:
        raise ValueError("Use at least 5,000 rows for this project.")

    rng = np.random.default_rng(random_state)
    time_of_day = rng.choice(
        ["Morning", "Afternoon", "Evening", "Night"],
        size=rows,
        p=[0.24, 0.31, 0.34, 0.11],
    )
    delivery_area = rng.choice(["Urban", "Suburban", "Rural"], size=rows, p=[0.58, 0.29, 0.13])
    weather = rng.choice(["Clear", "Cloudy", "Rain", "Fog", "Storm"], size=rows, p=[0.44, 0.24, 0.20, 0.08, 0.04])
    vehicle_type = rng.choice(["Bike", "Scooter", "Car", "Bicycle"], size=rows, p=[0.42, 0.30, 0.20, 0.08])
    day_of_week = rng.choice(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        size=rows,
        p=[0.14, 0.14, 0.14, 0.14, 0.17, 0.14, 0.13],
    )

    # Longer trips are more common outside dense urban areas.
    distance_km = rng.gamma(shape=2.3, scale=2.0, size=rows)
    distance_km += np.where(delivery_area == "Rural", rng.uniform(2, 7, rows), 0)
    distance_km = np.clip(distance_km, 0.5, 28).round(2)

    rush = np.isin(time_of_day, ["Morning", "Evening"])
    traffic_probability = np.where(rush, 0.50, 0.24) + np.where(delivery_area == "Urban", 0.12, 0)
    traffic_draw = rng.random(rows)
    traffic_level = np.where(
        traffic_draw < traffic_probability,
        "High",
        np.where(traffic_draw < traffic_probability + 0.43, "Medium", "Low"),
    )

    driver_experience_years = np.clip(rng.gamma(shape=2.3, scale=1.7, size=rows), 0, 15).round(1)
    restaurant_rating = np.clip(rng.normal(loc=4.15, scale=0.38, size=rows), 2.5, 5.0).round(1)
    order_items = np.clip(rng.poisson(lam=3.2, size=rows) + 1, 1, 15)
    preparation_time_min = np.clip(
        rng.gamma(shape=4.5, scale=3.2, size=rows) + order_items * 0.8,
        5,
        65,
    ).round(1)

    traffic_effect = pd.Series(traffic_level).map({"Low": 0, "Medium": 5, "High": 12}).to_numpy()
    weather_effect = pd.Series(weather).map({"Clear": 0, "Cloudy": 1.5, "Rain": 5, "Fog": 7, "Storm": 12}).to_numpy()
    vehicle_effect = pd.Series(vehicle_type).map({"Bike": 0, "Scooter": -1, "Car": 2, "Bicycle": 5}).to_numpy()
    area_effect = pd.Series(delivery_area).map({"Urban": 2, "Suburban": 0, "Rural": 5}).to_numpy()
    peak_effect = np.where(rush, 3.5, 0)
    weekend_effect = np.isin(day_of_week, ["Saturday", "Sunday"]).astype(float) * 1.5
    # These operational effects are nonlinear: a kitchen bottleneck appears
    # after a threshold, and heavy congestion becomes disproportionately costly
    # on longer trips.  They make the modelling exercise more realistic than a
    # purely linear formula while keeping all drivers interpretable.
    long_trip_congestion = ((traffic_level == "High") & (distance_km > 7)) * (4.5 + (distance_km - 7) * 0.7)
    kitchen_bottleneck = np.maximum(preparation_time_min - 28, 0) * 0.7
    severe_weather_congestion = np.isin(weather, ["Fog", "Storm"]) * (traffic_level == "High") * 5.5
    long_bicycle_trip = (vehicle_type == "Bicycle") * np.maximum(distance_km - 5, 0) * 0.75
    noise = rng.normal(loc=0, scale=3.0, size=rows)

    delivery_time_min = (
        7
        + distance_km * (2.25 + (traffic_level == "High") * 0.65)
        + preparation_time_min * 0.78
        + traffic_effect
        + weather_effect
        + vehicle_effect
        + area_effect
        + peak_effect
        + weekend_effect
        + long_trip_congestion
        + kitchen_bottleneck
        + severe_weather_congestion
        + long_bicycle_trip
        - driver_experience_years * 0.55
        - (restaurant_rating - 4.0) * 1.4
        + noise
    )

    dataset = pd.DataFrame(
        {
            "distance_km": distance_km,
            "driver_experience_years": driver_experience_years,
            "restaurant_rating": restaurant_rating,
            "order_items": order_items,
            "preparation_time_min": preparation_time_min,
            "weather": weather,
            "traffic_level": traffic_level,
            "time_of_day": time_of_day,
            "vehicle_type": vehicle_type,
            "delivery_area": delivery_area,
            "day_of_week": day_of_week,
            "delivery_time_min": np.clip(delivery_time_min, 12, 160).round(1),
        }
    )

    # Missingness is small and random.  Target rows with no label are removed
    # during training rather than imputed, because labels must be observed.
    missing_columns = [
        "distance_km",
        "driver_experience_years",
        "restaurant_rating",
        "preparation_time_min",
        "weather",
        "traffic_level",
        "vehicle_type",
    ]
    for column in missing_columns:
        mask = rng.random(rows) < 0.018
        dataset.loc[mask, column] = np.nan
    dataset.loc[rng.random(rows) < 0.004, "delivery_time_min"] = np.nan

    # Add a few exact duplicate records on purpose for duplicate analysis.
    duplicates = dataset.sample(n=max(10, rows // 200), random_state=random_state)
    return pd.concat([dataset, duplicates], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic delivery ETA dataset.")
    parser.add_argument("--rows", type=int, default=6_000, help="Base rows before duplicate injection (minimum 5,000).")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="CSV output location.")
    args = parser.parse_args()

    dataset = generate_dataset(rows=args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset):,} synthetic rows to {args.output}")
    print(f"Duplicate rows included: {dataset.duplicated().sum()}")
    print("Missing values by column:")
    print(dataset.isna().sum()[dataset.isna().sum() > 0].to_string())


if __name__ == "__main__":
    main()
