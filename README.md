# Delivery ETA Prediction

A production-style, beginner-friendly machine-learning project that estimates delivery time in minutes for food or e-commerce orders. It is designed as a B.Tech AI/ML portfolio project: the code is modular, the pipeline is reusable, and every modelling choice can be explained in an interview.

> **Dataset notice:** `data/delivery_eta.csv` is completely **synthetic**. It is generated for educational and demonstration purposes; it does not represent data from any real delivery company.

## Problem Statement

Customers and operations teams need a useful estimate of how long an order may take to reach its destination. ETA depends on distance, preparation time, traffic, weather, delivery context, and driver experience. This project predicts `delivery_time_min` from information available before dispatch.

## Business Objective

Minimize the difference between predicted and actual delivery time. The primary metric is Mean Absolute Error (MAE), because an MAE of, for example, 4 minutes is easy for non-technical stakeholders to understand: predictions are off by roughly four minutes on average.

## Features

| Type | Columns |
|---|---|
| Numerical | `distance_km`, `driver_experience_years`, `restaurant_rating`, `order_items`, `preparation_time_min` |
| Categorical | `weather`, `traffic_level`, `time_of_day`, `vehicle_type`, `delivery_area`, `day_of_week` |
| Target | `delivery_time_min` |

## Dataset

Run `python data/generate_dataset.py` to create more than 6,000 rows (including a small number of deliberate duplicate records). The generator uses plausible relationships: greater distance, high traffic, poor weather, long preparation time, and rural delivery areas tend to increase ETA; driver experience tends to reduce it. It also injects small amounts of missing feature data and a few missing targets so the cleaning code has realistic examples.

## Project Architecture

```text
delivery_eta_prediction/
├── data/
│   ├── delivery_eta.csv                 # generated synthetic data
│   └── generate_dataset.py              # reproducible data generator
├── models/
│   ├── eta_model.joblib                 # generated complete sklearn pipeline
│   └── model_metadata.json              # generated metrics and cleaning report
├── notebooks/
│   └── 01_eda_and_model_training.ipynb  # visual exploration and training walkthrough
├── src/
│   ├── config.py                        # paths, feature lists, input limits
│   ├── features.py                      # shared, leakage-safe feature engineering
│   ├── preprocessing.py                 # imputing, encoding, ColumnTransformer
│   ├── train.py                         # cleaning, comparison, persistence
│   └── predict.py                       # loading, validation, prediction, SHAP
├── tests/test_prediction.py             # inference and request-validation tests
├── app.py                               # Streamlit UI
├── api.py                               # FastAPI service
├── requirements.txt
└── Dockerfile
```

`src/__init__.py` marks the source directory as an importable package. The extra generated `model_metadata.json` records the true validation results and cleaning report, so the README never has to claim made-up metrics.

## Machine Learning Pipeline

1. The script inspects nulls, duplicates, invalid values, and IQR outlier flags.
2. Exact duplicate records are removed. Rows with an absent target or physically impossible values (such as negative distance) are excluded. Missing **features** are retained for imputation.
3. IQR outliers are not automatically deleted. A long-distance rural delivery may be rare but valid; the IQR results are a review signal, not a deletion rule.
4. `train_test_split(test_size=0.20, random_state=42)` splits the raw data. This happens before imputation or one-hot encoding.
5. `SimpleImputer`, `OneHotEncoder(handle_unknown="ignore")`, `ColumnTransformer`, and the estimator are stored together in one sklearn `Pipeline`.
6. The lowest validation MAE is selected, with RMSE as a tie-breaker, and the full pipeline is saved as `models/eta_model.joblib`.

Keeping transformations inside the pipeline prevents train/inference mismatch and lets the application safely handle unseen categories.

## Exploratory Data Analysis

Open `notebooks/01_eda_and_model_training.ipynb` after generating data. It includes:

- dataset overview, missing-value and duplicate analysis;
- target distribution;
- delivery time versus distance and preparation time;
- delivery time versus traffic, weather, vehicle, time of day, area, and day;
- a numerical correlation heatmap; and
- the real model-comparison table produced by the training module.

## Feature Engineering

All engineered fields are calculated only from delivery-time inputs, never from the target.

| Feature | Purpose |
|---|---|
| `rush_hour` | Marks morning/evening periods, which often have more congestion. |
| `weekend` | Distinguishes Saturday/Sunday demand patterns. |
| `distance_category` | Groups trips into Short (≤3km), Medium (3–8km), and Long (>8km). |
| `prep_distance_ratio` | Captures whether kitchen preparation dominates travel distance. |
| `traffic_score` | Converts Low/Medium/High traffic into an ordered 1/2/3 signal. |
| `traffic_distance` | Captures the interaction: high traffic matters more on longer trips. |

## Models

The training script compares:

1. Linear Regression — a transparent baseline.
2. Random Forest Regressor — a bagged tree ensemble for nonlinear relationships.
3. Gradient Boosting Regressor — sequential trees that reduce residual error.
4. XGBRegressor — optimized gradient-boosted trees.

## Evaluation Metrics

- **MAE:** average absolute ETA error in minutes; primary business metric.
- **RMSE:** penalizes larger misses more strongly.
- **R²:** proportion of target variation explained on the validation split.

## Model Comparison

The table below is the actual validation output from the checked-in synthetic dataset, generated with seed 42 and the prescribed split. Re-running `python -m src.train` prints the values and writes their full precision to `models/model_metadata.json`; metrics must be recomputed if you change the generator, features, or model settings.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 3.174 | 4.223 | 0.931 |
| Random Forest | 3.655 | 5.018 | 0.902 |
| Gradient Boosting | 3.020 | 4.105 | 0.934 |
| XGBoost | **2.913** | **3.968** | **0.939** |

## Explainable AI

For a selected tree-based model, the Streamlit app calls SHAP's `TreeExplainer` on the actual transformed feature vector. It displays only real positive and negative SHAP contributions—for example, a high-traffic feature that pushed *that prediction* upward. If SHAP is unavailable or the selected model is not supported, the UI says so rather than fabricating an explanation.

The app does **not** show a made-up confidence range. A statistically validated uncertainty method is intentionally left as a future enhancement.

## Streamlit Application

The UI collects delivery information in two columns, validates numerical entries, then uses `st.metric` and a clear success message to display the ETA. It reminds users that actual delivery times can vary.

```bash
streamlit run app.py
```

Open the local address printed by Streamlit (usually `http://localhost:8501`).

## FastAPI

Start the API from the project root:

```bash
uvicorn api:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger documentation.

`GET /` is a health/documentation pointer. `POST /predict` accepts:

```json
{
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
  "day_of_week": "Friday"
}
```

Example response:

```json
{
  "predicted_eta_minutes": 38.5
}
```

The numeric value is an example response shape, not a promised result. Pydantic returns a 422 validation error for invalid values, and the service returns 503 if no trained model exists.

## Installation

Python 3.11+ is recommended.

```bash
git clone <your-repository-url>
cd delivery_eta_prediction
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

```bash
# 1. Create reproducible synthetic data
python data/generate_dataset.py

# 2. Train all models, print real validation metrics, and save the best full pipeline
python -m src.train

# 3. Check core inference and validation behavior
pytest -q

# 4. Start the Streamlit interface
streamlit run app.py

# 5. In another terminal, optionally start the API
uvicorn api:app --reload
```

## Development Walkthrough

| Stage | What and why | Expected output | Common problem and fix |
|---|---|---|---|
| 1. Generate dataset | Makes reproducible synthetic records with known relationships. | `data/delivery_eta.csv` and missing/duplicate counts. | `ModuleNotFoundError`: activate `.venv` then `pip install -r requirements.txt`. |
| 2. Inspect/clean | Measures data quality; removes duplicates/impossible rows but preserves plausible IQR extremes. | Cleaning diagnostics in metadata/notebook. | `FileNotFoundError`: run the generator from the project root. |
| 3. EDA | Visualizes distributions and relationships before modelling. | Notebook charts and correlation heatmap. | Blank plots: run notebook cells in order after selecting the virtual-environment kernel. |
| 4. Engineer features | Adds operational context without reading the target. | Six new feature columns before preprocessing. | `KeyError`: check the input uses all required raw feature names. |
| 5. Preprocess | Learns imputing/encoding from the training fold only. | A fitted `ColumnTransformer` inside the saved pipeline. | Unseen category: `handle_unknown="ignore"` deliberately avoids a crash. |
| 6. Train/compare | Compares four regressors using MAE, RMSE, and R². | Printed comparison and `model_metadata.json`. | Slow first run: XGBoost and tree ensembles take longer; reduce estimators only for experimentation. |
| 7. Persist/predict | Saves preprocessing plus estimator so inference stays consistent. | `models/eta_model.joblib` and a float ETA. | Missing model: run `python -m src.train`. |
| 8. UI/API/tests | Makes validated predictions available to users and services. | Streamlit page, `/docs`, and passing pytest tests. | Port in use: choose another `--server.port` / uvicorn port. |

## Docker

Build after generating data and training the model (the Docker image includes both files):

```bash
docker build -t delivery-eta-predictor .
docker run --rm -p 8501:8501 delivery-eta-predictor
```

Open `http://localhost:8501`. The Dockerfile runs the Streamlit application.

## Project Screenshots Placeholder

Add screenshots of the EDA notebook, Streamlit input form, ETA metric, SHAP explanation, and FastAPI `/docs` page here after running the project locally.

## Limitations

- Synthetic patterns cannot capture real-world dispatch, restaurant, map, or driver behavior.
- The random split does not test time-based drift.
- ETA is a point estimate; no calibrated prediction interval is implemented.
- Category options and numeric limits are simplified for a learning project.

## Future Improvements

- Train on consented, anonymized real operational data with data-governance review.
- Add geospatial route, merchant load, and live traffic features.
- Evaluate time-based and region-based validation splits.
- Monitor data drift and prediction error after deployment.
- Add conformal prediction or quantile models for a validated ETA interval.

