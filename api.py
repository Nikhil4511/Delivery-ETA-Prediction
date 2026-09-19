"""FastAPI interface for Delivery ETA Prediction."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_eta


app = FastAPI(title="Delivery ETA Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    """Validated delivery details accepted by the prediction endpoint."""

    distance_km: float = Field(gt=0, le=100, examples=[5.5])
    weather: str = Field(min_length=1, max_length=50, examples=["Rain"])
    traffic_level: str = Field(min_length=1, max_length=50, examples=["High"])
    time_of_day: str = Field(min_length=1, max_length=50, examples=["Evening"])
    vehicle_type: str = Field(min_length=1, max_length=50, examples=["Bike"])
    driver_experience_years: float = Field(ge=0, le=60, examples=[2.5])
    restaurant_rating: float = Field(ge=1, le=5, examples=[4.3])
    order_items: int = Field(ge=1, le=100, examples=[4])
    preparation_time_min: float = Field(gt=0, le=240, examples=[18])
    delivery_area: str = Field(min_length=1, max_length=50, examples=["Urban"])
    day_of_week: str = Field(min_length=1, max_length=50, examples=["Friday"])


class PredictionResponse(BaseModel):
    predicted_eta_minutes: float


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    """Provide a small health and documentation pointer."""
    return {"message": "Delivery ETA Prediction API is running. Visit /docs for interactive documentation."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"]) # endpoint
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return a model-based ETA prediction for one validated delivery."""
    try:
        eta = predict_eta(request.model_dump() if hasattr(request, "model_dump") else request.dict())
        return PredictionResponse(predicted_eta_minutes=round(eta, 2))
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
