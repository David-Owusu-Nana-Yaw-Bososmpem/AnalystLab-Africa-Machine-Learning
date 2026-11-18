"""FastAPI service for next-hour electricity consumption forecasting (Zone 1)."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from xgboost import XGBRegressor

from app.schemas import ForecastRequest, ForecastResponse, HealthResponse

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "xgb_model.json"
META_PATH = ROOT / "model" / "model_meta.json"

model = None
meta = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the portable XGBoost model and its metadata when the API starts."""
    global model, meta

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    if not META_PATH.exists():
        raise RuntimeError(
            f"Model metadata not found at {META_PATH}. Run `python src/train.py` first."
        )

    with META_PATH.open(encoding="utf-8") as file:
        meta = json.load(file)

    feature_cols = meta.get("feature_cols")
    if not isinstance(feature_cols, list) or not feature_cols:
        raise RuntimeError("model_meta.json must contain a non-empty feature_cols list.")

    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    if model.n_features_in_ != len(feature_cols):
        raise RuntimeError(
            "Model feature count does not match model_meta.json: "
            f"{model.n_features_in_} != {len(feature_cols)}."
        )

    print(f"Model loaded from {MODEL_PATH}")
    yield


app = FastAPI(
    title="Electricity Consumption Forecast API",
    description=(
        "Predict next-hour mean electricity consumption (kW) for Zone 1 "
        "using historical patterns and environmental variables. "
        "Model: XGBoost trained on 2017 Tetouan (Morocco) data."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/", tags=["info"])
def root():
    return {
        "message": "Electricity Consumption Forecast API (Zone 1)",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.get("/model-info", tags=["info"])
def model_info():
    if not meta or model is None:
        raise HTTPException(status_code=503, detail="Model or metadata not loaded")
    return {
        "target_zone": meta.get("target_zone"),
        "best_run": meta.get("best_run"),
        "metrics": meta.get("metrics"),
        "feature_cols": meta.get("feature_cols"),
        "aggregation": meta.get("aggregation"),
        "date_range": meta.get("date_range"),
    }


@app.post("/predict", response_model=ForecastResponse, tags=["prediction"])
def predict(request: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build the row in the exact order used during model training.
    feature_values = request.model_dump()
    feature_cols = meta["feature_cols"]
    try:
        features = np.asarray(
            [[feature_values[name] for name in feature_cols]], dtype=np.float32
        )
        prediction = float(model.predict(features)[0])
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The model could not generate a prediction.") from exc

    return ForecastResponse(
        predicted_consumption=round(prediction, 2),
        zone="Zone1",
        unit="kW (hourly mean power)",
        model=meta.get("best_run", "XGBoost"),
    )
