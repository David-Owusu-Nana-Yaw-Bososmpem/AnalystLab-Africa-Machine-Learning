#!/usr/bin/env python3
"""
Train electricity consumption forecasting models (XGBoost + baseline).
Chronological split, feature engineering with lags/rolling, experiment tracking.

Usage:
    python src/train.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "powerconsumption.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
MLRUNS_DIR = ROOT / "mlruns"

TARGET_ZONE = "PowerConsumption_Zone1"
TARGET_COL = "consumption"

FEATURE_COLS = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "temperature",
    "humidity",
    "windspeed",
    "general_diffuse_flows",
    "diffuse_flows",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_24",
    "rolling_std_24",
]


def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load raw 10-min data, clean, aggregate to hourly, engineer features."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Datetime"] = pd.to_datetime(df["Datetime"], dayfirst=False, format="mixed")
    df = df.sort_values("Datetime").drop_duplicates(subset=["Datetime"])
    df = df.set_index("Datetime")

    rename = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "WindSpeed": "windspeed",
        "GeneralDiffuseFlows": "general_diffuse_flows",
        "DiffuseFlows": "diffuse_flows",
        "PowerConsumption_Zone1": "zone1",
        "PowerConsumption_Zone2": "zone2",
        "PowerConsumption_Zone3": "zone3",
    }
    df = df.rename(columns=rename)

    # Forward-fill only: back-filling could use future observations to fill
    # earlier timestamps. The supplied dataset currently has no missing values.
    if df.isna().any().any():
        df = df.ffill()

    # Data dictionary labels values as kWh, but magnitudes (~20k–40k) match
    # average power (kW) over the 10-minute interval for a city-scale zone.
    # We aggregate to hourly *mean power* (kW). This matches published analyses
    # of the Tetouan dataset and avoids inflating totals by summing interval
    # averages that already represent power levels.
    hourly = df.resample("h").agg(
        {
            "temperature": "mean",
            "humidity": "mean",
            "windspeed": "mean",
            "general_diffuse_flows": "mean",
            "diffuse_flows": "mean",
            "zone1": "mean",
            "zone2": "mean",
            "zone3": "mean",
        }
    )
    hourly = hourly.dropna()

    # Zone 1 has the highest average consumption
    hourly[TARGET_COL] = hourly["zone1"]
    hourly = hourly.drop(columns=["zone1", "zone2", "zone3"])

    hourly["hour"] = hourly.index.hour
    hourly["dayofweek"] = hourly.index.dayofweek
    hourly["month"] = hourly.index.month
    hourly["is_weekend"] = (hourly["dayofweek"] >= 5).astype(int)

    for lag in [1, 2, 3, 24]:
        hourly[f"lag_{lag}"] = hourly[TARGET_COL].shift(lag)

    # Rolling stats use only past values (shift before rolling)
    hourly["rolling_mean_3"] = hourly[TARGET_COL].shift(1).rolling(3).mean()
    hourly["rolling_mean_6"] = hourly[TARGET_COL].shift(1).rolling(6).mean()
    hourly["rolling_mean_24"] = hourly[TARGET_COL].shift(1).rolling(24).mean()
    hourly["rolling_std_24"] = hourly[TARGET_COL].shift(1).rolling(24).std()

    hourly = hourly.dropna()
    return hourly


def chronological_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(y_true, y_pred, prefix: str = "") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {
        f"{prefix}mae": round(mae, 2),
        f"{prefix}rmse": round(rmse, 2),
        f"{prefix}r2": round(r2, 4),
    }


def main():
    print("Loading and preparing data...")
    df = load_and_prepare(DATA_PATH)
    print(f"Hourly rows after feature engineering: {len(df)}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    print(f"Target zone: {TARGET_ZONE} (mean ≈ {df[TARGET_COL].mean():.1f})")

    train_df, test_df = chronological_split(df, test_ratio=0.2)
    print(f"Train: {len(train_df)} hours | Test: {len(test_df)} hours")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # Baseline: lag-1 persistence
    baseline_pred = X_test["lag_1"].values
    baseline_metrics = evaluate(y_test, baseline_pred, prefix="baseline_")
    print("\nBaseline (lag-1 persistence):")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v}")

    experiments_log = []

    def run_experiment(run_name: str, params: dict):
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        metrics.update(baseline_metrics)
        experiments_log.append({"run_name": run_name, "params": params, "metrics": metrics})
        print(f"\nXGBoost ({run_name}):")
        for k, v in metrics.items():
            if not k.startswith("baseline"):
                print(f"  {k}: {v}")
        
        # MLflow experiment tracking using SQLite
        try:
            import mlflow
            import mlflow.xgboost

            # Connect MLflow to the SQLite database
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

            # Create or retrieve the experiment
            mlflow.set_experiment("power_consumption_forecast")

            # Track this model training run
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.xgboost.log_model(model, "model")

            print("  Successfully logged to MLflow")

        except Exception as e:
            print(f"  MLflow tracking failed: {type(e).__name__}: {e}")
            raise
        return model, metrics

    params_list = [
        (
            "xgb_default",
            {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        (
            "xgb_deeper",
            {
                "n_estimators": 400,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 1.5,
                "min_child_weight": 3,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        (
            "xgb_regularised",
            {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.06,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "min_child_weight": 5,
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
    ]

    best_model = None
    best_metrics = None
    best_run_name = None
    for name, params in params_list:
        model, metrics = run_experiment(name, params)
        if best_metrics is None or metrics["rmse"] < best_metrics["rmse"]:
            best_model = model
            best_metrics = metrics
            best_run_name = name

    print(f"\nBest experiment: {best_run_name}")
    print(f"  Test MAE  = {best_metrics['mae']}")
    print(f"  Test RMSE = {best_metrics['rmse']}")
    print(f"  Test R²   = {best_metrics['r2']}")
    print(f"  Baseline RMSE = {best_metrics['baseline_rmse']}")

    # Persist the model in XGBoost's portable JSON format. The API loads this
    # file directly instead of relying on a Python pickle tied to library versions.
    best_model.save_model(MODEL_DIR / "xgb_model.json")
    meta = {
        "target_zone": TARGET_ZONE,
        "target_col": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "best_run": best_run_name,
        "metrics": best_metrics,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "date_range": {"start": str(df.index.min()), "end": str(df.index.max())},
        "aggregation": "hourly mean power (kW) from 10-min readings",
        "notes": (
            "Zone 1 selected (highest average consumption). "
            "Chronological 80/20 split. Features use only past information."
        ),
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(MODEL_DIR / "experiments_log.json", "w") as f:
        json.dump(experiments_log, f, indent=2)

    importance = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values(
        ascending=False
    )
    print("\nTop feature importances:")
    print(importance.head(10).to_string())
    importance.to_csv(MODEL_DIR / "feature_importance.csv")

    print(f"\nModel saved to {MODEL_DIR}")
    return best_metrics


if __name__ == "__main__":
    main()
