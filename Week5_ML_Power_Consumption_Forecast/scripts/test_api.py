#!/usr/bin/env python3
"""Simple smoke test for the forecast API (run while uvicorn is up)."""

import json
import sys

import requests

BASE = "http://127.0.0.1:8000"

sample = {
    "hour": 14,
    "dayofweek": 2,
    "month": 7,
    "is_weekend": 0,
    "temperature": 28.5,
    "humidity": 45.0,
    "windspeed": 1.5,
    "general_diffuse_flows": 150.0,
    "diffuse_flows": 20.0,
    "lag_1": 33500.0,
    "lag_2": 33000.0,
    "lag_3": 32500.0,
    "lag_24": 32000.0,
    "rolling_mean_3": 33000.0,
    "rolling_mean_6": 32500.0,
    "rolling_mean_24": 31000.0,
    "rolling_std_24": 2800.0,
}


def main():
    r = requests.get(f"{BASE}/health", timeout=5)
    print("Health:", r.status_code, r.json())
    if r.status_code != 200:
        sys.exit(1)

    r = requests.post(f"{BASE}/predict", json=sample, timeout=10)
    print("Predict:", r.status_code)
    print(json.dumps(r.json(), indent=2))
    if r.status_code != 200:
        sys.exit(1)

    payload = r.json()
    if not isinstance(payload.get("predicted_consumption"), (int, float)):
        print("Prediction response is missing a numeric predicted_consumption.")
        sys.exit(1)
    if payload.get("zone") != "Zone1":
        print("Unexpected prediction zone:", payload.get("zone"))
        sys.exit(1)

    print("API smoke test passed.")


if __name__ == "__main__":
    main()
