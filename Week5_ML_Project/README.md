# Electricity Consumption Forecasting – ML Engineering Track

Predict next-hour mean electricity consumption (kW) for **Zone 1** in Morocco using historical consumption patterns and environmental variables (temperature, humidity, wind speed, diffuse flows).

Built following the same structure and production practices as the loan-default reference project: notebook-style exploration → production training script → FastAPI deployment.

## Business problem

An energy management organisation collects 10-minute electricity readings across three zones together with weather and diffuse-flow measurements. They need:

- Understanding of temporal patterns (hour, day, month)
- Association between environmental conditions and consumption
- A practical forecasting model that can support operational planning
- A deployable prediction endpoint

This project focuses on **Machine Learning Engineering**: data preparation, feature engineering without leakage, XGBoost forecasting, baseline comparison, experiment tracking, and FastAPI deployment.

## Dataset

- `data/powerconsumption.csv` – ~52 416 rows, 10-minute intervals, 1 Jan 2017 – 30 Dec 2017
- `data/powerconsumption_datadictionary.csv`

**Unit interpretation.** The data dictionary labels power columns as kWh. Magnitudes (~20 k–40 k) are consistent with average power (kW) over each 10-minute interval for city-scale zones (Tetouan dataset). We therefore treat readings as power level and aggregate to **hourly mean power (kW)**. This matches common published usage of the dataset and avoids artificially inflating totals by summing interval averages.

**Zone selection.** Zone 1 has the highest average consumption and is used as the forecasting target.

## Project structure

```
power-consumption-forecast/
├── data/
│   ├── powerconsumption.csv
│   └── powerconsumption_datadictionary.csv
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── src/
│   └── train.py                 # full pipeline: prep → features → train → evaluate → save
├── model/                       # created by train.py
│   ├── xgb_model.json             # portable XGBoost model used by the API
│   ├── model_meta.json
│   ├── experiments_log.json
│   └── feature_importance.csv
├── app/
│   ├── main.py                  # FastAPI service
│   └── schemas.py               # Pydantic request/response
├── scripts/
│   └── test_api.py
├── reports/
│   └── technical_report.md
├── requirements.txt
└── README.md
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model

```bash
python src/train.py
```

What the script does:

1. Loads 10-minute data, cleans timestamps, aggregates to hourly mean power
2. Engineers time features, lags (1/2/3/24 h), rolling means/std (shifted to avoid leakage)
3. Chronological 80/20 split (no random shuffle)
4. Trains three XGBoost configurations + lag-1 baseline
5. Logs metrics to `model/experiments_log.json` (and MLflow when the store is available)
6. Saves the best model as portable XGBoost JSON, plus metadata and feature importance under `model/`

**Expected results (indicative):**

| Model              | Test MAE | Test RMSE | Test R²  |
|--------------------|----------|-----------|----------|
| Baseline (lag-1)   | ~1787    | ~2364     | ~0.85    |
| XGBoost (best)     | ~844     | ~1076     | ~0.97    |

Dominant features: `lag_24`, `lag_1`, `hour`.

## 3. Run the API locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive docs: http://localhost:8000/docs  
- Health: http://localhost:8000/health  
- Model info: http://localhost:8000/model-info  

The API loads `model/xgb_model.json` and follows the feature order in `model/model_meta.json`. This avoids Python-pickle compatibility and security problems.

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "rolling_std_24": 2800.0
  }'
```

Smoke test (API must be running):

```bash
python scripts/test_api.py
```

## Answering the business questions (ML perspective)

1. **Patterns across zones** – Zone 1 has the highest average load; all zones show strong diurnal cycles.  
2. **Highest/lowest hours & days** – Peaks typically late afternoon / early evening; weekends differ from weekdays (`is_weekend`, `dayofweek` matter).  
3. **Monthly variation** – Captured by the `month` feature; summer months generally higher (cooling-related load).  
4. **Environmental association** – Temperature and diffuse-flow features contribute; humidity and windspeed have smaller importance.  
5. **Highest/lowest zone** – Zone 1 highest on average.  
6. **Planning support** – Hour-ahead forecasts with ~0.97 R² on held-out chronological data enable better reserve planning, anomaly detection, and demand-response targeting.

## Limitations

- Single-year data (2017); year-to-year climate or load-growth shifts not modelled.  
- One-step (next hour) forecast only; multi-step would need recursive or direct multi-horizon models.  
- Diffuse-flow physical meaning is ambiguous in the supplied dictionary; treated as numeric covariates.  
- No explicit holiday calendar or special-event features.

## Evaluation criteria covered

- Clear business use-case and unit interpretation  
- Data preparation + leakage-free feature engineering  
- Chronological split + baseline comparison  
- Multiple XGBoost experiments with metric tracking  
- Production training script  
- FastAPI endpoint with input validation  
- Technical report and README

## License

Educational project for the Machine Learning Engineering track.
