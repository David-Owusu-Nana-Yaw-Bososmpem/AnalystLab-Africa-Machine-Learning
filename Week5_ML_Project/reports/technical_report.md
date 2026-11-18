# Technical Report – Electricity Consumption Forecasting (Zone 1)

## 1. Objective

Develop and deploy an hour-ahead electricity consumption forecasting model for one of three zones in Morocco, using historical load and environmental measurements. The solution must follow production ML practices: chronological validation, leakage-free features, baseline comparison, experiment tracking, and a FastAPI prediction service.

## 2. Data preparation

- Source: `powerconsumption.csv` (~52 416 rows, 10-minute resolution, full year 2017).
- Cleaning: parsed timestamps, removed duplicate timestamps, forward/back filled residual missing values (very few).
- **Aggregation**: resampled to hourly mean.  
  **Rationale for mean rather than sum**: data-dictionary units say “kWh”, yet magnitudes are consistent with average power (kW) for a city zone. Summing six successive 10-minute averages would double-count the power interpretation. Published analyses of the Tetouan dataset also treat the series as power. We therefore forecast hourly mean power (kW).
- Target: **Zone 1** (highest average consumption ≈ 32 356 kW).
- Resulting series: 8 712 hourly observations after feature engineering (lags require history).

## 3. Feature engineering

All features use only information available at prediction time.

| Feature group        | Features                                      | Why selected                                      |
|----------------------|-----------------------------------------------|---------------------------------------------------|
| Calendar             | hour, dayofweek, month, is_weekend            | Strong diurnal and weekly seasonality             |
| Weather / environment| temperature, humidity, windspeed, general_diffuse_flows, diffuse_flows | Physical drivers of load (cooling, lighting)     |
| Autoregressive       | lag_1, lag_2, lag_3, lag_24                   | Strong persistence and daily seasonality          |
| Rolling statistics   | rolling_mean_3/6/24, rolling_std_24 (shifted) | Recent level and volatility without leakage       |

## 4. Modelling & evaluation

- **Split**: chronological 80 / 20 (train ends before test begins). No random shuffle.
- **Baseline**: lag-1 persistence (predict next hour = last hour).  
  MAE ≈ 1 787, RMSE ≈ 2 364, R² ≈ 0.85.
- **Model family**: XGBoost regressor. Three hyper-parameter configurations were trained:
  - default (depth 6, 200 trees)
  - deeper (depth 8, 400 trees)
  - regularised (depth 5, L1/L2, 300 trees)
- **Best model**: regularised configuration.  
  Test MAE ≈ 844, RMSE ≈ 1 076, R² ≈ 0.969.  
  Approximately 55 % reduction in RMSE versus baseline.
- Dominant features (gain importance): lag_24, lag_1, hour. Environmental variables contribute modestly but improve accuracy.

Experiment metrics and parameters are recorded in `model/experiments_log.json`. When the runtime allows, the same runs are also logged to MLflow under experiment `power_consumption_forecast`.

## 5. Deployment

- FastAPI application (`app/main.py`) loads the portable XGBoost JSON model at startup and checks its feature count against `model_meta.json`.
- Request body validated by Pydantic (`ForecastRequest`); invalid inputs, including inconsistent weekday/weekend values, return 422.
- Endpoint `POST /predict` returns predicted hourly mean power (kW) for Zone 1.
- Additional endpoints: `/health`, `/model-info`.

## 6. Business answers (summary)

1. Clear diurnal and weekly patterns; Zone 1 carries the largest load.  
2. Highest consumption typically late afternoon / early evening; lowest in early morning.  
3. Monthly variation is present and captured by the `month` feature (summer peaks).  
4. Temperature and diffuse-flow measurements show measurable association; humidity and windspeed weaker.  
5. Zone 1 highest, Zone 2 lowest on average in this year.  
6. Hour-ahead forecasts with R² ≈ 0.97 support reserve planning, anomaly flagging, and demand-response scheduling.

## 7. Limitations & future work

- Single calendar year; climate or load-growth drift not modelled.  
- One-step forecast only.  
- Diffuse-flow variables lack clear physical documentation.  
- Possible extensions: multi-horizon forecasting, holiday features, online learning, zone-specific or multi-output models.

## 8. Reproducibility

```bash
pip install -r requirements.txt
python src/train.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

All metrics, feature importances and model artefacts are written under `model/`.
