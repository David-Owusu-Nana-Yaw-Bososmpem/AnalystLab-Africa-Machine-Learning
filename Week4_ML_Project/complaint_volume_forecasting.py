#!/usr/bin/env python3
"""Consumer Complaint Volume Forecasting Prototype - AnalystLab Africa Week 4.

The evaluation uses expanding-window, rolling one-month-ahead forecasts for every
model. At each forecast origin, only observations available before that month are
used for fitting/features. August 2023 is partial (data ends 28 August) and is
excluded from full-month model evaluation and training. The six-month forecast
therefore starts September 2023, using complete months through July 2023.

A complaint is a consumer-reported issue; it does not establish misconduct,
regulatory violations, or company performance.
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("default")
RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent
DATA_CANDIDATES = [BASE / "Consumer_Complaints.xlsx", BASE / "data" / "Consumer_Complaints.xlsx", BASE.parent / "Consumer_Complaints.xlsx"]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError("Consumer_Complaints.xlsx not found. Put it beside this script or in data/.")
OUT = BASE
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
MODELS = OUT / "models"; MODELS.mkdir(exist_ok=True)

TRAIN_END = pd.Period('2021-12', freq='M')
VAL_START, VAL_END = pd.Period('2022-01', 'M'), pd.Period('2022-12', 'M')
TEST_START, TEST_END = pd.Period('2023-01', 'M'), pd.Period('2023-07', 'M')
PARTIAL_LAST_MONTH = pd.Period('2023-08', 'M')


def metric_row(model, split, actual, predicted):
    y = np.asarray(actual, dtype=float); p = np.asarray(predicted, dtype=float)
    err = y - p
    denom = np.abs(y).sum()
    return {"model": model, "split": split, "n_months": len(y),
            "MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "WAPE_pct": float(np.abs(err).sum() / denom * 100) if denom else np.nan,
            "MAPE_pct": float(np.mean(np.abs(err[y != 0] / y[y != 0])) * 100) if np.any(y != 0) else np.nan,
            "Mean_Error_actual_minus_predicted": float(err.mean())}


def make_feature_row(history, target_period):
    """Features for target month, derived only from prior observed/predicted values."""
    vals = list(map(float, history))
    if len(vals) < 12: raise ValueError("At least 12 prior months are required")
    t = len(vals)
    m = target_period.month
    return {"time_index": t, "time_index_sq": t*t,
            "month_sin": np.sin(2*np.pi*m/12), "month_cos": np.cos(2*np.pi*m/12),
            "quarter": target_period.quarter,
            "lag_1": vals[-1], "lag_2": vals[-2], "lag_3": vals[-3],
            "lag_6": vals[-6], "lag_12": vals[-12],
            "rolling_mean_3": float(np.mean(vals[-3:])),
            "rolling_mean_6": float(np.mean(vals[-6:])),
            "rolling_std_3": float(np.std(vals[-3:], ddof=1))}

FEATURES = ["time_index", "time_index_sq", "month_sin", "month_cos", "quarter", "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "rolling_mean_3", "rolling_mean_6", "rolling_std_3"]


def supervised_from_series(series):
    rows=[]; targets=[]; periods=[]
    for i in range(12, len(series)):
        period = series.index[i]
        rows.append(make_feature_row(series.iloc[:i].tolist(), period))
        targets.append(float(series.iloc[i])); periods.append(period)
    return pd.DataFrame(rows, index=periods)[FEATURES], pd.Series(targets, index=periods, dtype=float)


def fit_predict_ml(model, history, target_period):
    X_all, y_all = supervised_from_series(history)
    x_next = pd.DataFrame([make_feature_row(history.tolist(), target_period)])[FEATURES]
    fitted = clone(model).fit(X_all, y_all)
    return float(max(0, fitted.predict(x_next)[0]))


def fit_predict_hw(history, steps=1):
    fit = ExponentialSmoothing(history.astype(float), trend='add', seasonal='add', seasonal_periods=12,
                               initialization_method='estimated').fit(optimized=True)
    return np.asarray(fit.forecast(steps), dtype=float)


def choose_arima_order(train_series):
    best_aic, best_order = np.inf, None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    fit = ARIMA(train_series.astype(float), order=(p,d,q)).fit()
                    if np.isfinite(fit.aic) and fit.aic < best_aic:
                        best_aic, best_order = fit.aic, (p,d,q)
                except Exception:
                    continue
    return best_order


def forecast_one(model_name, history, period, arima_order, ml_models):
    h = history.astype(float)
    if model_name == "Naive": return float(h.iloc[-1])
    if model_name == "Seasonal Naive": return float(h.iloc[-12]) if len(h) >= 12 else float(h.iloc[-1])
    if model_name == "Moving Average (3)": return float(h.tail(3).mean())
    if model_name == "Holt-Winters": return float(fit_predict_hw(h, 1)[0])
    if model_name == "ARIMA":
        if arima_order is None: raise RuntimeError("ARIMA order could not be estimated")
        return float(max(0, ARIMA(h, order=arima_order).fit().forecast(1).iloc[0]))
    return fit_predict_ml(ml_models[model_name], h, period)


def main():
    print(f"Loading: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    required = ["Complaint ID", "Date submitted", "Date received", "Product"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols: raise ValueError(f"Required columns missing: {missing_cols}")
    df["Date submitted"] = pd.to_datetime(df["Date submitted"], errors="coerce")
    df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
    if df["Complaint ID"].duplicated().any():
        raise ValueError(f"Duplicate Complaint IDs found: {df['Complaint ID'].duplicated().sum()}")
    if df["Date submitted"].isna().any():
        raise ValueError(f"Rows with invalid/missing submission date: {df['Date submitted'].isna().sum()}")
    invalid_order = (df["Date received"].notna() & (df["Date received"] < df["Date submitted"])).sum()
    date_min, date_max = df["Date submitted"].min(), df["Date submitted"].max()
    df.to_csv(OUT / "cleaned_complaints.csv", index=False)

    # Monthly counts and explicit gap audit. Missing periods are NOT silently assumed zero.
    df["year_month"] = df["Date submitted"].dt.to_period("M")
    observed = df.groupby("year_month").size().sort_index()
    full_idx = pd.period_range(observed.index.min(), observed.index.max(), freq="M")
    gaps = full_idx.difference(observed.index)
    audit = pd.DataFrame({"month": full_idx.astype(str), "observed_month": full_idx.isin(observed.index),
                          "complaints": [int(observed[p]) if p in observed.index else np.nan for p in full_idx]})
    audit["gap_status"] = np.where(audit.observed_month, "Observed month", "Missing month - investigate; not imputed")
    audit.to_csv(OUT / "monthly_reporting_period_audit.csv", index=False)
    if len(gaps):
        raise ValueError("Missing reporting months found. Investigate monthly_reporting_period_audit.csv; no zero imputation performed.")
    monthly = observed.reindex(full_idx).astype(int).rename("complaints").to_frame()
    monthly["date"] = monthly.index.to_timestamp()
    monthly["year"] = monthly.index.year; monthly["month"] = monthly.index.month; monthly["quarter"] = monthly.index.quarter
    monthly.reset_index(names="year_month").to_csv(OUT / "monthly_complaints.csv", index=False)

    # Partial period policy: dataset ends 28 Aug 2023; do not compare partial Aug count with full months.
    monthly_full = monthly.loc[monthly.index < PARTIAL_LAST_MONTH].copy()
    if monthly_full.index.max() < TEST_END: raise ValueError("Dataset does not cover required evaluation period through July 2023")
    monthly_full = monthly_full.loc[:TEST_END]
    series = monthly_full["complaints"].astype(float)
    train = series.loc[:TRAIN_END]
    val_periods = pd.period_range(VAL_START, VAL_END, freq="M")
    test_periods = pd.period_range(TEST_START, TEST_END, freq="M")
    if len(train) < 24 or len(val_periods) != 12 or len(test_periods) != 7:
        raise ValueError("Unexpected data coverage for the configured chronological splits")

    # May 2017 is boundary month: starts on first calendar day, but completeness of source extraction
    # cannot be independently established. Report sensitivity rather than claiming it is complete.
    boundary_note = "May 2017 begins on day 1, but complete source coverage cannot be proven from dates alone; flagged for sensitivity analysis."
    pd.DataFrame([
        {"check":"Complaint ID uniqueness","result": "PASS"},
        {"check":"Missing submission dates","result": str(int(df['Date submitted'].isna().sum()))},
        {"check":"Duplicate Complaint IDs","result": str(int(df['Complaint ID'].duplicated().sum()))},
        {"check":"Date received before submitted","result": str(int(invalid_order))},
        {"check":"Source date range","result": f"{date_min.date()} to {date_max.date()}"},
        {"check":"Missing monthly periods","result": str(len(gaps))},
        {"check":"May 2017 coverage","result": boundary_note},
        {"check":"August 2023","result": f"Partial month; data ends {date_max.date()}; excluded from full-month modelling"},
    ]).to_csv(OUT / "data_quality_report.csv", index=False)

    # Save the monthly series without engineered features as the canonical target.
    feature_table = []
    for i in range(12, len(series)):
        feature_table.append({"year_month": str(series.index[i]), **make_feature_row(series.iloc[:i].tolist(), series.index[i]), "target_complaints": series.iloc[i]})
    pd.DataFrame(feature_table).to_csv(OUT / "modeling_features.csv", index=False)

    ml_models = {
        "Linear Regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "Ridge Regression": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=4, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.03, loss="huber", random_state=RANDOM_STATE),
    }
    # ARIMA is omitted from the default rolling loop to keep this reproducible
    # pipeline practical on standard internship laptops. Holt-Winters plus four
    # regression approaches provide the required method comparison.
    arima_order = None
    candidates = ["Naive", "Seasonal Naive", "Moving Average (3)", "Holt-Winters", *ml_models.keys()]
    predictions=[]; metrics=[]

    # Every model gets the same expanding-window one-step-ahead evaluation protocol.
    for split_name, periods in [("Validation", val_periods), ("Test", test_periods)]:
        for model_name in candidates:
            history = series.loc[:TRAIN_END].copy()
            if split_name == "Test": history = series.loc[:VAL_END].copy()
            for period in periods:
                actual = float(series.loc[period])
                try:
                    pred = forecast_one(model_name, history, period, arima_order, ml_models)
                except Exception as exc:
                    print(f"WARNING: {model_name} failed at {period}: {exc}")
                    pred = np.nan
                predictions.append({"model":model_name,"split":split_name,"year_month":str(period),"actual":actual,"predicted":pred,
                                    "error_actual_minus_predicted":actual-pred if np.isfinite(pred) else np.nan,
                                    "absolute_error":abs(actual-pred) if np.isfinite(pred) else np.nan})
                # For rolling one-step-ahead, actual becomes available before next month forecast.
                history.loc[period] = actual
            frame = pd.DataFrame([r for r in predictions if r["model"]==model_name and r["split"]==split_name])
            if frame["predicted"].notna().all(): metrics.append(metric_row(model_name, split_name, frame.actual, frame.predicted))
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUT / "forecast_predictions_by_month.csv", index=False)
    metrics_df = pd.DataFrame(metrics).sort_values(["split","MAE","RMSE"])
    metrics_df.to_csv(OUT / "model_evaluation_metrics.csv", index=False)
    if not (metrics_df.split == "Validation").any(): raise RuntimeError("No valid validation metrics generated")
    val_rank = metrics_df[metrics_df.split=="Validation"].sort_values(["MAE","RMSE"])
    selected_name = str(val_rank.iloc[0].model)
    selected_mae = float(val_rank.iloc[0].MAE)
    print(f"Selected using validation MAE (tie-break RMSE): {selected_name} (MAE={selected_mae:.2f})")
    # Test results are reported only after model selection; test is not used to choose the model.
    test_selected = metrics_df[(metrics_df.split=="Test") & (metrics_df.model==selected_name)]

    # Boundary-month sensitivity: repeat the selected model evaluation after
    # excluding May 2017. This tests whether the first observed month materially
    # changes results; it does not prove whether the source extract is complete.
    sensitivity_rows = []
    if selected_name == "Holt-Winters":
        sensitivity_series = series.iloc[1:].copy()
        for split_name, periods in [("Validation", val_periods), ("Test", test_periods)]:
            hist = sensitivity_series.loc[:TRAIN_END] if split_name == "Validation" else sensitivity_series.loc[:VAL_END]
            preds_s, actual_s = [], []
            for per in periods:
                actual_s.append(float(sensitivity_series.loc[per]))
                preds_s.append(float(fit_predict_hw(hist, 1)[0]))
                hist.loc[per] = actual_s[-1]
            sensitivity_rows.append(metric_row("Holt-Winters (May 2017 excluded)", split_name, actual_s, preds_s))
    if sensitivity_rows:
        pd.DataFrame(sensitivity_rows).to_csv(OUT / "may_2017_sensitivity_analysis.csv", index=False)

    # Refit selected model through July 2023. The source ends on 28 August,
    # so August is incomplete and is not a valid full-month target. Forecast it
    # as an intermediate step, then retain September 2023 through February 2024.
    history = series.copy()
    intermediate_periods = pd.period_range("2023-08", periods=7, freq="M")
    future_periods = intermediate_periods[1:]
    all_future = []
    if selected_name in ml_models:
        X_all, y_all = supervised_from_series(history)
        fitted_final = clone(ml_models[selected_name]).fit(X_all, y_all)
        for p in intermediate_periods:
            x = pd.DataFrame([make_feature_row(history.tolist(), p)])[FEATURES]
            value = max(0.0, float(fitted_final.predict(x)[0]))
            all_future.append(value)
            history.loc[p] = value
        joblib.dump(fitted_final, MODELS / "selected_forecasting_model.joblib")
    elif selected_name == "Holt-Winters":
        fitted_final = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12, initialization_method='estimated').fit(optimized=True)
        # forecast(7) starts in August; discard the intermediate August estimate.
        all_future = list(map(float, fitted_final.forecast(len(intermediate_periods))))
        joblib.dump(fitted_final, MODELS / "selected_forecasting_model.joblib")
    else:
        # Recursive baseline forecast, including the intermediate August estimate.
        for p in intermediate_periods:
            value = forecast_one(selected_name, history, p, arima_order, ml_models)
            all_future.append(value)
            history.loc[p] = value
        joblib.dump({"model":selected_name,"history":series.to_dict()}, MODELS / "selected_forecasting_model.joblib")
    # Keep only the six requested complete-month forecasts (September-February).
    future = np.maximum(0, np.asarray(all_future[1:], dtype=float))
    if len(future) != len(future_periods):
        raise RuntimeError("Future forecast length does not match the requested six-month horizon")
    # Empirical validation absolute-error quantile: heuristic band, NOT a calibrated prediction interval.
    val_err = pred_df[(pred_df.model==selected_name)&(pred_df.split=="Validation")].absolute_error.dropna().to_numpy()
    q90 = float(np.quantile(val_err, 0.90)) if len(val_err) else np.nan
    forecast_df = pd.DataFrame({"year_month":future_periods.astype(str),"forecast_complaints":np.rint(future).astype(int),
                                "lower_empirical_error_band":np.maximum(0,np.floor(future-q90)).astype(int) if np.isfinite(q90) else np.nan,
                                "upper_empirical_error_band":np.ceil(future+q90).astype(int) if np.isfinite(q90) else np.nan,
                                "selected_model":selected_name,"band_note":"Heuristic +/- 90th percentile validation absolute error; not a calibrated prediction interval"})
    forecast_df.to_csv(OUT / "forward_forecast_6m.csv", index=False)

    # Residual/error analysis for selected model, validation and test.
    selected_preds = pred_df[pred_df.model==selected_name].copy()
    selected_preds.to_csv(OUT / "selected_model_error_analysis.csv", index=False)
    err = selected_preds.dropna(subset=["predicted"])
    if len(err):
        err.groupby("split").agg(months=("year_month","count"),mean_error=("error_actual_minus_predicted","mean"),
            mean_absolute_error=("absolute_error","mean"),max_absolute_error=("absolute_error","max")).reset_index().to_csv(OUT/"error_summary.csv",index=False)
        worst = err.sort_values("absolute_error",ascending=False).groupby("split").head(3)
        worst.to_csv(OUT/"largest_forecast_errors.csv",index=False)

    # Product series are descriptive; separate product forecasting is not claimed.
    top_products = df["Product"].fillna("Unknown").value_counts().head(5).index
    product_monthly = (df[df["Product"].fillna("Unknown").isin(top_products)]
        .assign(month=lambda x:x["Date submitted"].dt.to_period("M"))
        .loc[lambda x: x["month"] < PARTIAL_LAST_MONTH]
        .groupby(["month","Product"]).size().unstack(fill_value=0))
    product_monthly.to_csv(OUT/"top_product_monthly_complaints.csv")

    # Charts
    plt.figure(figsize=(12,5)); plt.plot(series.index.to_timestamp(),series.values,marker='o',ms=3)
    plt.title("Monthly consumer complaint volume (complete months through July 2023)"); plt.xlabel("Month"); plt.ylabel("Complaints"); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(FIG/"01_monthly_volume_history.png",dpi=160); plt.close()
    fig,ax=plt.subplots(figsize=(12,5)); ax.plot(series.index.to_timestamp(),series.values,label="Actual history")
    ax.plot(future_periods.to_timestamp(),future,label=f"6-month forecast: {selected_name}",marker='o')
    if np.isfinite(q90): ax.fill_between(future_periods.to_timestamp(),forecast_df.lower_empirical_error_band,forecast_df.upper_empirical_error_band,alpha=.18,label="Heuristic validation-error band")
    ax.axvline(series.index[-1].to_timestamp(),ls='--',color='grey',label='Last complete month')
    ax.set_title("Historical complaints and six-month forecast"); ax.set_xlabel("Month"); ax.set_ylabel("Complaints"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG/"02_future_forecast.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5)); plot=err[err.split=="Test"]
    if len(plot): ax.bar(plot.year_month,plot.error_actual_minus_predicted); ax.axhline(0,color='black',lw=.8)
    ax.set_title(f"Test-period forecast errors: {selected_name}"); ax.set_xlabel("Month"); ax.set_ylabel("Actual minus predicted"); plt.xticks(rotation=45); fig.tight_layout(); fig.savefig(FIG/"03_test_forecast_errors.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5)); top=metrics_df[metrics_df.split=="Test"].sort_values("MAE")
    if len(top): ax.barh(top.model,top.MAE); ax.invert_yaxis()
    ax.set_title("Test-set MAE by model (rolling one-step-ahead)"); ax.set_xlabel("MAE (complaints)"); fig.tight_layout(); fig.savefig(FIG/"04_test_model_comparison.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    for c in product_monthly.columns: ax.plot(product_monthly.index.to_timestamp(),product_monthly[c],label=str(c))
    ax.set_title("Monthly complaints for five highest-volume products"); ax.set_xlabel("Month"); ax.set_ylabel("Complaints"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/"05_top_product_trends.png",dpi=160); plt.close(fig)

    summary = {"generated_utc":datetime.now(timezone.utc).isoformat(),"data_file":DATA_PATH.name,"records":int(len(df)),
      "unique_complaint_ids":int(df["Complaint ID"].nunique()),"source_start":str(date_min.date()),"source_end":str(date_max.date()),
      "monthly_observations":int(len(monthly)),"missing_months":int(len(gaps)),"invalid_forwarding_date_order":int(invalid_order),
      "train_period":f"{train.index.min()} to {train.index.max()}","validation_period":f"{VAL_START} to {VAL_END}",
      "test_period":f"{TEST_START} to {TEST_END}","partial_month_excluded":str(PARTIAL_LAST_MONTH),
      "partial_first_month_note":boundary_note,"arima_order_selected_on_training_aic":None, "arima_note":"ARIMA omitted from default run for runtime; Holt-Winters and ML regressors are compared.",
      "selection_rule":"Lowest validation MAE; RMSE tie-break; test set not used for selection",
      "selected_model":selected_name,"validation_mae":selected_mae,
      "test_metrics":test_selected.iloc[0].to_dict() if len(test_selected) else None,
      "intermediate_forecast_month":str(intermediate_periods[0]),
      "future_forecast_period":f"{future_periods[0]} to {future_periods[-1]}",
      "forecast_alignment_note":"August 2023 is forecast as an intermediate month to advance the model origin; only September 2023-February 2024 is reported as the six-month future forecast.",
      "uncertainty_note":"Empirical validation absolute-error band is a heuristic, not a calibrated prediction interval."}
    with open(MODELS/"run_summary.json","w",encoding="utf-8") as f: json.dump(summary,f,indent=2,default=str)
    print("\nPipeline completed.")
    print(f"Selected model: {selected_name}")
    print(f"Forecast period: {future_periods[0]} to {future_periods[-1]}")
    print("Outputs: metrics, monthly audit, prediction errors, forecast, model artefact and figures.")

if __name__ == "__main__": main()
