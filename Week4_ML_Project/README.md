# Consumer Complaint Volume Forecasting Prototype

**AnalystLab Africa – Experience Lab Internship Programme**  
**Week 4 – Machine Learning Track**

## 1. Project purpose

This project builds a reproducible prototype to forecast monthly counts of consumer complaints recorded against Bank of America in the supplied CFPB extract. The target is the number of complaints submitted in each calendar month.

A complaint is a consumer-reported issue. It does not by itself establish misconduct, a regulatory violation, or company performance. The model estimates complaint volume only.

## 2. Prediction definition

| Item | Definition |
|---|---|
| Prediction target | Monthly complaint count, aggregated from `Date submitted` |
| Unit | Complaints per calendar month |
| Main evaluation approach | Expanding-window rolling one-month-ahead forecast |
| Training period | May 2017 – December 2021 |
| Validation period | January – December 2022 |
| Test period | January – July 2023 |
| Partial final month | August 2023 (data ends 28 August); excluded from full-month model evaluation and fitting |
| Future forecast | September 2023 – February 2024 |

**Important on forecast origin:** The supplied extract ends on 28 August 2023, but August is incomplete. To avoid treating a partial month as a full monthly count, the final forecasting model is fitted using complete months through July 2023. The model first generates an intermediate August 2023 estimate to advance the forecast origin, then reports September 2023 to February 2024 as the six-month forecast. The August estimate is not included in the reported six-month forecast or used as an observed value. The August partial count is not mixed into the monthly target series used for model fitting.

May 2017 begins on the first calendar day, but the date range alone cannot prove that the source extract contains every complaint for that month. The pipeline flags this uncertainty and performs a Holt-Winters sensitivity run excluding May 2017.

## 3. Data preparation and validation

The script:

1. Reads the supplied `Consumer_Complaints.xlsx` file.
2. Parses submission and forwarding dates with invalid dates set to missing for inspection.
3. Checks that `Complaint ID` is unique and that submission dates are present.
4. Checks whether a forwarding date precedes the submission date.
5. Aggregates complaints by calendar month.
6. Creates a complete monthly calendar and writes a reporting-period audit.
7. Does **not** automatically replace missing months with zero. If a month is missing, the script stops and asks for investigation because a gap may represent missing coverage rather than a genuine zero count.
8. Excludes August 2023 from full-month evaluation because it is partial.
9. Saves the monthly target series and data-quality report.

The supplied extract contains 62,516 records and 62,516 unique complaint IDs. Its submission-date range is 1 May 2017 to 28 August 2023. The generated audit reports 76 observed calendar months and no missing months in that span. The supplied data contains no submission-date values that precede the forwarding date in the reverse direction (the check for `Date received` earlier than `Date submitted` returned zero).

## 4. Features and models

The machine-learning regressors use only information available before the target month:

- Calendar seasonality: month sine and cosine, quarter.
- Time trend: time index and squared time index.
- Lagged complaint counts: 1, 2, 3, 6 and 12 months.
- Rolling mean: previous 3 and 6 months.
- Rolling standard deviation: previous 3 months.

The comparison includes:

**Baselines**
- Naive: previous observed month.
- Seasonal naive: same month in the previous year.
- Three-month moving average.

**Statistical forecasting**
- Holt-Winters additive trend and additive seasonality (12-month season).

**Machine-learning regression**
- Linear Regression with standardised features.
- Ridge Regression with standardised features.
- Random Forest Regressor.
- Gradient Boosting Regressor.

ARIMA is not included in the default executable pipeline. It was removed from the default rolling evaluation to keep execution practical on ordinary laptops. The required comparison is still provided through the baselines, Holt-Winters and four machine-learning regressors. It can be added as an optional extension if execution time and library convergence are acceptable.

## 5. Evaluation design and leakage prevention

Every model is assessed using the same expanding-window rolling one-month-ahead procedure:

- To predict a month, the model sees only observations before that month.
- Once the month is predicted, its actual value is added to history before forecasting the next month.
- Validation is January–December 2022, starting from the training history through December 2021.
- Test is January–July 2023, starting from history through December 2022.
- Test results are not used to select the model.

This is a rolling one-step-ahead use case, not a fixed-origin forecast of all 12 validation or 7 test months at once. It assumes the previous month's actual complaint count becomes available before the next month's forecast is made. A fixed-origin multi-month forecast would be a different operational scenario and should be evaluated separately.

Metrics:
- **MAE:** average absolute error in complaints; lower is better.
- **RMSE:** penalises large errors more heavily; lower is better.
- **WAPE:** sum of absolute errors divided by sum of actual counts; lower is better.
- **MAPE:** mean absolute percentage error for non-zero actual months; lower is better.
- **Mean error:** average of actual minus predicted; positive values indicate average under-forecasting.

The final model is selected using the lowest validation MAE, with RMSE as a tie-breaker. In this run, Holt-Winters was selected. Its test metrics are reported as an out-of-sample check.

## 6. Results from the generated run

The script-generated `model_evaluation_metrics.csv` is the authoritative output. In this run:

| Model | Validation MAE | Test MAE | Test RMSE | Test WAPE |
|---|---:|---:|---:|---:|
| Holt-Winters | 71.78 | 165.08 | 279.62 | 13.73% |
| Ridge Regression | 76.04 | 164.61 | 251.04 | 13.69% |
| Linear Regression | 81.68 | 168.72 | 272.35 | 14.04% |
| Naive | 82.75 | 180.29 | 292.11 | 15.00% |
| Random Forest | 87.86 | 155.90 | 260.65 | 12.97% |
| Moving Average (3) | 94.19 | 164.19 | 273.17 | 13.66% |
| Gradient Boosting | 114.91 | 141.48 | 265.43 | 11.77% |
| Seasonal Naive | 152.00 | 197.29 | 290.61 | 16.41% |

Holt-Winters was selected because it had the lowest validation MAE. Gradient Boosting achieved the lowest test MAE, but test results must not be used to retroactively change the validation-based model selection. This difference is a useful reminder that the best validation model may not be the best model on a later period.

The selected Holt-Winters model's mean test error is positive (about 72.55 complaints), indicating under-forecasting on average in the test months. The test period is short (seven months), so the ranking should not be treated as definitive evidence of long-term superiority.

### May 2017 sensitivity check

`may_2017_sensitivity_analysis.csv` repeats the selected Holt-Winters rolling evaluation after removing May 2017. Validation MAE changes from 71.78 to 71.34; test MAE changes from 165.08 to 165.78. These small differences suggest that this particular model's reported errors are not highly sensitive to the first month, but do not prove the source coverage of May 2017.

## 7. Future forecast and uncertainty

`forward_forecast_6m.csv` contains forecasts for September 2023 to February 2024. It also contains a lower and upper **heuristic empirical-error band** based on the 90th percentile of validation absolute errors.

These bands are not formal, calibrated prediction intervals and should not be interpreted as guaranteed bounds. The model has no external explanatory variables such as product changes, economic conditions, policy changes, campaigns or unusual events. Forecasts should be treated as planning estimates and updated when new complete-month data becomes available.

## 8. Error analysis and business interpretation

The top-product monthly trend file and its chart exclude August 2023 because it is a partial reporting month. Product-level monthly counts are descriptive only; they are not separate product forecasts.

The pipeline creates:

- `forecast_predictions_by_month.csv`: actual, predicted, signed error and absolute error for every model and validation/test month.
- `selected_model_error_analysis.csv`: month-by-month errors for the selected model.
- `error_summary.csv`: mean error, MAE and maximum absolute error by split.
- `largest_forecast_errors.csv`: three largest absolute errors per split.

Positive `error_actual_minus_predicted` means the model under-predicted; negative means it over-predicted. A large error may arise from a sudden change in complaint volume that historical patterns alone could not anticipate. Operationally, the forecast may help with preliminary complaint-handling capacity planning, but it should not be used as a standalone staffing or performance decision.

## 9. Outputs

| File | Purpose |
|---|---|
| `cleaned_complaints.csv` | Reproducible cleaned copy of source records with parsed dates |
| `data_quality_report.csv` | Primary-key, date and boundary-period checks |
| `monthly_reporting_period_audit.csv` | Observed/missing calendar-month audit; gaps are not silently imputed |
| `monthly_complaints.csv` | Monthly complaint target series |
| `modeling_features.csv` | Lagged and calendar features with target |
| `model_evaluation_metrics.csv` | Metrics by model and split |
| `forecast_predictions_by_month.csv` | Actual and predicted values for every evaluated month |
| `selected_model_error_analysis.csv` | Selected model residual details |
| `error_summary.csv` | Error summary by split |
| `largest_forecast_errors.csv` | Largest monthly errors |
| `may_2017_sensitivity_analysis.csv` | Selected model evaluation excluding May 2017 |
| `forward_forecast_6m.csv` | Six-month future forecast and heuristic error band |
| `top_product_monthly_complaints.csv` | Descriptive monthly counts for top product categories; not separate product forecasts |
| `models/selected_forecasting_model.joblib` | Saved fitted selected model |
| `models/run_summary.json` | Machine-readable run metadata and selected-model summary |
| `tests/test_forecasting.py` | Automated checks for past-only features and forecast-period alignment |
| `figures/` | Generated charts |

## 10. Run the project on Windows (VS Code)

Open a terminal in this project folder and run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python complaint_volume_forecasting.py
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

After closing VS Code, you do **not** need to create another virtual environment. Reopen the folder, open a terminal, activate the existing `.venv`, and run the script again. If the environment is already selected as the VS Code interpreter, VS Code may activate it automatically in a new terminal.

The Excel source file should remain beside the script or inside a `data/` subfolder. Outputs are written into this project folder and its `figures/` and `models/` subfolders. Run the automated checks with `python -m pytest -q` before running the full pipeline.

## 11. Automated tests

The project includes four pytest checks covering lag and rolling-feature construction, separation of features from the current target, the September 2023–February 2024 forecast labels, and non-negative forecast output. Run them with:

```powershell
python -m pytest -q
```

The tests are lightweight checks of key forecasting logic. They do not replace a full independent review of model assumptions or forecast quality.

## 12. Limitations and next steps

1. The time series is relatively short (about 75 complete months through July 2023).
2. August 2023 is a partial month and is not included in full-month model evaluation.
3. May 2017 source completeness cannot be established from the date range alone; a sensitivity analysis is supplied.
4. Rolling one-step-ahead evaluation assumes the previous month's actual volume is available at each forecast origin.
5. No external drivers are included, so sudden structural changes cannot be anticipated reliably.
6. Product-level monthly counts are provided descriptively. Separate product forecasting requires sufficient history and separate validation.
7. The empirical error band is heuristic, not a calibrated prediction interval.
8. The test set contains only seven months; additional future observations are needed for stronger conclusions.

Recommended future work: compare against a fixed-origin multi-step strategy, add rolling-origin backtesting over multiple historical cutoffs, investigate product-level forecasting only for sufficiently populated series, and recalibrate uncertainty estimates using a larger validation history.

## 13. Assignment deliverables coverage

- [x] Defined monthly complaint-volume target.
- [x] Monthly aggregation and missing-period investigation.
- [x] Time-based and past-only lag/rolling features.
- [x] Chronological train, validation and test periods.
- [x] Baseline and multiple forecasting approaches.
- [x] Consistent rolling one-step-ahead evaluation.
- [x] MAE, RMSE, WAPE and MAPE.
- [x] Actual-versus-predicted values and error analysis.
- [x] Six-month forecast visualisation and saved model artefact.
- [x] Intermediate August forecast is discarded; reported forecast labels align with September 2023–February 2024.
- [x] Partial August 2023 is excluded from the top-product trend chart and CSV.
- [x] Automated unit tests for past-only features and forecast-period alignment.
- [x] Limitations and practical interpretation.

---

**Data source:** Consumer Financial Protection Bureau (CFPB), Bank of America complaint extract supplied for the AnalystLab Africa Week 4 assignment.
