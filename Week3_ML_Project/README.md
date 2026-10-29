# Healthcare Operations: Predictive Model for Encounter Total Claim Cost

**Track C – Machine Learning Engineering**  
AnalystLab Africa | Healthcare Intelligence Project

## 1. Problem Selection & Justification

**Selected problem:** Predict the **total claim cost** of a hospital encounter (regression).

**Why this problem is feasible and operationally relevant**
- Target `TOTAL_CLAIM_COST` is present and non-null for all 27,891 encounters.
- Sufficient volume for supervised learning.
- Operational value: at the point an encounter begins (or is scheduled), an estimate of expected cost supports capacity planning, financial forecasting, and early identification of potentially high-cost cases for care management.
- Alternative candidates were examined and rejected or deprioritised:
  - Duration > 24 h: only ~75 positive cases (0.27 %) → severe class imbalance + data-quality outliers (multi-year ambulatory records).
  - Zero payer coverage: highly determined by the known `PAYER` field (NO_INSURANCE always zero, Medicaid/Medicare almost never zero) → near-trivial once payer is included; less interesting as a pure predictive exercise.
  - 30-day repeat encounter: constructible but requires careful definition of observation windows, handling of overlapping encounters (negative inter-arrival times exist), and risks treating every subsequent visit as a clinical readmission (explicitly discouraged).

**Prediction point:** Features known or knowable at the **start** of the encounter (or at scheduling time).  
No information generated *during* or *after* the encounter is used as a feature.

## 2. Target & Features

### Target
- `TOTAL_CLAIM_COST` (USD). Continuous, right-skewed (mean ≈ $3.6k, median ≈ $279, max > $600k).
- Also computed `LOG_TOTAL_COST = log1p(TOTAL_CLAIM_COST)` for exploratory analysis; models were trained on the original scale with non-negativity clipping at inference.

### Features available at prediction time
| Feature | Type | Notes |
|---------|------|-------|
| ENCOUNTERCLASS | categorical | ambulatory, outpatient, urgentcare, emergency, wellness, inpatient |
| PAYER_NAME | categorical | 10 payers including NO_INSURANCE |
| GENDER | categorical | M / F |
| RACE | categorical | white, black, asian, … |
| ETHNICITY | categorical | hispanic / nonhispanic |
| MARITAL | categorical | M / S / U (filled for 12 missing) |
| AGE_AT_START | numeric | derived from BIRTHDATE + START |
| YEAR, MONTH, DOW, IS_WEEKEND | numeric / binary | calendar features from START |

### Features deliberately excluded (leakage or unavailable)
- `STOP`, `DURATION_HOURS` – outcome of the encounter itself.
- All procedure-level data (counts, base costs, codes) – generated during the encounter.
- `BASE_ENCOUNTER_COST`, `PAYER_COVERAGE`, `REASONCODE` – not reliably known at the pure start / scheduling moment, or would leak total cost information.
- Patient identifiers, names, addresses – privacy and non-predictive for population-level ops model.
- Future information of any kind.

### Missing-data handling
- MARITAL: 12 missing → filled with “U”.
- All other modelling features complete after join.
- No imputation of target; 89 zero-cost encounters retained (valid).

### Limitations of the target definition
- Synthetic data (Synthea-style); cost generation process is not a real claims adjudication engine.
- Single organisation only.
- Extreme right tail driven by rare high-intensity procedure combinations that the model cannot see (by design). Consequently R² remains modest (~0.15).
- Model is **not** a diagnostic, clinical, or treatment system. It is an operational cost-estimation tool only.

## 3. Baseline, Metrics, Validation Strategy

### Baseline
- DummyRegressor (strategy = “mean”) – primary baseline.
- Median predictor also reported (more robust to skew).

### Metrics (regression)
- **MAE** (primary): average absolute dollar error – directly interpretable for finance/ops.
- **RMSE**: penalises large errors more heavily (important because a few extreme cases dominate risk).
- **R²**: proportion of variance explained.
- **Median Absolute Error (MedAE)**: robust central tendency of error.

Why these matter: Hospital finance cares about absolute dollar accuracy more than relative percentage error on low-cost wellness visits. Large under-predictions of high-cost inpatient stays are operationally costly.

### Validation strategy
- **Chronological split**: Train on encounters with START year ≤ 2019 (n = 21,622); Test on 2020–2022 (n = 6,269).
- Rationale: Operational deployment would train on historical data and score future encounters. Random or k-fold CV would leak future information and overstate performance. Year 2022 is partial (only early months).

## 4. Models Trained & Comparison

| Model | Test MAE | Test RMSE | Test R² | Test MedAE |
|-------|----------|-----------|---------|------------|
| Mean baseline | 5,125 | 12,305 | −0.002 | 3,598 |
| Median baseline | 3,083 | 12,616 | −0.053 | 238 |
| Ridge (linear) | 4,391 | 11,991 | 0.048 | 2,811 |
| Random Forest | **3,066** | 11,326 | 0.151 | 1,123 |
| Gradient Boosting | 3,124 | **11,279** | **0.158** | 1,316 |

- Tree ensembles clearly outperform linear models and the mean baseline on MAE and R².
- Median baseline still wins on MedAE because most encounters are low-cost; the ensembles improve the mean error by better modelling the higher-cost classes.
- Feature importance (Random Forest) is dominated by age, gender, encounter class (especially urgentcare / ambulatory / inpatient), and the NO_INSURANCE payer indicator.

## 5. Error Analysis & Limitations

- Largest absolute errors occur on rare ultra-high-cost encounters (inpatient / emergency, often > $100k). The top residual is ~$637k (true cost $642k, predicted ~$5k). These cases are driven by procedures that are invisible to the model by design.
- By encounter class, inpatient has the highest MAE (~$10k) and emergency / urgentcare are next; ambulatory has the lowest.
- Residuals are roughly centred (mean residual ≈ +11) but heavy-tailed.
- **Data leakage check:** All features are constructed exclusively from information available at START; procedure table is never joined into the feature matrix; STOP-derived fields are excluded.
- **Further monitoring needed before any operational use:**
  - Prediction vs actual cost distributions over time (drift).
  - Performance segmented by encounter class and payer.
  - Calibration of high-cost tail (perhaps a separate high-cost classifier or quantile regression).
  - Retraining cadence (annual or when payer contracts / practice patterns change).
  - Human review of predicted high-cost cases rather than automated decisions.

## 6. Reproducible Workflow & Deliverables

```
ml_project/
├── data/
│   └── prepared_encounters.csv      # engineered features + target
├── models/
│   ├── ridge_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   └── gbr_pipeline.joblib
├── reports/
│   ├── metrics.json
│   ├── rf_feature_importance.csv
│   └── test_predictions.csv
├── src/
│   ├── data_prep.py                 # feature engineering & chronological split
│   ├── train.py                     # train baseline + 3 models, save artefacts
│   └── inference.py                 # load model, validate input, predict
├── tests/
│   └── test_pipeline.py             # unit + integration tests (all passing)
└── README.md                        # this file
```

**Reproduce**
```bash
cd ml_project/src
python data_prep.py          # optional, train.py calls it
python train.py              # trains & writes models/ + reports/
python -m pytest ../tests/   # or python ../tests/test_pipeline.py
python inference.py --input sample.csv --model random_forest --output preds.csv
```

**Inference requirements:** CSV must contain exactly the columns listed in `REQUIRED_FEATURES` (see `inference.py`). Nulls are rejected; costs are clipped at zero.

## 7. Responsible Use Statement

This work concerns **healthcare operations** only.  
The model is **not** a diagnostic or treatment system.  
No clinical validity is claimed.  
Patient names, addresses and other direct identifiers are never used as features and never appear in reports or saved artefacts.  
The underlying data are synthetic; results must not be generalised to real clinical populations without further validation on real data under appropriate governance.
