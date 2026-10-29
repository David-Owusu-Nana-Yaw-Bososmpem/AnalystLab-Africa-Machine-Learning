# Model Comparison & Evaluation Report

## Problem
Predict total claim cost of a hospital encounter using only features available at encounter start.

## Data split
- Train: 2011–2019 (21,622 encounters)
- Test: 2020–2022 (6,269 encounters)
- Chronological to simulate real operational use.

## Results (test set)

| Model              | MAE ($) | RMSE ($) | R²    | MedAE ($) |
|--------------------|---------|----------|-------|-----------|
| Mean baseline      | 5,125   | 12,305   | −0.00 | 3,598     |
| Median baseline    | 3,083   | 12,616   | −0.05 | 238       |
| Ridge              | 4,391   | 11,991   | 0.05  | 2,811     |
| **Random Forest**  | **3,066** | 11,326 | 0.15 | 1,123     |
| Gradient Boosting  | 3,124   | **11,279** | **0.16** | 1,316  |

**Selected production candidate:** Random Forest (best MAE, competitive RMSE/R², interpretable feature importances).

## Why the metrics matter
- MAE is the most decision-relevant metric for operations/finance (average dollar miss).
- RMSE highlights the impact of the heavy right tail of costs.
- R² remains modest because procedure-level drivers of cost are intentionally excluded to prevent leakage. This is a feature of the problem definition, not a modelling failure.

## Error patterns
- Highest absolute errors: rare inpatient/emergency encounters with total costs > $100k (procedure-driven).
- Lowest errors: ambulatory encounters.
- Residuals are approximately centred; model does not systematically over- or under-predict on average.

## Leakage & validity checks
- No future information, no STOP-derived features, no procedure table joined into features.
- Input validation rejects missing required columns or nulls.
- All unit/integration tests pass.

## Next steps for monitoring (pre-deployment)
1. Track MAE / RMSE by month and by ENCOUNTERCLASS.
2. Monitor prediction distribution shift (PSI or simple mean/variance checks).
3. Consider a two-stage approach: classifier for “high-cost risk” + regressor, or quantile regression for tail awareness.
4. Annual retrain or retrain on significant payer-mix / practice-pattern change.
5. Keep human-in-the-loop for any operational action based on high predicted cost.
