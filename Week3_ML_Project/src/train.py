"""
Train baseline + two model families for TOTAL_CLAIM_COST prediction.
Saves models, metrics, and feature importance.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

from data_prep import (
    load_and_prepare, chronological_split,
    FEATURE_COLS, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
)

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
REPORTS_DIR = Path(__file__).resolve().parents[1] / 'reports'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor():
    """One-hot categoricals, scale numerics. Handle unknown categories at inference."""
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
            ('num', StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder='drop'
    )


def evaluate(y_true, y_pred, prefix=''):
    """Return dict of regression metrics. Costs are skewed → MAE primary, RMSE secondary."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Median AE more robust to the few extreme costs
    medae = np.median(np.abs(y_true - y_pred))
    return {
        f'{prefix}MAE': float(mae),
        f'{prefix}RMSE': float(rmse),
        f'{prefix}R2': float(r2),
        f'{prefix}MedAE': float(medae),
    }


def main():
    print('Loading and preparing data...')
    df = load_and_prepare(save=True)
    train_df, test_df = chronological_split(df, train_end_year=2019)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET]

    print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')
    print(f'Train cost mean/median: {y_train.mean():.1f} / {y_train.median():.1f}')
    print(f'Test  cost mean/median: {y_test.mean():.1f} / {y_test.median():.1f}')

    # ----- Baseline: Dummy (mean) -----
    baseline = DummyRegressor(strategy='mean')
    baseline.fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    metrics_base = evaluate(y_test, y_pred_base, prefix='baseline_')
    print('\nBaseline (mean predictor):', metrics_base)

    # Also median baseline for reference
    median_base = DummyRegressor(strategy='median')
    median_base.fit(X_train, y_train)
    y_pred_med = median_base.predict(X_test)
    metrics_med = evaluate(y_test, y_pred_med, prefix='median_baseline_')
    print('Median baseline:', metrics_med)

    # ----- Preprocessor + models -----
    preproc = build_preprocessor()

    models = {
        'ridge': Pipeline([
            ('pre', preproc),
            ('model', Ridge(alpha=1.0, random_state=42))
        ]),
        'random_forest': Pipeline([
            ('pre', preproc),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42
            ))
        ]),
        'gbr': Pipeline([
            ('pre', preproc),
            ('model', GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=42
            ))
        ]),
    }

    all_metrics = {
        'baseline_mean': metrics_base,
        'baseline_median': metrics_med,
        'train_years': '2011-2019',
        'test_years': '2020-2022',
        'n_train': len(X_train),
        'n_test': len(X_test),
    }
    predictions = {'y_true': y_test.values}

    for name, pipe in models.items():
        print(f'\nTraining {name}...')
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # Clip negative predictions (costs cannot be negative)
        y_pred = np.maximum(y_pred, 0)
        m = evaluate(y_test, y_pred, prefix=f'{name}_')
        all_metrics.update(m)
        predictions[name] = y_pred
        print(m)

        # Persist
        joblib.dump(pipe, MODELS_DIR / f'{name}_pipeline.joblib')
        print(f'Saved {name}_pipeline.joblib')

    # Feature importance for tree models (approximate via RF)
    rf_pipe = models['random_forest']
    # Get feature names after one-hot
    ohe = rf_pipe.named_steps['pre'].named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feat_names = cat_names + NUMERIC_FEATURES
    importances = rf_pipe.named_steps['model'].feature_importances_
    imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False)
    imp_df.to_csv(REPORTS_DIR / 'rf_feature_importance.csv', index=False)
    print('\nTop 15 features (RF):')
    print(imp_df.head(15).to_string(index=False))

    # Save metrics
    with open(REPORTS_DIR / 'metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Save test predictions for error analysis
    pred_df = test_df[['Id', 'PATIENT', 'START', 'ENCOUNTERCLASS', 'PAYER_NAME', TARGET]].copy()
    pred_df['pred_ridge'] = predictions['ridge']
    pred_df['pred_rf'] = predictions['random_forest']
    pred_df['pred_gbr'] = predictions['gbr']
    pred_df['pred_baseline'] = y_pred_base
    pred_df['abs_err_rf'] = np.abs(pred_df[TARGET] - pred_df['pred_rf'])
    pred_df.to_csv(REPORTS_DIR / 'test_predictions.csv', index=False)

    print('\n=== Summary ===')
    print(json.dumps({k: v for k, v in all_metrics.items() if 'MAE' in k or 'RMSE' in k or 'R2' in k}, indent=2))
    print('\nDone. Models and reports written.')


if __name__ == '__main__':
    main()
