"""
Inference script for encounter total claim cost prediction.
Loads a saved pipeline and scores new records that contain the required features.
"""

from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
REQUIRED_FEATURES = [
    'ENCOUNTERCLASS', 'PAYER_NAME', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL',
    'AGE_AT_START', 'YEAR', 'MONTH', 'DOW', 'IS_WEEKEND'
]


def load_model(name: str = 'random_forest'):
    path = MODELS_DIR / f'{name}_pipeline.joblib'
    if not path.exists():
        raise FileNotFoundError(f'Model not found: {path}. Train first with train.py')
    return joblib.load(path)


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if df[REQUIRED_FEATURES].isnull().any().any():
        raise ValueError('Input contains nulls in required features. Impute or drop first.')


def predict(df: pd.DataFrame, model_name: str = 'random_forest') -> np.ndarray:
    """Return non-negative predicted total claim costs."""
    validate_input(df)
    pipe = load_model(model_name)
    preds = pipe.predict(df[REQUIRED_FEATURES])
    return np.maximum(preds, 0.0)


def main():
    parser = argparse.ArgumentParser(description='Predict encounter total claim cost')
    parser.add_argument('--input', required=True, help='CSV with required feature columns')
    parser.add_argument('--model', default='random_forest', choices=['ridge', 'random_forest', 'gbr'])
    parser.add_argument('--output', default='predictions.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    preds = predict(df, args.model)
    out = df.copy()
    out['predicted_total_claim_cost'] = preds
    out.to_csv(args.output, index=False)
    print(f'Wrote {len(out)} predictions to {args.output}')
    print(out[['predicted_total_claim_cost']].describe())


if __name__ == '__main__':
    main()
