"""
Data preparation and feature engineering for encounter cost prediction.
Target: TOTAL_CLAIM_COST (regression)
Prediction point: start of the encounter (features known or knowable at scheduling / arrival).
"""

from pathlib import Path
import pandas as pd
import numpy as np

# --- Portable paths (work on any machine) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CHANGE THIS to where your original CSVs live
DATA_DIR = PROJECT_ROOT / 'original_data'

OUT_DIR = PROJECT_ROOT / 'data'                      # prepared data goes here


CATEGORICAL_FEATURES = [
    'ENCOUNTERCLASS', 'PAYER_NAME', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL'
]
NUMERIC_FEATURES = [
    'AGE_AT_START', 'YEAR', 'MONTH', 'DOW', 'IS_WEEKEND'
]
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = 'TOTAL_CLAIM_COST'
LOG_TARGET = 'LOG_TOTAL_COST'


def load_and_prepare(save: bool = True) -> pd.DataFrame:
    """Load raw CSVs, engineer features available at encounter start, return modeling frame."""
    enc = pd.read_csv(DATA_DIR / 'encounters.csv')
    pat = pd.read_csv(DATA_DIR / 'patients.csv')
    payers = pd.read_csv(DATA_DIR / 'payers.csv')

    # Dates: strip timezone for arithmetic consistency
    enc['START'] = pd.to_datetime(enc['START']).dt.tz_localize(None)
    enc['STOP'] = pd.to_datetime(enc['STOP']).dt.tz_localize(None)
    pat['BIRTHDATE'] = pd.to_datetime(pat['BIRTHDATE'])

    # Patient demographics only (exclude names, addresses, identifiers beyond Id for join)
    pat_feats = pat[['Id', 'BIRTHDATE', 'GENDER', 'RACE', 'ETHNICITY', 'MARITAL']].rename(
        columns={'Id': 'PATIENT'}
    )
    df = enc.merge(pat_feats, on='PATIENT', how='left')
    df = df.merge(
        payers[['Id', 'NAME']].rename(columns={'Id': 'PAYER', 'NAME': 'PAYER_NAME'}),
        on='PAYER', how='left'
    )

    # Target
    df[TARGET] = df[TARGET].astype(float)
    df[LOG_TARGET] = np.log1p(df[TARGET])

    # Features known / estimable at START
    df['AGE_AT_START'] = (df['START'] - df['BIRTHDATE']).dt.days / 365.25
    df['YEAR'] = df['START'].dt.year
    df['MONTH'] = df['START'].dt.month
    df['DOW'] = df['START'].dt.day_of_week
    df['IS_WEEKEND'] = (df['DOW'] >= 5).astype(int)

    # Derived for analysis / error analysis only (NOT used as features)
    df['DURATION_HOURS'] = (df['STOP'] - df['START']).dt.total_seconds() / 3600

    # Minimal missing handling
    df['MARITAL'] = df['MARITAL'].fillna('U')  # 12 records

    meta = ['Id', 'PATIENT', 'START', 'STOP', 'DURATION_HOURS',
            'BASE_ENCOUNTER_COST', 'PAYER_COVERAGE']
    model_df = df[FEATURE_COLS + [TARGET, LOG_TARGET] + meta].copy()

    # Sanity
    assert model_df['AGE_AT_START'].notna().all()
    assert (model_df['AGE_AT_START'] >= 0).all()

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        model_df.to_csv(OUT_DIR / 'prepared_encounters.csv', index=False)

    return model_df


def chronological_split(df: pd.DataFrame, train_end_year: int = 2019):
    """
    Time-based split: train on encounters with START year <= train_end_year,
    test on later years. Appropriate for operational forecasting.
    """
    train = df[df['YEAR'] <= train_end_year].copy()
    test = df[df['YEAR'] > train_end_year].copy()
    return train, test


if __name__ == '__main__':
    df = load_and_prepare()
    print('Prepared shape:', df.shape)
    print(df[FEATURE_COLS + [TARGET]].head())
    train, test = chronological_split(df)
    print(f'Train: {len(train)} ({train.YEAR.min()}-{train.YEAR.max()})')
    print(f'Test : {len(test)} ({test.YEAR.min()}-{test.YEAR.max()})')
