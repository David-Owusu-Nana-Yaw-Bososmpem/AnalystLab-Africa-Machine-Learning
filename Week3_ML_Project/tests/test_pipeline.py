"""
Basic unit / integration tests for the cost prediction pipeline.
Run from project root: python -m pytest tests/ or python tests/test_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import pandas as pd
import numpy as np
import joblib
from data_prep import load_and_prepare, chronological_split, FEATURE_COLS, TARGET
from inference import validate_input, predict, load_model, REQUIRED_FEATURES


def test_data_prep_shape_and_columns():
    df = load_and_prepare(save=False)
    assert len(df) == 27891
    for c in FEATURE_COLS + [TARGET]:
        assert c in df.columns
    assert df['AGE_AT_START'].min() >= 0
    assert df[TARGET].min() >= 0


def test_chronological_split_no_leakage():
    df = load_and_prepare(save=False)
    train, test = chronological_split(df, train_end_year=2019)
    assert train['YEAR'].max() <= 2019
    assert test['YEAR'].min() > 2019
    assert len(train) + len(test) == len(df)


def test_model_files_exist():
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"

    for name in ["ridge", "random_forest", "gbr"]:
        p = models_dir / f"{name}_pipeline.joblib"
        assert p.exists(), f"Missing {p}"


def test_inference_positive_and_shape():
    df = load_and_prepare(save=False)
    sample = df[FEATURE_COLS].head(10).copy()
    # Ensure no nulls
    sample['MARITAL'] = sample['MARITAL'].fillna('U')
    preds = predict(sample, model_name='random_forest')
    assert len(preds) == 10
    assert (preds >= 0).all()


def test_validate_input_raises():
    bad = pd.DataFrame({'ENCOUNTERCLASS': ['ambulatory']})  # missing cols
    try:
        validate_input(bad)
        assert False, 'Should have raised'
    except ValueError:
        pass


def test_pipeline_predict_consistency():
    """Same input → same output (determinism)."""
    df = load_and_prepare(save=False)
    sample = df[FEATURE_COLS].iloc[[0]].copy()
    p1 = predict(sample, 'ridge')
    p2 = predict(sample, 'ridge')
    assert np.allclose(p1, p2)


if __name__ == '__main__':
    test_data_prep_shape_and_columns()
    print('test_data_prep OK')
    test_chronological_split_no_leakage()
    print('test_split OK')
    test_model_files_exist()
    print('test_models_exist OK')
    test_inference_positive_and_shape()
    print('test_inference OK')
    test_validate_input_raises()
    print('test_validate OK')
    test_pipeline_predict_consistency()
    print('test_consistency OK')
    print('All tests passed.')
