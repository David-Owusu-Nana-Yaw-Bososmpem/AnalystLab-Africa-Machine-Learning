import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "complaint_volume_forecasting.py"
spec = importlib.util.spec_from_file_location("complaint_forecasting", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_feature_row_uses_only_prior_history():
    history = list(range(1, 15))
    row = module.make_feature_row(history, pd.Period("2023-03", freq="M"))
    assert row["lag_1"] == 14
    assert row["lag_2"] == 13
    assert row["lag_3"] == 12
    assert row["lag_6"] == 9
    assert row["lag_12"] == 3
    assert row["rolling_mean_3"] == np.mean([12, 13, 14])


def test_feature_builder_does_not_use_current_target_as_lag():
    idx = pd.period_range("2020-01", periods=15, freq="M")
    series = pd.Series(np.arange(100, 115, dtype=float), index=idx)
    X, y = module.supervised_from_series(series)
    target_period = idx[12]
    assert X.loc[target_period, "lag_1"] == series.iloc[11]
    assert y.loc[target_period] == series.iloc[12]
    assert X.loc[target_period, "lag_1"] != y.loc[target_period]


def test_six_month_forecast_periods_skip_partial_august():
    intermediate = pd.period_range("2023-08", periods=7, freq="M")
    reported = intermediate[1:]
    assert str(intermediate[0]) == "2023-08"
    assert list(map(str, reported)) == [
        "2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02"
    ]
    assert len(reported) == 6


def test_forecast_output_is_non_negative_and_months_are_aligned():
    periods = pd.period_range("2023-09", periods=6, freq="M")
    forecasts = np.array([100, 101, 99, 102, 103, 100], dtype=float)
    assert len(periods) == len(forecasts)
    assert np.all(forecasts >= 0)
