import numpy as np
import pandas as pd
import pytest

from cwfc.models import damped_trend_forecast, ridge_forecast, gbm_forecast


def test_damped_trend_flat_series_forecasts_flat():
    train = pd.DataFrame({"value": [5.0] * 20})
    template = pd.DataFrame(index=range(4))
    out = damped_trend_forecast(train, template, "value")
    assert out.to_numpy() == pytest.approx([5.0] * 4, abs=1e-6)


def test_damped_trend_requires_min_points():
    train = pd.DataFrame({"value": [1.0, 2.0]})
    template = pd.DataFrame(index=range(2))
    with pytest.raises(ValueError):
        damped_trend_forecast(train, template, "value")


def test_damped_trend_damps_toward_flat_not_unbounded_trend():
    # Linear uptrend: forecast should rise but by less than a non-damped
    # linear extrapolation would, because phi < 1.
    values = [float(i) for i in range(30)]
    train = pd.DataFrame({"value": values})
    template = pd.DataFrame(index=range(10))
    out = damped_trend_forecast(train, template, "value", alpha=0.9, beta=0.9, phi=0.85)
    naive_linear_last_step = values[-1] + 10  # undamped drift would add ~10
    assert out.iloc[-1] < naive_linear_last_step


def test_ridge_forecast_runs_and_returns_right_length():
    rng = np.random.default_rng(0)
    values = 10 + 0.01 * np.arange(60) + rng.normal(0, 0.1, 60)
    train = pd.DataFrame({"value": values})
    template = pd.DataFrame(index=range(5))
    out = ridge_forecast(train, template, "value")
    assert len(out) == 5
    assert out.notna().all()


def test_ridge_forecast_raises_when_not_enough_history():
    train = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    template = pd.DataFrame(index=range(2))
    with pytest.raises(ValueError):
        ridge_forecast(train, template, "value")


def test_gbm_forecast_runs_and_returns_right_length():
    rng = np.random.default_rng(0)
    values = 10 + rng.normal(0, 0.1, 60)
    train = pd.DataFrame({"value": values})
    template = pd.DataFrame(index=range(5))
    out = gbm_forecast(train, template, "value")
    assert len(out) == 5
    assert out.notna().all()
