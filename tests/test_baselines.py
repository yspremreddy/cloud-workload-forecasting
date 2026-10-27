import pandas as pd
import pytest

from cwfc.baselines import moving_average_forecast, drift_forecast


def test_moving_average_forecast():
    train = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
    template = pd.DataFrame(index=range(3))
    out = moving_average_forecast(train, template, "value", window=2)
    assert (out == 3.5).all()
    assert len(out) == 3


def test_moving_average_requires_positive_window():
    train = pd.DataFrame({"value": [1.0]})
    template = pd.DataFrame(index=range(1))
    with pytest.raises(ValueError):
        moving_average_forecast(train, template, "value", window=0)


def test_drift_forecast_extrapolates_linearly():
    train = pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]})  # slope 1 per step
    template = pd.DataFrame(index=range(2))
    out = drift_forecast(train, template, "value")
    assert list(out) == [4.0, 5.0]


def test_drift_forecast_requires_two_points():
    train = pd.DataFrame({"value": [1.0]})
    template = pd.DataFrame(index=range(1))
    with pytest.raises(ValueError):
        drift_forecast(train, template, "value")
