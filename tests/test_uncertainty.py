import pandas as pd
import pytest

from cwfc.baselines import persistence_forecast
from cwfc.evaluate import rolling_origin_evaluate
from cwfc.uncertainty import lead_time_margins, margin_for_step, residuals_frame


def _biased_forecaster(train, template, target):
    # always forecasts 1 below the last training value -> residual always +1
    last = float(train[target].iloc[-1])
    return pd.Series(last - 1.0, index=template.index, dtype=float)


def test_residuals_frame_sign_convention():
    df = pd.DataFrame({"timestamp": range(60), "value": [10.0] * 60})
    res = rolling_origin_evaluate(
        df, target="value", forecaster=_biased_forecaster,
        horizon=10, min_train_size=30, step=10, name="biased",
    )
    resid = residuals_frame(res)
    assert (resid["residual"] == 1.0).all()


def test_lead_time_margins_positive_for_underforecasting_model():
    df = pd.DataFrame({"timestamp": range(60), "value": [10.0] * 60})
    res = rolling_origin_evaluate(
        df, target="value", forecaster=_biased_forecaster,
        horizon=10, min_train_size=30, step=10, name="biased",
    )
    margins = lead_time_margins(res, quantile=0.9, buckets=((1, 10),))
    assert margins["1-10"] == pytest.approx(1.0)


def test_lead_time_margins_floored_at_zero_for_overforecasting_model():
    def overforecaster(train, template, target):
        last = float(train[target].iloc[-1])
        return pd.Series(last + 5.0, index=template.index, dtype=float)

    df = pd.DataFrame({"timestamp": range(60), "value": [10.0] * 60})
    res = rolling_origin_evaluate(
        df, target="value", forecaster=overforecaster,
        horizon=10, min_train_size=30, step=10, name="over",
    )
    margins = lead_time_margins(res, quantile=0.9, buckets=((1, 10),))
    assert margins["1-10"] == 0.0


def test_margin_for_step_falls_back_when_bucket_missing():
    margins = {"1-8": 0.5, "9-16": 1.0}
    assert margin_for_step(3, margins, buckets=((1, 8), (9, 16))) == 0.5
    assert margin_for_step(12, margins, buckets=((1, 8), (9, 16))) == 1.0
    # step outside all buckets falls back to the largest known margin
    assert margin_for_step(20, margins, buckets=((1, 8), (9, 16))) == 1.0
