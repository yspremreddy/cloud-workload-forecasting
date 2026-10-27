import pandas as pd

from cwfc.policy import apply_headroom_policy, provisioning_summary


def test_apply_headroom_policy_adds_margin_and_clips():
    forecast = pd.Series([95.0, 98.0, 50.0])
    margins = {"1-3": 10.0}
    provisioned = apply_headroom_policy(forecast, margins, buckets=((1, 3),), lower=0.0, upper=100.0)
    assert list(provisioned) == [100.0, 100.0, 60.0]  # clipped at 100 for the first two


def test_apply_headroom_policy_clips_lower_bound():
    forecast = pd.Series([2.0])
    margins = {"1-1": -0.0}  # no margin
    provisioned = apply_headroom_policy(forecast, margins, buckets=((1, 1),), lower=5.0, upper=100.0)
    assert provisioned.iloc[0] == 5.0


def test_provisioning_summary():
    forecast = pd.Series([10.0, 20.0])
    provisioned = pd.Series([12.0, 23.0])
    summary = provisioning_summary(forecast, provisioned)
    assert summary["mean_margin"] == 2.5
    assert summary["max_margin"] == 3.0
