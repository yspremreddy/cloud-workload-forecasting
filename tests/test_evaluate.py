import pandas as pd
import pytest

from cwfc.baselines import persistence_forecast, moving_average_forecast, drift_forecast
from cwfc.evaluate import rolling_origin_evaluate
from cwfc.validation import expanding_window_index_splits


def test_expanding_window_index_splits_non_overlapping():
    splits = expanding_window_index_splits(n_rows=110, horizon=24, min_train_size=40, step=10)
    assert len(splits) == 5
    assert splits[0].train_end_idx == 40
    assert splits[0].val_start_idx == 40
    assert splits[0].val_end_idx == 64
    assert splits[-1].train_end_idx == 80
    assert splits[-1].val_end_idx == 104
    # every val window stays within bounds
    for s in splits:
        assert s.val_end_idx <= 110


def test_expanding_window_index_splits_empty_when_too_short():
    assert expanding_window_index_splits(n_rows=10, horizon=24, min_train_size=5) == []


def test_rolling_origin_evaluate_persistence_on_flat_series():
    df = pd.DataFrame({"timestamp": range(70), "value": [5.0] * 70})
    result = rolling_origin_evaluate(
        df, target="value", forecaster=persistence_forecast,
        horizon=10, min_train_size=30, step=10, name="persistence",
    )
    assert result.mean_mae == 0.0
    assert result.mean_rmse == 0.0
    assert len(result.folds) == 4


def test_rolling_origin_forecaster_template_excludes_validation_targets():
    df = pd.DataFrame({"timestamp": range(40), "value": [float(i) for i in range(40)]})
    seen_templates = []

    def inspect_template(train, template, target):
        assert target not in template.columns
        seen_templates.append(template.copy())
        return persistence_forecast(train, template, target)

    result = rolling_origin_evaluate(
        df, target="value", forecaster=inspect_template,
        horizon=5, min_train_size=20, step=5,
    )

    assert len(result.folds) == len(seen_templates) == 4
    assert all(template.columns.tolist() == ["timestamp"] for template in seen_templates)


def test_rolling_origin_evaluate_moving_average_and_drift_run():
    df = pd.DataFrame({"timestamp": range(70), "value": [float(i % 5) for i in range(70)]})

    def ma3(train, template, target):
        return moving_average_forecast(train, template, target, window=3)

    res_ma = rolling_origin_evaluate(
        df, target="value", forecaster=ma3, horizon=10, min_train_size=30, step=10, name="ma3",
    )
    res_drift = rolling_origin_evaluate(
        df, target="value", forecaster=drift_forecast, horizon=10, min_train_size=30, step=10, name="drift",
    )
    assert res_ma.mean_mae >= 0.0
    assert res_drift.mean_mae >= 0.0


def test_rolling_origin_evaluate_raises_when_no_folds_fit():
    df = pd.DataFrame({"timestamp": range(10), "value": range(10)})
    with pytest.raises(ValueError):
        rolling_origin_evaluate(df, target="value", forecaster=persistence_forecast, horizon=24, min_train_size=40)


def test_per_step_errors_shape():
    df = pd.DataFrame({"timestamp": range(70), "value": [5.0] * 70})
    result = rolling_origin_evaluate(
        df, target="value", forecaster=persistence_forecast,
        horizon=10, min_train_size=30, step=10, name="persistence",
    )
    per_step = result.per_step_errors()
    assert len(per_step) == 10
    assert set(per_step.columns) == {"step_ahead", "mae", "rmse"}
