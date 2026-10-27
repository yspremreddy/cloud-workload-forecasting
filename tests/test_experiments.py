import numpy as np
import pandas as pd
import pytest

from cwfc.experiments import evaluate_ridge_configurations


def test_ridge_sweep_reports_each_chronological_fold_and_is_deterministic():
    rng = np.random.default_rng(12)
    values = 8 + 0.3 * np.sin(np.arange(55) / 5) + rng.normal(0, 0.05, 55)
    frame = pd.DataFrame({"timestamp": np.arange(55) * 300, "value": values})
    sets = {"full": ((1, 2, 3), (3, 6)), "lags": ((1, 2), ())}
    folds, summary = evaluate_ridge_configurations(
        frame, "value", alphas=(0.1, 1.0), feature_sets=sets,
        horizon=5, min_train_size=20, step=5,
    )
    assert len(folds) == 7 * 2 * 2
    assert set(folds.groupby(["feature_set", "alpha"])["fold"].count()) == {7}
    assert all(v == [1, 2, 3, 4, 5, 6, 7] for v in folds.groupby(["feature_set", "alpha"])["fold"].apply(list))
    assert np.isfinite(folds[["mae", "rmse"]].to_numpy()).all()
    assert summary["selected"].sum() == 1
    assert summary.iloc[0]["mean_mae"] == pytest.approx(folds.groupby(["feature_set", "alpha"])["mae"].mean().min())
    folds_again, summary_again = evaluate_ridge_configurations(
        frame, "value", alphas=(0.1, 1.0), feature_sets=sets,
        horizon=5, min_train_size=20, step=5,
    )
    pd.testing.assert_frame_equal(folds, folds_again)
    pd.testing.assert_frame_equal(summary, summary_again)


def test_ridge_sweep_rejects_invalid_configuration():
    frame = pd.DataFrame({"timestamp": range(20), "value": np.arange(20, dtype=float)})
    with pytest.raises(ValueError, match="positive finite"):
        evaluate_ridge_configurations(frame, "value", alphas=(0,))
