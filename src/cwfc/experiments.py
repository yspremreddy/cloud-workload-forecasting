"""Reproducible Ridge configuration sweeps over chronological validation folds."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from cwfc.evaluate import rolling_origin_evaluate
from cwfc.features import DEFAULT_LAGS, DEFAULT_ROLL_WINDOWS
from cwfc.models import ridge_forecast

SELECTED_RIDGE_ALPHA = 1.0
SELECTED_RIDGE_FEATURE_SET = "lags_only"

DEFAULT_FEATURE_SETS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "full": (DEFAULT_LAGS, DEFAULT_ROLL_WINDOWS),
    "compact": ((1, 2, 3), (3, 6)),
    "lags_only": (DEFAULT_LAGS, ()),
}


def evaluate_ridge_configurations(
    train_df: pd.DataFrame,
    target: str,
    alphas: Sequence[float] = (0.1, 1.0, 5.0, 50.0),
    feature_sets: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
    horizon: int = 24,
    min_train_size: int = 40,
    step: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return fold-level and mean metrics for every Ridge alpha/feature set.

    Every configuration runs through the same expanding-window folds. Features
    and each Ridge estimator are rebuilt/fitted from that fold's training rows;
    validation targets are used only for scoring. The caller must pass only the
    public training window, never the final holdout.
    """
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("alphas must be a non-empty sequence of positive finite values")
    sets = DEFAULT_FEATURE_SETS if feature_sets is None else feature_sets
    if not sets:
        raise ValueError("feature_sets must not be empty")

    rows: list[dict[str, float | int | str]] = []
    for feature_set, (lags, roll_windows) in sets.items():
        if not lags and not roll_windows:
            raise ValueError(f"feature set {feature_set!r} has no features")
        for alpha in alphas:
            def forecaster(fold_train, template, column, *, a=alpha, ls=lags, ws=roll_windows):
                return ridge_forecast(fold_train, template, column, alpha=float(a), lags=ls, roll_windows=ws)

            result = rolling_origin_evaluate(
                train_df, target, forecaster, horizon=horizon,
                min_train_size=min_train_size, step=step,
                name=f"ridge_{feature_set}_alpha_{alpha:g}",
            )
            for fold in result.folds:
                rows.append({
                    "feature_set": feature_set,
                    "alpha": float(alpha),
                    "fold": fold.fold + 1,
                    "train_end_idx": fold.train_end_idx,
                    "origin_timestamp": fold.origin_timestamp,
                    "mae": fold.mae,
                    "rmse": fold.rmse,
                })

    fold_results = pd.DataFrame(rows)
    summary = (
        fold_results.groupby(["feature_set", "alpha"], as_index=False)
        .agg(n_folds=("fold", "count"), mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"))
        .sort_values(["mean_mae", "mean_rmse", "feature_set", "alpha"], ignore_index=True)
    )
    summary["selected"] = False
    summary.loc[0, "selected"] = True
    return fold_results, summary
