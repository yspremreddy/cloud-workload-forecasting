"""Forecasting models, in increasing order of complexity.

Every model shares the same `forecaster(train, template, target) -> pd.Series`
interface as `cwfc.baselines`, so each one plugs directly into
`cwfc.evaluate.rolling_origin_evaluate` without special-casing.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from cwfc.features import (
    DEFAULT_LAGS,
    DEFAULT_ROLL_WINDOWS,
    build_features,
    feature_columns,
    make_direct_training_rows,
)

MIN_ROWS_PER_STEP_MODEL = 5


def _damped_trend_fit(
    values: np.ndarray,
    alpha: float | None,
    beta: float | None,
    phi: float | None,
) -> tuple[float, float, float, float, float]:
    """Fit Holt's damped-trend exponential smoothing by grid-searching the
    smoothing parameters to minimize in-sample one-step-ahead SSE (the
    standard way to fit ETS when not using a full likelihood optimizer).
    Returns (alpha, beta, phi, level, trend) after the fit.
    """

    def run(a: float, b: float, p: float) -> tuple[float, float, float]:
        level = values[0]
        trend = values[1] - values[0]
        sse = 0.0
        for t in range(1, len(values)):
            pred = level + p * trend
            sse += (values[t] - pred) ** 2
            new_level = a * values[t] + (1 - a) * (level + p * trend)
            new_trend = b * (new_level - level) + (1 - b) * p * trend
            level, trend = new_level, new_trend
        return level, trend, sse

    alpha_grid = [alpha] if alpha is not None else list(np.linspace(0.05, 0.95, 7))
    beta_grid = [beta] if beta is not None else list(np.linspace(0.05, 0.95, 7))
    phi_grid = [phi] if phi is not None else list(np.linspace(0.80, 0.98, 4))

    best = None
    for a in alpha_grid:
        for b in beta_grid:
            for p in phi_grid:
                level, trend, sse = run(a, b, p)
                if best is None or sse < best[0]:
                    best = (sse, a, b, p, level, trend)
    _, best_a, best_b, best_p, best_level, best_trend = best
    return best_a, best_b, best_p, best_level, best_trend


def damped_trend_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    alpha: float | None = None,
    beta: float | None = None,
    phi: float | None = None,
) -> pd.Series:
    """Holt's damped-trend exponential smoothing (a well-behaved member of the
    ETS family). Damping (phi < 1) flattens the trend out at longer horizons
    instead of extrapolating it forever, which suits a short, noisy series
    where we have no evidence a local trend will persist for 2 hours."""
    values = train[target].astype(float).to_numpy()
    if len(values) < 3:
        raise ValueError("damped trend forecast requires at least 3 points")
    horizon = len(template)

    a, b, p, level, trend = _damped_trend_fit(values, alpha, beta, phi)

    forecast = []
    damp_sum = 0.0
    for h in range(1, horizon + 1):
        damp_sum += p**h
        forecast.append(level + damp_sum * trend)
    return pd.Series(forecast, index=template.index, dtype=float)


def _direct_multistep_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    estimator_factory: Callable[[], object],
    lags: tuple[int, ...] = DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
) -> pd.Series:
    series = train[target].astype(float)
    horizon = len(template)
    cols = feature_columns(lags, roll_windows)
    rows_by_h = make_direct_training_rows(series, horizon, lags, roll_windows)

    origin_feats = build_features(series, lags, roll_windows).iloc[[-1]][cols]
    if origin_feats.isna().any(axis=None):
        raise ValueError(
            "not enough training history to build features at the forecast origin "
            f"(need at least {max(max(lags), max(roll_windows) + 1)} points)"
        )
    x_pred = origin_feats.to_numpy()

    last_value = float(series.iloc[-1])
    preds: list[float] = []
    for h in range(1, horizon + 1):
        df = rows_by_h[h]
        if len(df) < MIN_ROWS_PER_STEP_MODEL:
            # Too few labeled examples to fit this step's model reliably at
            # this origin (happens for large h on the earliest validation
            # folds, which have little history); fall back to persistence
            # for that step rather than fitting an unstable model.
            preds.append(last_value)
            continue
        model = estimator_factory()
        model.fit(df[cols].to_numpy(), df["y"].to_numpy())
        preds.append(float(model.predict(x_pred)[0]))
    return pd.Series(preds, index=template.index, dtype=float)


def ridge_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    alpha: float = 1.0,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
) -> pd.Series:
    """Ridge regression on lag/rolling-stat features, one model per step-ahead
    (direct multi-step strategy)."""
    return _direct_multistep_forecast(
        train, template, target, lambda: Ridge(alpha=alpha), lags, roll_windows
    )


def gbm_forecast(
    train: pd.DataFrame,
    template: pd.DataFrame,
    target: str,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
    **gbm_kwargs,
) -> pd.Series:
    """Gradient-boosted trees on the same features, one model per step-ahead.
    Depth and estimator count are kept small given the tiny per-fold training
    set (well under a hundred rows)."""
    params = dict(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=0)
    params.update(gbm_kwargs)
    return _direct_multistep_forecast(
        train, template, target, lambda: GradientBoostingRegressor(**params), lags, roll_windows
    )
