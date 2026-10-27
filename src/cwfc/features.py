"""Leakage-safe feature construction for lag/rolling-stat forecasting models.

Every feature at row i is built from `series[:i]` only (strictly past values),
so it is always computable at a forecast origin positioned at i -- never from
the value being predicted or anything after it.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_LAGS = (1, 2, 3, 6, 12)
DEFAULT_ROLL_WINDOWS = (3, 6, 12)


def build_features(
    series: pd.Series,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
) -> pd.DataFrame:
    s = series.reset_index(drop=True).astype(float)
    feats: dict[str, pd.Series] = {}
    for lag in lags:
        feats[f"lag_{lag}"] = s.shift(lag)
    shifted = s.shift(1)  # exclude the current value from any rolling window
    for w in roll_windows:
        feats[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        feats[f"roll_std_{w}"] = shifted.rolling(w).std()
    return pd.DataFrame(feats)


def feature_columns(
    lags: tuple[int, ...] = DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
) -> list[str]:
    cols = [f"lag_{lag}" for lag in lags]
    for w in roll_windows:
        cols += [f"roll_mean_{w}", f"roll_std_{w}"]
    return cols


def make_direct_training_rows(
    series: pd.Series,
    horizon: int,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    roll_windows: tuple[int, ...] = DEFAULT_ROLL_WINDOWS,
) -> dict[int, pd.DataFrame]:
    """Build one training frame per step-ahead h in 1..horizon (direct multi-step
    strategy): each frame has the feature columns plus `y = series[t + h]`,
    restricted to origins t where both the features and the target are available
    inside `series`. Using a separate model per horizon, instead of recursively
    feeding back single-step predictions, avoids compounding one-step errors
    across the 24-step horizon.
    """
    feats = build_features(series, lags, roll_windows)
    out: dict[int, pd.DataFrame] = {}
    s = series.reset_index(drop=True).astype(float)
    for h in range(1, horizon + 1):
        df = feats.copy()
        df["y"] = s.shift(-h)
        out[h] = df.dropna()
    return out
