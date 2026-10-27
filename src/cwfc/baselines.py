from __future__ import annotations

import pandas as pd


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    if train.empty:
        raise ValueError("train must not be empty")
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)


def moving_average_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, window: int) -> pd.Series:
    """Flat forecast at the mean of the last `window` training observations."""
    if window <= 0:
        raise ValueError("window must be positive")
    if train.empty:
        raise ValueError("train must not be empty")
    tail = train[target].astype(float).to_numpy()[-window:]
    level = float(tail.mean())
    return pd.Series(level, index=template.index, dtype=float)


def drift_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    """Linear extrapolation from the first to the last training observation
    (the classic "drift method": random walk with drift, fit without noise)."""
    if len(train) < 2:
        raise ValueError("drift forecast requires at least 2 training points")
    values = train[target].astype(float).to_numpy()
    n = len(values)
    slope = (values[-1] - values[0]) / (n - 1)
    forecast = [float(values[-1] + slope * (h + 1)) for h in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)
