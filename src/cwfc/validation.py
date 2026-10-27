from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True)
class IndexSplit:
    """Expanding-window split expressed as row positions (not wall-clock time).

    Used for short, plain-integer-cadence series (e.g. "seconds since trace
    start") where coercing the timestamp column to a datetime would be
    meaningless. Positions are into a frame that is already sorted
    chronologically and contiguous (one row per timestamp, no gaps).
    """

    train_end_idx: int  # exclusive: train is frame.iloc[:train_end_idx]
    val_start_idx: int  # inclusive
    val_end_idx: int  # exclusive: val is frame.iloc[val_start_idx:val_end_idx]


def expanding_window_index_splits(
    n_rows: int,
    horizon: int,
    min_train_size: int,
    step: int | None = None,
) -> list[IndexSplit]:
    """Expanding-window rolling-origin splits over row positions 0..n_rows-1.

    Each split trains on rows [0, train_end_idx) and validates on the next
    `horizon` rows. The origin advances by `step` (default: `horizon`, i.e.
    non-overlapping validation windows) until there is no longer room for a
    full validation window. This mirrors production: at each origin, only
    data up to that point is used, and the forecast horizon matches the
    horizon we actually have to predict (24 steps for this challenge).
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    step = horizon if step is None else step
    out: list[IndexSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n_rows:
        out.append(IndexSplit(train_end_idx, train_end_idx, train_end_idx + horizon))
        train_end_idx += step
    return out


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 288,
    min_train_size: int = 7 * 24 * 12,
    step: int | None = None,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    `horizon=288` is one day at 5-minute cadence.
    """
    if horizon <= 0 or min_train_size <= 0:
        raise ValueError("horizon and min_train_size must be positive")
    ts = pd.to_datetime(frame[timestamp_col], utc=True)
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    step = horizon if step is None else step
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out
