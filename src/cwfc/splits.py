"""Chronological train/holdout split for the target series.

The holdout size is derived from the data (last N timestamps), never hard-coded
to assume exactly 134 rows, so this also works against a smaller smoke dataset.
"""
from __future__ import annotations

import pandas as pd


def train_holdout_split(
    target_df: pd.DataFrame, n_holdout: int = 24, timestamp_col: str = "timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not target_df[timestamp_col].is_monotonic_increasing:
        raise ValueError("target_df must be sorted chronologically by timestamp")

    n = len(target_df)
    if n <= n_holdout:
        raise ValueError(
            f"target series has only {n} timestamps, not enough to hold out {n_holdout}"
        )

    train = target_df.iloc[: n - n_holdout].reset_index(drop=True)
    holdout = target_df.iloc[n - n_holdout :].reset_index(drop=True)
    return train, holdout
