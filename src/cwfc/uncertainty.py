"""Empirical, lead-time-aware uncertainty margins from rolling-origin residuals.

No distributional assumption is made; margins are empirical quantiles of the
residuals already produced by `cwfc.evaluate.rolling_origin_evaluate`, so they
come from the same leakage-safe validation as everything else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cwfc.evaluate import EvaluationResult

DEFAULT_LEAD_BUCKETS: tuple[tuple[int, int], ...] = ((1, 8), (9, 16), (17, 24))


def residuals_frame(result: EvaluationResult) -> pd.DataFrame:
    rows = []
    for f in result.folds:
        for step, (yt, yp) in enumerate(zip(f.y_true, f.y_pred), start=1):
            rows.append(
                {"fold": f.fold, "step_ahead": step, "y_true": yt, "y_pred": yp, "residual": yt - yp}
            )
    return pd.DataFrame(rows)


def _bucket_for_step(step: int, buckets: tuple[tuple[int, int], ...]) -> str:
    for lo, hi in buckets:
        if lo <= step <= hi:
            return f"{lo}-{hi}"
    return "unbucketed"


def lead_time_margins(
    result: EvaluationResult,
    quantile: float = 0.9,
    buckets: tuple[tuple[int, int], ...] = DEFAULT_LEAD_BUCKETS,
) -> dict[str, float]:
    """One-sided upper margin per lead-time bucket: the `quantile`-th percentile
    of the signed residual (y_true - y_pred) within that bucket. Adding this
    margin to the point forecast would have covered the true value in roughly
    `quantile` of the observed rolling-origin folds at that lead time. Steps
    are grouped into coarse lead-time buckets (rather than estimated one step
    at a time) because each exact step only has as many residual samples as
    there are validation folds (5 here) -- far too few for a stable quantile;
    pooling across a lead-time range trades some resolution for a usable
    sample size. Negative quantiles (the model already over-forecasts at that
    confidence level) are floored at 0 since we never want a *negative*
    safety margin.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    df = residuals_frame(result)
    if df.empty:
        raise ValueError("no residuals available to compute margins from")
    df["bucket"] = df["step_ahead"].apply(lambda s: _bucket_for_step(s, buckets))
    margins: dict[str, float] = {}
    for bucket, g in df.groupby("bucket"):
        q = float(np.quantile(g["residual"], quantile))
        margins[bucket] = max(q, 0.0)
    return margins


def margin_for_step(
    step: int,
    margins: dict[str, float],
    buckets: tuple[tuple[int, int], ...] = DEFAULT_LEAD_BUCKETS,
) -> float:
    bucket = _bucket_for_step(step, buckets)
    if bucket in margins:
        return margins[bucket]
    return max(margins.values()) if margins else 0.0
