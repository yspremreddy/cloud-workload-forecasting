"""Operational headroom policy: point forecast + lead-time-aware margin.

Captures the under- vs over-provisioning trade-off the assignment asks for
(section G): the margin's quantile directly trades off the probability of
under-provisioning (true demand exceeds the provisioned target) against how
much capacity is held idle on average (over-provisioning).
"""
from __future__ import annotations

import pandas as pd

from cwfc.uncertainty import DEFAULT_LEAD_BUCKETS, margin_for_step


def apply_headroom_policy(
    point_forecast: pd.Series,
    margins: dict[str, float],
    buckets: tuple[tuple[int, int], ...] = DEFAULT_LEAD_BUCKETS,
    lower: float = 0.0,
    upper: float = 100.0,
) -> pd.Series:
    """Add a lead-time-dependent quantile margin to a point forecast to get a
    provisioning target, clipped to a valid range (e.g. [0, 100] for a CPU %).
    `point_forecast` is assumed to be indexed 0..n-1 in forecast order, i.e.
    position i is step-ahead i+1 from the origin the forecast was made at.
    """
    provisioned = []
    for i in range(len(point_forecast)):
        step = i + 1
        margin = margin_for_step(step, margins, buckets)
        value = float(point_forecast.iloc[i]) + margin
        provisioned.append(min(upper, max(lower, value)))
    return pd.Series(provisioned, index=point_forecast.index, dtype=float)


def provisioning_summary(point_forecast: pd.Series, provisioned: pd.Series) -> dict[str, float]:
    """Headline numbers for the report: average and max headroom added."""
    added = provisioned.to_numpy() - point_forecast.to_numpy()
    return {
        "mean_margin": float(added.mean()),
        "max_margin": float(added.max()),
        "mean_point_forecast": float(point_forecast.mean()),
        "mean_provisioned": float(provisioned.mean()),
    }
