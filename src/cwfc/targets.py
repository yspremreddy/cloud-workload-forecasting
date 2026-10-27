"""Construct the two required aggregate target series from VM-level readings."""
from __future__ import annotations

import pandas as pd


def build_target_series(cpu: pd.DataFrame) -> pd.DataFrame:
    """Build active_vm_count(t) and aggregate_cpu_utilization(t) from VM-level readings.

    Missing-reading handling: a row with a null `avg_cpu` is not a valid observation,
    so it does not count toward `active_vm_count` at that timestamp and is excluded
    (via pandas' default NaN-skipping `mean`) from `aggregate_cpu_utilization`. A VM
    is counted at most once per timestamp regardless of duplicate rows, by taking
    `nunique` over `vm_id` restricted to rows with a non-null `avg_cpu`.

    Timestamps are not resampled or invented: the output has exactly one row per
    distinct timestamp value present in the input, sorted chronologically. Any gap
    or duplicate in the source cadence is preserved as-is for the caller to inspect
    (see `cwfc.quality.audit_cpu_readings`), not silently filled in here.
    """
    valid = cpu.dropna(subset=["avg_cpu"])

    active_vm_count = valid.groupby("timestamp")["vm_id"].nunique()
    aggregate_cpu_utilization = valid.groupby("timestamp")["avg_cpu"].mean()

    out = pd.DataFrame(
        {
            "active_vm_count": active_vm_count,
            "aggregate_cpu_utilization": aggregate_cpu_utilization,
        }
    )
    out = out.sort_index().reset_index().rename(columns={"index": "timestamp"})
    out["active_vm_count"] = out["active_vm_count"].astype(int)
    return out
