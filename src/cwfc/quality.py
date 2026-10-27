"""Data quality audit utilities for the raw VM CPU readings and metadata."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    n_rows: int
    n_vms: int
    n_timestamps: int
    timestamp_min: int
    timestamp_max: int
    cadence_seconds: int
    irregular_cadence: bool
    n_duplicate_vm_timestamp: int
    readings_per_vm_min: int
    readings_per_vm_max: int
    vms_with_incomplete_trace: int
    n_missing_avg_cpu: int
    n_missing_min_cpu: int
    n_missing_max_cpu: int
    n_rows_avg_outside_min_max: int
    avg_cpu_min: float
    avg_cpu_max: float
    metadata_vm_mismatch: int
    metadata_duplicate_vm: int
    notes: list[str] = field(default_factory=list)


def audit_cpu_readings(cpu: pd.DataFrame, meta: pd.DataFrame | None = None) -> QualityReport:
    """Run structural checks on the raw CPU readings (and optionally metadata).

    Produces the facts needed to justify downstream target-construction decisions,
    without silently fixing anything.
    """
    notes: list[str] = []

    n_rows = len(cpu)
    n_vms = cpu["vm_id"].nunique()
    unique_ts = np.sort(cpu["timestamp"].unique())
    n_timestamps = len(unique_ts)
    ts_min, ts_max = int(unique_ts[0]), int(unique_ts[-1])

    diffs = np.diff(unique_ts)
    cadence_values = np.unique(diffs)
    cadence_seconds = int(cadence_values[0]) if len(cadence_values) else 0
    irregular_cadence = len(cadence_values) > 1
    if irregular_cadence:
        notes.append(f"Irregular cadence detected: step sizes {cadence_values.tolist()}")

    n_dup = int(cpu.duplicated(subset=["vm_id", "timestamp"]).sum())
    if n_dup:
        notes.append(f"{n_dup} duplicate (vm_id, timestamp) rows found")

    per_vm_counts = cpu.groupby("vm_id").size()
    readings_min = int(per_vm_counts.min())
    readings_max = int(per_vm_counts.max())
    vms_incomplete = int((per_vm_counts != n_timestamps).sum())
    if vms_incomplete:
        notes.append(f"{vms_incomplete} VMs do not have a reading at every timestamp")

    n_missing_avg = int(cpu["avg_cpu"].isna().sum())
    n_missing_min = int(cpu["min_cpu"].isna().sum())
    n_missing_max = int(cpu["max_cpu"].isna().sum())

    outside_mask = (cpu["avg_cpu"] > cpu["max_cpu"]) | (cpu["avg_cpu"] < cpu["min_cpu"])
    n_outside = int(outside_mask.sum())
    if n_outside:
        notes.append(f"{n_outside} rows have avg_cpu outside [min_cpu, max_cpu]")

    avg_min = float(cpu["avg_cpu"].min())
    avg_max = float(cpu["avg_cpu"].max())

    meta_mismatch = 0
    meta_dup = 0
    if meta is not None:
        meta_ids = set(meta["vm_id"])
        cpu_ids = set(cpu["vm_id"])
        meta_mismatch = len(meta_ids.symmetric_difference(cpu_ids))
        meta_dup = int(meta["vm_id"].duplicated().sum())
        if meta_mismatch:
            notes.append(f"{meta_mismatch} VM ids differ between metadata and CPU readings")
        if meta_dup:
            notes.append(f"{meta_dup} duplicate vm_id rows in metadata")

    if n_timestamps == 1:
        pass
    elif per_vm_counts.nunique() == 1 and vms_incomplete == 0:
        notes.append(
            "Every VM has a reading at every timestamp: active_vm_count(t) is "
            "constant by construction (the release only includes VMs with complete traces)."
        )

    return QualityReport(
        n_rows=n_rows,
        n_vms=n_vms,
        n_timestamps=n_timestamps,
        timestamp_min=ts_min,
        timestamp_max=ts_max,
        cadence_seconds=cadence_seconds,
        irregular_cadence=irregular_cadence,
        n_duplicate_vm_timestamp=n_dup,
        readings_per_vm_min=readings_min,
        readings_per_vm_max=readings_max,
        vms_with_incomplete_trace=vms_incomplete,
        n_missing_avg_cpu=n_missing_avg,
        n_missing_min_cpu=n_missing_min,
        n_missing_max_cpu=n_missing_max,
        n_rows_avg_outside_min_max=n_outside,
        avg_cpu_min=avg_min,
        avg_cpu_max=avg_max,
        metadata_vm_mismatch=meta_mismatch,
        metadata_duplicate_vm=meta_dup,
        notes=notes,
    )
