import numpy as np
import pandas as pd
import pytest

from cwfc.predict import generate_predictions


def _write_smoke_dataset(tmp_path, n_vms=12, n_timestamps=40):
    rows = []
    rng = np.random.default_rng(0)
    for t in range(n_timestamps):
        for v in range(n_vms):
            rows.append(
                {
                    "timestamp": t * 300,
                    "vm_id": f"vm_{v}",
                    "min_cpu": 0.0,
                    "max_cpu": 50.0,
                    "avg_cpu": float(rng.uniform(1, 40)),
                }
            )
    cpu_df = pd.DataFrame(rows)
    cpu_df.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")

    meta_rows = [
        {
            "vm_id": f"vm_{v}",
            "subscription_id": "sub_0",
            "deployment_id": "dep_0",
            "timestamp_vm_created": 0,
            "timestamp_vm_deleted": 999999,
            "max_cpu": 50.0,
            "avg_cpu": 20.0,
            "p95_max_cpu": 45.0,
            "vm_category": "Interactive",
            "virtual_core_bucket": "2",
            "memory_gb_bucket": "4",
        }
        for v in range(n_vms)
    ]
    pd.DataFrame(meta_rows).to_csv(tmp_path / "vm_metadata.csv.gz", index=False, compression="gzip")


def test_generate_predictions_schema_on_smoke_dataset(tmp_path):
    _write_smoke_dataset(tmp_path, n_vms=12, n_timestamps=40)
    preds = generate_predictions(tmp_path, n_holdout=8)

    assert list(preds.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(preds) == 8
    assert preds["timestamp"].is_monotonic_increasing
    assert preds["timestamp"].is_unique
    assert preds.notna().all().all()
    assert np.isfinite(preds["active_vm_count"].to_numpy()).all()
    assert np.isfinite(preds["aggregate_cpu_utilization"].to_numpy()).all()
    assert (preds["active_vm_count"] >= 0).all()
    assert (preds["aggregate_cpu_utilization"] >= 0).all()
    assert (preds["aggregate_cpu_utilization"] <= 100).all()


def test_generate_predictions_works_with_different_holdout_size(tmp_path):
    _write_smoke_dataset(tmp_path, n_vms=8, n_timestamps=30)
    preds = generate_predictions(tmp_path, n_holdout=5)
    assert len(preds) == 5


def test_generate_predictions_raises_when_too_few_timestamps(tmp_path):
    _write_smoke_dataset(tmp_path, n_vms=5, n_timestamps=10)
    with pytest.raises(ValueError):
        generate_predictions(tmp_path, n_holdout=24)


def test_generate_predictions_do_not_depend_on_holdout_targets(tmp_path):
    _write_smoke_dataset(tmp_path, n_vms=8, n_timestamps=40)
    original = generate_predictions(tmp_path, n_holdout=8)

    cpu_path = tmp_path / "vm_cpu_readings.csv.gz"
    cpu = pd.read_csv(cpu_path)
    holdout_timestamps = sorted(cpu["timestamp"].unique())[-8:]
    is_holdout = cpu["timestamp"].isin(holdout_timestamps)
    cpu.loc[is_holdout, "avg_cpu"] = 45.0
    cpu = cpu.loc[~(is_holdout & cpu["vm_id"].isin(["vm_0", "vm_1"]))]
    cpu.to_csv(cpu_path, index=False, compression="gzip")

    changed_holdout = generate_predictions(tmp_path, n_holdout=8)
    pd.testing.assert_frame_equal(original, changed_holdout, check_dtype=False, atol=0, rtol=0)
