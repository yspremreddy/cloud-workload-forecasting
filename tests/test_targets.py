import pandas as pd

from cwfc.targets import build_target_series


def test_active_vm_count_counts_each_vm_once():
    cpu = pd.DataFrame(
        {
            "timestamp": [0, 0, 0, 300, 300],
            "vm_id": ["a", "b", "a", "a", "b"],
            "min_cpu": [1.0, 2.0, 1.0, 1.0, 2.0],
            "max_cpu": [3.0, 4.0, 3.0, 3.0, 4.0],
            "avg_cpu": [2.0, 3.0, 2.0, 2.0, 3.0],
        }
    )
    out = build_target_series(cpu)
    assert list(out["timestamp"]) == [0, 300]
    # duplicate (a, 0) row should not double-count vm 'a'
    assert list(out["active_vm_count"]) == [2, 2]


def test_missing_avg_cpu_excluded_from_both_targets():
    cpu = pd.DataFrame(
        {
            "timestamp": [0, 0],
            "vm_id": ["a", "b"],
            "min_cpu": [1.0, 2.0],
            "max_cpu": [3.0, 4.0],
            "avg_cpu": [2.0, None],
        }
    )
    out = build_target_series(cpu)
    assert out.loc[0, "active_vm_count"] == 1
    assert out.loc[0, "aggregate_cpu_utilization"] == 2.0


def test_output_sorted_chronologically():
    cpu = pd.DataFrame(
        {
            "timestamp": [300, 0, 300, 0],
            "vm_id": ["a", "a", "b", "b"],
            "min_cpu": [1.0, 1.0, 1.0, 1.0],
            "max_cpu": [3.0, 3.0, 3.0, 3.0],
            "avg_cpu": [2.0, 1.0, 2.0, 1.0],
        }
    )
    out = build_target_series(cpu)
    assert list(out["timestamp"]) == [0, 300]
