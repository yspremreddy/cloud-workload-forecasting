import pandas as pd

from cwfc.quality import audit_cpu_readings


def test_audit_detects_constant_active_vm_count():
    cpu = pd.DataFrame(
        {
            "timestamp": [0, 0, 300, 300],
            "vm_id": ["a", "b", "a", "b"],
            "min_cpu": [1.0, 1.0, 1.0, 1.0],
            "max_cpu": [3.0, 3.0, 3.0, 3.0],
            "avg_cpu": [2.0, 2.0, 2.0, 2.0],
        }
    )
    report = audit_cpu_readings(cpu)
    assert report.n_vms == 2
    assert report.n_timestamps == 2
    assert report.cadence_seconds == 300
    assert report.vms_with_incomplete_trace == 0
    assert report.n_duplicate_vm_timestamp == 0
    assert any("constant" in n for n in report.notes)


def test_audit_flags_duplicates_and_missing():
    cpu = pd.DataFrame(
        {
            "timestamp": [0, 0, 0],
            "vm_id": ["a", "a", "b"],
            "min_cpu": [1.0, 1.0, 1.0],
            "max_cpu": [3.0, 3.0, 3.0],
            "avg_cpu": [2.0, 2.0, None],
        }
    )
    report = audit_cpu_readings(cpu)
    assert report.n_duplicate_vm_timestamp == 1
    assert report.n_missing_avg_cpu == 1
