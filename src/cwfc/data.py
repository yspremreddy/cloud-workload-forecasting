"""Loading raw challenge data files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_cpu_readings(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "vm_cpu_readings.csv.gz"
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="raise")
    return df


def load_vm_metadata(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "vm_metadata.csv.gz"
    df = pd.read_csv(path)
    return df
