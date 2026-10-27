"""Reproducible prediction CLI.

    python -m cwfc.predict --data challenge_data --output predictions.csv

Loads the raw VM-level readings, builds the two required aggregate target
series, splits off the last N timestamps (default 24) as the holdout horizon,
fits the selected models on the training portion only, and writes
`predictions.csv` in the schema required by SUBMISSION.md.

Model selection (frozen after rolling-origin validation inside the training
window -- see notebooks/02_modeling_and_error_analysis.ipynb):
  - active_vm_count: persistence (carries forward the last training value).
    Validated at MAE = 0.0 on this release because the dataset only contains
    VMs with complete traces, so there is no churn signal to model; persistence
    is also the generically correct choice if a smoke dataset has (small)
    real churn, rather than hard-coding a constant.
  - aggregate_cpu_utilization: Ridge regression on lag-only features with alpha=1,
    selected by a documented sweep over four alpha values and three feature sets
    using five expanding-window folds inside the 110-point training window.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cwfc.baselines import persistence_forecast
from cwfc.data import load_cpu_readings
from cwfc.models import ridge_forecast
from cwfc.experiments import DEFAULT_FEATURE_SETS, SELECTED_RIDGE_ALPHA, SELECTED_RIDGE_FEATURE_SET
from cwfc.splits import train_holdout_split
from cwfc.targets import build_target_series


def generate_predictions(data_dir: str | Path, n_holdout: int = 24) -> pd.DataFrame:
    cpu = load_cpu_readings(data_dir)
    targets = build_target_series(cpu)
    train, holdout_template = train_holdout_split(targets, n_holdout=n_holdout)
    # Forecasters need only known timestamps and horizon length; keep held-out
    # target values on the scoring side of the model boundary.
    forecast_template = holdout_template[["timestamp"]].copy()

    active_vm_count = persistence_forecast(train, forecast_template, "active_vm_count")
    selected_lags, selected_windows = DEFAULT_FEATURE_SETS[SELECTED_RIDGE_FEATURE_SET]
    aggregate_cpu_utilization = ridge_forecast(
        train, forecast_template, "aggregate_cpu_utilization",
        alpha=SELECTED_RIDGE_ALPHA, lags=selected_lags, roll_windows=selected_windows,
    )

    # Defensive clip to the schema's required ranges; the model is not
    # expected to violate these given the EDA, but the submission contract
    # requires it regardless of what any particular fit happens to produce.
    active_vm_count = active_vm_count.clip(lower=0)
    aggregate_cpu_utilization = aggregate_cpu_utilization.clip(lower=0.0, upper=100.0)

    predictions = pd.DataFrame(
        {
            "timestamp": holdout_template["timestamp"].to_numpy(),
            "active_vm_count": active_vm_count.round().astype(int).to_numpy(),
            "aggregate_cpu_utilization": aggregate_cpu_utilization.to_numpy(),
        }
    )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions.csv for the Cloud Workload Forecasting Challenge")
    parser.add_argument("--data", required=True, help="Path to the challenge_data directory")
    parser.add_argument("--output", required=True, help="Path to write predictions.csv")
    parser.add_argument(
        "--n-holdout", type=int, default=24, help="Number of final timestamps to forecast (default: 24)"
    )
    args = parser.parse_args()

    predictions = generate_predictions(args.data, n_holdout=args.n_holdout)
    predictions.to_csv(args.output, index=False)
    print(f"Wrote {len(predictions)} rows to {args.output}")


if __name__ == "__main__":
    main()
