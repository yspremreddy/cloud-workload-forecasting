# Cloud Workload Forecasting Challenge — Submission

Solution to the [Cloud Workload Forecasting Challenge](ASSIGNMENT.md) (dataset
contract: [DATASET.md](DATASET.md); submission contract: [SUBMISSION.md](SUBMISSION.md)).

Full write-up: [`reports/final_report.md`](reports/final_report.md).

## Install

Python 3.11 or newer is required.

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

Verify the install:

```bash
python -m pytest -q
```

Notebook execution, the Ridge configuration sweep, and report rendering use the declared `notebooks` extra installed above.

## Dataset

Not committed to the repo (see `.gitignore`). Download the `data-v1` release
and extract it into `challenge_data/`:

```powershell
Invoke-WebRequest -Uri "https://github.com/atomityhq/forecasting-challenge/releases/download/data-v1/data-v1.tar.gz" -OutFile "data-v1.tar.gz"
mkdir challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

Verify checksums (`challenge_data/SHA256SUMS`) before use; see [DATASET.md](DATASET.md)
for the full contract.

## Generate predictions

One command, from a clean checkout with the dataset in place:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This fits the frozen models on the first 110 timestamps and writes a forecast
for the final 24 (the schema required by [SUBMISSION.md](SUBMISSION.md)). It
does not depend on there being exactly 10,000 VMs or 134 timestamps — `--n-holdout`
(default 24) is the only assumption, and `tests/test_predict.py` exercises the
pipeline against a small synthetic dataset to confirm this.

## Reproduce evaluation and render the report

Run the existing tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Execute the EDA and modeling notebooks with saved outputs. The modeling notebook
runs the 12-configuration Ridge sweep, compares both targets on the final
24-point holdout, and evaluates the capacity policy using margins calibrated
only from the five training-window folds. It writes the fold and summary tables
to reports/.

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling_and_error_analysis.ipynb
```

For a local page-count and layout check, render the Markdown report to a
review-only PDF in the ignored virtual environment directory. This uses an
installed Chrome or Edge browser; set CHROME_PATH if it is in a custom location.
The PDF is a preview and Markdown remains the submission source.

```powershell
.\.venv\Scripts\python.exe tools/render_report.py --output .venv/final_report_preview.pdf
```

## Repository layout

```text
src/cwfc/
  data.py          raw file loading
  quality.py       structural data-quality audit
  targets.py       build active_vm_count(t) and aggregate_cpu_utilization(t)
  splits.py        chronological train/holdout split (last N timestamps)
  validation.py    expanding-window rolling-origin splits
  evaluate.py       runs a forecaster through rolling-origin validation + metrics
  baselines.py     persistence, moving-average, drift
  features.py      leakage-safe lag/rolling-stat features
  models.py        damped-trend ETS, ridge, gradient boosting (direct multi-step)
  uncertainty.py   lead-time-bucketed quantile margins from residuals
  policy.py        headroom policy: point forecast + margin
  metrics.py       MAE, RMSE, SMAPE, normalized MAE
  predict.py       CLI entry point (above)
notebooks/
  01_eda.ipynb                         data understanding / EDA (assignment section A)
  02_modeling_and_error_analysis.ipynb baselines, Ridge sweep, holdout, policy, errors (D-G)
reports/
  final_report.md   written report (sections 1-9 of the assignment)
  ridge_sweep_*.csv fold and summary results from the executed notebook
  figures/          figures referenced by the notebooks and report
tests/               pytest suite for every module above
```

Notebooks import from `src/cwfc` and render results; they do not duplicate any
forecasting logic.

## Main decisions (see the final report for full reasoning)

- **`active_vm_count`**: the released dataset only includes VMs with complete
  134-point traces, so this target is exactly constant (10,000) at every
  timestamp — confirmed under rolling-origin validation, not just inspection.
  It is forecast with persistence (not a hard-coded constant), which is both
  the data-correct answer here and the generically correct choice if a smoke
  dataset has real churn.
- **`aggregate_cpu_utilization`**: a noisy, mean-reverting series with no
  seasonality (an 11-hour window admits none) and autocorrelation that decays
  toward the 24-step horizon. Persistence is the strongest baseline tested
  (beats moving-average and drift). Damped-trend ETS was tried and
  **underperforms** the baseline, consistent with the series having no
  persistent trend to extrapolate. **Ridge regression with lag-only features and alpha=1 is selected** (11.7%
  mean MAE improvement over persistence across the five training-window folds).
  The saved alpha/feature sweep contains all fold-level results. On the final
  holdout, persistence has lower CPU MAE; the holdout was not used for retuning.
- **Validation**: expanding-window rolling-origin folds entirely inside the
  110-point public training window; the 24-point holdout is touched exactly
  once, at the end, for reporting.
- **Capacity policy**: point forecast + an empirical, lead-time-bucketed
  quantile margin derived from rolling-origin residuals, motivated directly by
  the error-analysis finding that the model underestimates peaks.
