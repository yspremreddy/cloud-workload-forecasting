# Cloud Workload Forecasting Challenge — Final Report

## 1. Executive summary

This report answers the assignment's question — *how predictable is this
cloud workload, how accurately can near-term demand be forecast, and what
should an infrastructure team do with the forecast?* — for a curated,
11-hour, 10,000-VM slice of the Azure Public Dataset V2 sampled every five
minutes (134 points per VM).

Two targets were required. The first, `active_vm_count(t)`, turned out to be
**exactly constant (10,000) at every one of the 134 timestamps**, a direct
consequence of the dataset only including VMs with complete traces — there is
no churn in this release to forecast, and the right model is to recognize
that rather than fit one. The second, `aggregate_cpu_utilization(t)`, is a
real but narrow forecasting problem: a noisy, mean-reverting series
(range ≈9.0–9.9%, std≈0.19pp) with short, fast-decaying autocorrelation and no
seasonality (impossible in an 11-hour window). Persistence (last observed
value) is a strong baseline. A **Ridge model using lag-only features and alpha=1 improves mean validation
MAE by 11.7%** over persistence on the five expanding-window folds. The
selected model has lower validation error on these folds, but on the final
holdout its MAE is higher than persistence. That held-out difference is
reported without retuning. Its clearest failure mode is
**underestimating peak periods** (a ridge shrinkage effect), which directly
motivates an asymmetric, lead-time-aware headroom margin for the capacity
recommendation in Section 8.

## 2. Data understanding and preprocessing

**Structural audit** (`src/cwfc/quality.py`, run in `notebooks/01_eda.ipynb`):
the raw release (1,340,000 rows, 10,000 VMs × 134 readings) sits on a
perfectly regular 5-minute grid — 134 unique timestamps, constant 300-second
step, zero duplicate `(vm_id, timestamp)` pairs, zero missing `avg_cpu` /
`min_cpu` / `max_cpu`, and `min_cpu ≤ avg_cpu ≤ max_cpu` holds on every row.
Every VM has exactly 134 readings; metadata and CPU-reading VM populations
match 1:1 with no duplicates. This matches the release's own `DATA_CARD.md`,
which states the dataset was validated to this exact standard before
publication.

**Why it matters operationally:** none of the usual production
data-quality failure modes (dropped samples, clock skew, duplicate ingestion)
are present here, so no imputation or resampling decision was forced on us —
any model performance difference we measure reflects the workload, not
data-cleaning artifacts. A real telemetry pipeline cannot assume this; it
would need the same checks run continuously, not as a one-off audit.

**VM lifetimes:** all 10,000 VMs were already running at the window start
(`timestamp_vm_created = 0`), and almost all have a recorded deletion
timestamp near 2.59M seconds — the length of the full 30-day upstream trace —
meaning they simply outlive this 134-point window. Combined with "every VM
has 134 readings," there is **no VM churn inside this dataset by
construction**: the curators only kept VMs with complete traces. This is why
`active_vm_count(t)` is constant (Section 3).

**Target construction** (`src/cwfc/targets.py`): `active_vm_count(t)` counts
each VM at most once per timestamp, counting only rows with a non-null
`avg_cpu`; `aggregate_cpu_utilization(t)` is the mean of `avg_cpu` over the
same valid rows. A row with a missing `avg_cpu` is excluded from both targets
rather than imputed — on this release the point is moot (zero missing
values), but the rule is applied generically so it also behaves correctly
against a smaller smoke dataset that does have gaps. No resampling or gap-
filling is performed; the target series has exactly one row per observed
timestamp.

**Metadata leakage exclusion:** `vm_category`, `virtual_core_bucket`, and
`memory_gb_bucket` are static per-VM attributes and were used only for
segment-level EDA, never as model features computed from future information.
`max_cpu`, `avg_cpu`, `p95_max_cpu`, and `timestamp_vm_deleted` in
`vm_metadata.csv.gz` are lifetime-level statistics computed over the *whole*
VM trace (per `DATASET.md`) and were excluded entirely, as required.

## 3. Workload analysis

`active_vm_count(t)` has zero variance across all 134 timestamps — confirmed
both by direct inspection and by rolling-origin validation of every baseline
(all score exactly MAE = RMSE = 0.0). There is nothing further to analyze or
model for this target in this release.

`aggregate_cpu_utilization(t)` ranges roughly 9.0–9.9% with mean 9.4% and
std 0.19pp (`reports/figures/target_series_and_distribution.png`). Lag-1
autocorrelation on the training window is 0.68, decaying through 0.44 at
lag 12 (1 hour) and turning **slightly negative by lag 18–24**
(`reports/figures/autocorrelation.png`) — by the 2-hour forecast horizon the
series is close to indistinguishable from noise around its local mean. There
is no daily/weekly cycle to exploit; the entire window is ~11 hours.

The series shows one notable feature: a mid-window bump (roughly
t=15000–22000s) rising to ~9.9% before decaying back down through the end of
training into the holdout (train mean 9.41% vs. holdout mean 9.27%, a mild
downward drift smaller than one standard deviation of the series).
Segment-level breakdown (`reports/figures/segment_by_category.png`) shows
this bump-and-decay is driven mainly by the "Unknown" `vm_category`
(swinging ~9–12%), while "Interactive" (~11–12%) and "Delay-insensitive"
(~8–9%) are comparatively flat. Category level is also a large, stable
effect in its own right — separation between categories (~3.3pp) is an order
of magnitude larger than the aggregate's temporal variance — which is a
reasonable part of *why* the aggregate is so stable: averaging CPU over
10,000 VMs across a few persistent sub-populations washes out VM-level noise
by the law of large numbers.

By contrast, **`virtual_core_bucket` and `memory_gb_bucket` show no clean,
monotonic relationship with CPU level.** Mean CPU by core count is 7.9%
(4 cores), 9.7% (2 and 8 cores), 11.4% (24 cores), but only 6.6% at >24
cores — despite that being the *largest* VM size; the memory-bucket pattern
is similarly non-monotonic (8.8%/11.3%/9.2%/7.8%/11.3%/6.6% across
2/4/8/32/64/>64 GB). The largest-size buckets (`>24` cores, `>64` GB) are
also the most volatile (std ≈2.0pp vs. ≈0.2–0.4pp elsewhere), which is
largely a small-sample artifact — they contain only 10 of the 10,000 VMs, so
their means are unstable. **Operationally, provisioned VM size is a much
weaker and noisier lever on utilization than workload category in this
sample** — an infrastructure team reasoning about expected load should weight
workload intent (`vm_category`) far more heavily than instance size, and
should treat any size-bucket conclusion for the `>24`/`>64` buckets as
low-confidence given how few VMs support it.

**Operational read:** the dataset's construction (complete-trace VMs only)
suppresses exactly the churn signal a real-time capacity system would care
about most. In production, `active_vm_count` would likely be the more
volatile and operationally urgent target; this release cannot demonstrate
that, and the report treats this as a limitation of the sample, not a
property of real Azure workloads (Section 9).

## 4. Forecasting approach

Models were tried in increasing order of complexity, per the assignment's
guidance that complexity earns nothing by itself:

1. **Baselines** (`src/cwfc/baselines.py`): persistence (last value),
   moving average (windows 6/12/24), drift (linear extrapolation from first
   to last training point).
2. **Damped-trend ETS** (`src/cwfc/models.py`): Holt's damped-trend
   exponential smoothing, fit by grid-searching (α, β, φ) to minimize
   in-sample one-step SSE. Damping (φ<1) flattens the trend at longer
   horizons rather than extrapolating it indefinitely.
3. **Ridge regression and gradient boosting on lag/rolling-stat features**
   (`src/cwfc/features.py`, `models.py`): lags {1,2,3,6,12} and rolling
   mean/std over windows {3,6,12}, every feature built from `series.shift(1)`
   or later so it uses only values strictly before the forecast origin —
   never the value being predicted or anything after it. Forecasting uses a
   **direct multi-step** strategy: one model per step-ahead h∈{1..24}, each
   trained to predict `y[t+h]` from features at `t`, rather than recursively
   feeding back single-step predictions — this avoids compounding one-step
   errors across a 24-step horizon.

## 5. Validation methodology

All model selection used **expanding-window rolling-origin validation**
inside the 110-point public training window
(`src/cwfc/validation.py::expanding_window_index_splits`,
`src/cwfc/evaluate.py`): five origins at training sizes 40, 50, 60, 70,
and 80, each forecasting the next 24 observations. At every origin, the
Ridge models build their lag features from that fold's training rows and fit
separate models using only those rows. No holdout observations enter feature
selection, regularization selection, fitting, or uncertainty calibration.

### Ridge regularization and feature-set comparison

The executed modeling notebook compares alphas 0.1, 1, 5, and 50 across
three feature sets using the same five folds. The full set uses lags
{1, 2, 3, 6, 12} and rolling mean/std windows {3, 6, 12}; compact uses lags
{1, 2, 3} and windows {3, 6}; lag-only uses lags {1, 2, 3, 6, 12} with no
rolling features. Each fold independently constructs its training examples
and fits each per-horizon Ridge model. There is no preprocessing transform to
fit; validation targets are used only for metrics.

The lowest mean validation MAE selects lag-only Ridge with alpha=1. Its
mean MAE is 0.211790 and mean RMSE is 0.257646. The previous full feature
set at alpha=1 scores 0.212701 MAE and 0.259011 RMSE, so removing the rolling
features improves mean MAE by 0.000911 on these folds. The complete results
for all 12 configurations and all 60 fold/configuration rows are saved in
`reports/ridge_sweep_summary.csv` and `reports/ridge_sweep_folds.csv`.

Selected configuration, fold-level metrics (CPU percentage points):

| fold | training rows | origin timestamp (s) | MAE | RMSE |
|---:|---:|---:|---:|---:|
| 1 | 40 | 11,700 | 0.220677 | 0.285781 |
| 2 | 50 | 14,700 | 0.315633 | 0.352433 |
| 3 | 60 | 17,700 | 0.190377 | 0.237333 |
| 4 | 70 | 20,700 | 0.163328 | 0.200489 |
| 5 | 80 | 23,700 | 0.168938 | 0.212193 |
| **Mean** | — | — | **0.211790** | **0.257646** |

The 24-point holdout is kept separate and used only for final reporting
(Section 6). The capacity margin is calibrated from the selected model's
five training-window validation folds, before holdout coverage is measured.

## 6. Results vs. baseline

| model | mean MAE | mean RMSE | vs. persistence |
|---|---:|---:|---:|
| **Ridge (lag-only, alpha=1)** | **0.2118** | **0.2576** | **+11.7%** |
| gbm (lags + rolling stats) | 0.2264 | 0.2699 | +5.6% |
| persistence (baseline) | 0.2398 | 0.2804 | — |
| moving_avg_6 | 0.2538 | 0.2917 | −5.8% |
| moving_avg_12 | 0.2606 | 0.2987 | −8.7% |
| drift | 0.2675 | 0.3117 | −11.6% |
| moving_avg_24 | 0.2677 | 0.3025 | −11.7% |
| damped-trend ETS | 0.2821 | 0.3201 | −17.7% |

Persistence is the strongest simple baseline. Moving averages, drift, and
damped-trend ETS underperform it on the same folds. The selected Ridge
configuration improves validation MAE over persistence from 0.239761 to
0.211790 ((0.239761 - 0.211790) / 0.239761 = 11.7%). Its fold scores vary,
with the highest errors in folds 1–2 during the workload bump; this limited
five-fold comparison supports the selected configuration but does not
establish that the small improvement will generalize to other traces. SMAPE
is 2.23% for selected Ridge versus 2.51% for persistence. The smallest
observed denominator, |y_true| + |y_pred|, is 18.48, well above the metric's
1e-8 zero guard.

For `active_vm_count`, all baselines score exactly 0.0 MAE/RMSE under the
same validation scheme, confirming the target requires no further modeling.

### Final 24-point holdout (reporting only)

The following scores are calculated from the submitted predictions and the
final 24 observed targets. The persistence baseline carries forward the last
value in the 110-point training window. Relative MAE improvement is
(baseline MAE - selected-model MAE) / baseline MAE × 100. For active VM
count both errors are zero, so percentage improvement is undefined.

| target | selected model | model MAE | model RMSE | persistence MAE | persistence RMSE | relative MAE improvement |
|---|---|---:|---:|---:|---:|---:|
| active_vm_count | persistence | 0.000 | 0.000 | 0.000 | 0.000 | undefined (zero baseline error) |
| aggregate_cpu_utilization | Ridge, lag-only, alpha=1 | 0.158117 pp | 0.171002 pp | 0.146782 pp | 0.184785 pp | -7.72% |

On this final holdout, selected Ridge has lower RMSE than persistence but
higher MAE. This differs from five-fold validation: the held-out observations
are reported after model selection and were not used to retune the model.
The selected CPU predictions average 9.4172% versus 9.2690% observed. The
validation-calibrated capacity policy covers 24 of 24 holdout observations;
this coverage is a final diagnostic, not a target used to choose its quantile.

## 7. Error and failure analysis

**Validation error rises with lead time** for both persistence and selected
Ridge (`reports/figures/mae_by_step_ahead.png`), consistent with the rapid
autocorrelation decay. Ridge is not better at every step; its pooled five-fold
mean MAE is the comparison used for model selection and reporting.

Using the training-only 75th-percentile threshold, Ridge peak MAE is 0.316
(50 validation observations) versus 0.272 for persistence; peak bias is
-0.299 percentage points, showing under-forecasting. Off-peak Ridge MAE is
0.138 (70 observations) versus 0.217 for persistence. The largest absolute
errors cluster in early validation folds 1–3, which forecast into the workload
bump before enough of it is present in their training history. This is the
reason for assessing an upward headroom margin separately from the point
forecast.

## 8. Capacity-planning recommendation

**Policy: provisioned target = point forecast + lead-time-bucketed quantile
margin**, computed from the same rolling-origin residuals as Section 7
(`src/cwfc/uncertainty.py`, `policy.py`). Residuals are pooled into three
lead-time buckets (steps 1–8, 9–16, 17–24) rather than estimated per exact
step, because 5 folds gives far too few residuals (5) for a stable per-step
quantile; pooling trades resolution for a usable sample (~40 residuals per
bucket).

At the 90th percentile, the margins are 0.427 percentage points (steps 1–8),
0.477 pp (9–16), and 0.435 pp (17–24), for mean headroom of 0.446 pp. On the
final holdout, applying these already-fixed margins to the submitted point
forecasts covers 24 of 24 CPU observations. This is a holdout diagnostic; the
quantile and buckets are calibrated from training-window validation residuals.

**The under- vs. over-provisioning trade-off:** choosing a higher residual
quantile adds more reserved CPU and reduces the chance that realized load
exceeds capacity; a lower quantile reserves less CPU and accepts more
under-capacity risk. Because peak under-forecasting is observed in validation,
the existing policy uses the 90th percentile. The short trace and five
validation folds do not establish a production coverage guarantee.

For `active_vm_count`: since it is exactly constant in this release, no
margin is applied or needed — the recommendation is simply to provision for
10,000 concurrent VMs with no additional buffer, while flagging to the
infrastructure team (Section 9) that this number would carry real buffer
requirements in a production setting with actual VM churn, which this sample
cannot demonstrate.

## 9. Limitations and future work

- **`active_vm_count` is not a real forecasting problem in this release.**
  The dataset's "complete traces only" selection criterion removes VM churn
  by construction. A production system would need churn data (VM
  creation/deletion events within the forecast window) to forecast this
  target meaningfully; this is the single most consequential limitation of
  the exercise and is stated plainly rather than disguised with elaborate
  modeling of a flat line.
- **Small training set (110 points) limits model complexity that can be
  validated.** 5 rolling-origin folds is enough to see a stable, non-fragile
  effect (as checked via the four-alpha, three-feature-set sweep) but not enough to
  estimate per-step uncertainty or to trust a much more expressive model
  without risking an unverifiable overfit; this is why gradient boosting was
  tried but ridge, not GBM, was selected.
- **No external data or cross-cloud comparison was pursued** (an optional
  extension) given the time budget; the core requirements were prioritized.
- **Segment-level (bottom-up) forecasting was investigated only
  descriptively** (Section 3), not as a competing model. A natural next step
  would be forecasting each `vm_category`'s series separately and comparing
  the recombined bottom-up forecast's accuracy to the direct aggregate-level
  model — cheap to try given only 3 well-separated segments, and could reduce
  the peak-underestimation bias if the bump is isolated to one segment, as
  the EDA suggests.
- **The peak-underestimation bias (Section 7) is a model property worth
  addressing directly**, not only compensating for with a margin — e.g. a
  quantile-regression or asymmetric-loss variant of ridge, trained to target
  the 90th percentile directly rather than the mean, would be a natural
  follow-up now that the failure mode is identified.
- **Lead-time margin buckets (Section 8) are coarse** (3 buckets from 5
  folds); with more training history, per-step or smoothly interpolated
  margins would likely be a better match to how the error curve
  (`reports/figures/mae_by_step_ahead.png`) actually behaves than three flat
  steps.
