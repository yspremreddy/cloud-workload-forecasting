"""Run a forecaster (baseline or model) through expanding-window rolling-origin
validation, entirely within the public training window — never the holdout."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from cwfc.metrics import mae, rmse
from cwfc.validation import expanding_window_index_splits

Forecaster = Callable[[pd.DataFrame, pd.DataFrame, str], pd.Series]


@dataclass
class FoldResult:
    fold: int
    train_end_idx: int
    origin_timestamp: int
    mae: float
    rmse: float
    y_true: list[float] = field(repr=False)
    y_pred: list[float] = field(repr=False)


@dataclass
class EvaluationResult:
    name: str
    target: str
    folds: list[FoldResult]

    @property
    def mean_mae(self) -> float:
        return float(np.mean([f.mae for f in self.folds]))

    @property
    def mean_rmse(self) -> float:
        return float(np.mean([f.rmse for f in self.folds]))

    def per_step_errors(self) -> pd.DataFrame:
        """MAE/RMSE broken out by step-ahead (1..horizon), pooled across folds."""
        rows = []
        for f in self.folds:
            for step, (yt, yp) in enumerate(zip(f.y_true, f.y_pred), start=1):
                rows.append({"step_ahead": step, "y_true": yt, "y_pred": yp})
        df = pd.DataFrame(rows)
        out = df.groupby("step_ahead").apply(
            lambda g: pd.Series(
                {"mae": mae(g["y_true"], g["y_pred"]), "rmse": rmse(g["y_true"], g["y_pred"])}
            ),
            include_groups=False,
        )
        return out.reset_index()


def rolling_origin_evaluate(
    train_df: pd.DataFrame,
    target: str,
    forecaster: Forecaster,
    horizon: int = 24,
    min_train_size: int = 40,
    step: int | None = None,
    timestamp_col: str = "timestamp",
    name: str = "model",
) -> EvaluationResult:
    """Evaluate `forecaster` across expanding-window origins inside `train_df`.

    `forecaster(train_subset, template, target)` must return predictions aligned
    to `template`'s index/length, exactly like the functions in `cwfc.baselines`.
    This never touches rows outside `train_df`, so it is safe to call on the
    110-point public training window without any holdout leakage.
    """
    splits = expanding_window_index_splits(
        n_rows=len(train_df), horizon=horizon, min_train_size=min_train_size, step=step
    )
    if not splits:
        raise ValueError(
            f"no rolling-origin folds fit: n_rows={len(train_df)}, horizon={horizon}, "
            f"min_train_size={min_train_size}"
        )

    folds: list[FoldResult] = []
    for i, split in enumerate(splits):
        fold_train = train_df.iloc[: split.train_end_idx]
        fold_val = train_df.iloc[split.val_start_idx : split.val_end_idx]
        # Validation labels remain available for scoring only. Give the
        # forecaster the known timestamps and horizon, not actual targets.
        fold_template = fold_val[[timestamp_col]].copy()
        y_pred = forecaster(fold_train, fold_template, target)
        y_true = fold_val[target].astype(float).to_numpy()
        y_pred_arr = np.asarray(y_pred, dtype=float)
        folds.append(
            FoldResult(
                fold=i,
                train_end_idx=split.train_end_idx,
                origin_timestamp=int(fold_train[timestamp_col].iloc[-1]),
                mae=mae(y_true, y_pred_arr),
                rmse=rmse(y_true, y_pred_arr),
                y_true=y_true.tolist(),
                y_pred=y_pred_arr.tolist(),
            )
        )
    return EvaluationResult(name=name, target=target, folds=folds)


def peak_offpeak_errors(result: EvaluationResult, peak_threshold: float) -> dict[str, float]:
    """Split pooled fold errors by whether the true value was at/above a peak
    threshold (computed from training data only, passed in by the caller) to
    see whether the model degrades specifically during high-load periods."""
    peak_true: list[float] = []
    peak_pred: list[float] = []
    off_true: list[float] = []
    off_pred: list[float] = []
    for f in result.folds:
        for yt, yp in zip(f.y_true, f.y_pred):
            if yt >= peak_threshold:
                peak_true.append(yt)
                peak_pred.append(yp)
            else:
                off_true.append(yt)
                off_pred.append(yp)
    out: dict[str, float] = {}
    if peak_true:
        out["peak_mae"] = mae(peak_true, peak_pred)
        out["peak_rmse"] = rmse(peak_true, peak_pred)
        out["peak_n"] = float(len(peak_true))
        out["peak_bias"] = float(np.mean(np.asarray(peak_pred) - np.asarray(peak_true)))
    if off_true:
        out["offpeak_mae"] = mae(off_true, off_pred)
        out["offpeak_rmse"] = rmse(off_true, off_pred)
        out["offpeak_n"] = float(len(off_true))
        out["offpeak_bias"] = float(np.mean(np.asarray(off_pred) - np.asarray(off_true)))
    return out


def summarize(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": r.name,
                "target": r.target,
                "n_folds": len(r.folds),
                "mean_mae": r.mean_mae,
                "mean_rmse": r.mean_rmse,
            }
            for r in results
        ]
    )
