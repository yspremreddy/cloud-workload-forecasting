import pandas as pd

from cwfc.evaluate import rolling_origin_evaluate, peak_offpeak_errors
from cwfc.baselines import persistence_forecast


def test_peak_offpeak_split_by_threshold():
    # ramp so persistence (flat forecast) is wrong by a growing amount at every step,
    # for both low and high true values
    values = [float(i % 20) for i in range(80)]
    df = pd.DataFrame({"timestamp": range(80), "value": values})
    res = rolling_origin_evaluate(
        df, target="value", forecaster=persistence_forecast,
        horizon=10, min_train_size=30, step=10, name="persistence",
    )
    out = peak_offpeak_errors(res, peak_threshold=10.0)
    assert out["peak_n"] + out["offpeak_n"] == sum(len(f.y_true) for f in res.folds)
    assert out["peak_mae"] > 0
    assert out["offpeak_mae"] > 0
