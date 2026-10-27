import pandas as pd

from cwfc.features import build_features, feature_columns, make_direct_training_rows


def test_build_features_lag1_uses_only_past_value():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    feats = build_features(s, lags=(1, 2), roll_windows=(2,))
    assert feats.loc[2, "lag_1"] == 20.0
    assert feats.loc[2, "lag_2"] == 10.0
    # roll_mean_2 at row 2 is mean of rows 0,1 (shifted by 1, excludes row 2 itself)
    assert feats.loc[2, "roll_mean_2"] == 15.0


def test_build_features_early_rows_are_nan():
    s = pd.Series([1.0, 2.0, 3.0])
    feats = build_features(s, lags=(1, 2), roll_windows=(2,))
    assert feats.loc[0].isna().all()


def test_feature_columns_order_matches_build_features():
    cols = feature_columns(lags=(1, 2), roll_windows=(3,))
    assert cols == ["lag_1", "lag_2", "roll_mean_3", "roll_std_3"]


def test_make_direct_training_rows_targets_are_future_values():
    s = pd.Series([float(i) for i in range(20)])
    rows = make_direct_training_rows(s, horizon=3, lags=(1,), roll_windows=(2,))
    assert set(rows.keys()) == {1, 2, 3}
    df1 = rows[1]
    # for h=1, y at origin t should equal series value at t+1
    for idx, row in df1.iterrows():
        assert row["y"] == s.iloc[idx + 1]
    # no NaNs should survive
    assert not df1.isna().any().any()
