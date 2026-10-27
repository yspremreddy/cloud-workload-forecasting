import pandas as pd
import pytest

from cwfc.splits import train_holdout_split


def test_split_sizes():
    df = pd.DataFrame({"timestamp": range(134), "value": range(134)})
    train, holdout = train_holdout_split(df, n_holdout=24)
    assert len(train) == 110
    assert len(holdout) == 24
    assert list(train["timestamp"]) == list(range(110))
    assert list(holdout["timestamp"]) == list(range(110, 134))


def test_split_requires_sorted_input():
    df = pd.DataFrame({"timestamp": [0, 300, 100], "value": [1, 2, 3]})
    with pytest.raises(ValueError):
        train_holdout_split(df, n_holdout=1)


def test_split_raises_when_too_small():
    df = pd.DataFrame({"timestamp": range(10), "value": range(10)})
    with pytest.raises(ValueError):
        train_holdout_split(df, n_holdout=24)
