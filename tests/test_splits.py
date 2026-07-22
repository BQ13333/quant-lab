import pandas as pd

from quant_lab.validation.splits import walk_forward_splits


def test_walk_forward_splits_are_chronological():
    index = pd.bdate_range("2024-01-01", periods=100)
    splits = list(walk_forward_splits(index, train_size=40, test_size=10, purge=2, embargo=1))
    assert splits
    first = splits[0]
    assert first.train_start == 0
    assert first.train_end == 38
    assert first.test_start == 40
    assert first.test_end == 50
    assert first.train_end < first.test_start
