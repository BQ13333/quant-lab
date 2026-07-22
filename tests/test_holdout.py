import pandas as pd
import pytest

from quant_lab.validation.holdout import HoldoutManager


def test_holdout_manager_splits_index():
    index = pd.bdate_range("2024-01-01", periods=100)
    manager = HoldoutManager(index, holdout_fraction=0.2)
    assert len(manager.train_index) == 80
    assert len(manager.holdout_index) == 20


def test_is_holdout():
    index = pd.bdate_range("2024-01-01", periods=100)
    manager = HoldoutManager(index, holdout_fraction=0.2)
    assert manager.is_holdout(index[90]) is True
    assert manager.is_holdout(index[50]) is False


def test_assert_no_leakage_passes():
    index = pd.bdate_range("2024-01-01", periods=100)
    manager = HoldoutManager(index, holdout_fraction=0.2)
    search_results = [
        {"params": {"a": 1}, "fold_dates": [index[10], index[20]]},
    ]
    manager.assert_no_holdout_optimization(search_results)  # should not raise


def test_assert_no_leakage_raises():
    index = pd.bdate_range("2024-01-01", periods=100)
    manager = HoldoutManager(index, holdout_fraction=0.2)
    search_results = [
        {"params": {"a": 1}, "fold_dates": [index[10], index[85]]},
    ]
    with pytest.raises(RuntimeError):
        manager.assert_no_holdout_optimization(search_results)
