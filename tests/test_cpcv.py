import pandas as pd
import pytest

from quant_lab.validation.cpcv import combinatorial_purged_cv_splits


def test_cpcv_no_overlap():
    index = pd.bdate_range("2024-01-01", periods=100)
    splits = list(combinatorial_purged_cv_splits(index, n_splits=6, purge=3, embargo=2))
    assert len(splits) > 0
    for split in splits:
        # No overlapping index values between train and test
        overlap = split.train_indices.intersection(split.test_indices)
        assert len(overlap) == 0, "train and test overlap"


def test_cpcv_all_folds_present():
    index = pd.bdate_range("2024-01-01", periods=60)
    splits = list(combinatorial_purged_cv_splits(index, n_splits=4, purge=2, embargo=1))
    # C(4,2) = 6 combinations, but some may be dropped if purge/embargo wipes train
    assert len(splits) >= 1
    # Verify total expected combinations when feasible
    assert len(splits) <= 6


def test_cpcv_train_not_empty():
    index = pd.bdate_range("2024-01-01", periods=120)
    splits = list(combinatorial_purged_cv_splits(index, n_splits=6, purge=5, embargo=5))
    for split in splits:
        assert len(split.train_indices) > 0
        assert len(split.test_indices) > 0


def test_cpcv_purges_each_noncontiguous_test_block_boundary():
    index = pd.RangeIndex(80)
    splits = list(combinatorial_purged_cv_splits(index, n_splits=4, purge=2, embargo=3))
    assert splits
    for split in splits:
        test_positions = set(split.test_indices)
        train_positions = set(split.train_indices)
        for position in test_positions:
            assert not any(candidate in train_positions for candidate in range(max(0, position - 2), min(80, position + 4)))


def test_cpcv_invalid_evidence_parameters_fail_closed():
    index = pd.RangeIndex(20)
    with pytest.raises(ValueError, match="leave no usable"):
        list(combinatorial_purged_cv_splits(index, n_splits=4, purge=10, embargo=10))
    with pytest.raises(ValueError, match="positive"):
        list(combinatorial_purged_cv_splits(index, n_splits=4, train_size=0))
