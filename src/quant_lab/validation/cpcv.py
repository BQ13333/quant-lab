from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CPCVResult:
    train_indices: pd.Index
    test_indices: pd.Index


def combinatorial_purged_cv_splits(
    index: pd.Index,
    n_splits: int,
    train_size: int | None = None,
    test_size: int | None = None,
    purge: int = 5,
    embargo: int = 5,
    *, label_start=None, label_end=None, one_bar_labels: bool = False,
) -> Iterable[CPCVResult]:
    """Generate combinatorial purged cross-validation splits.

    Divides the index into *n_splits* contiguous groups, then yields every
    possible combination of ``n_splits // 2`` groups as the test set (the
    remainder form the train set).  Purge and embargo gaps are applied so that
    train and test never overlap in time.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")

    n = len(index)
    if train_size is not None and train_size <= 0:
        raise ValueError("train_size must be positive when provided")
    if test_size is not None and test_size <= 0:
        raise ValueError("test_size must be positive when provided")
    if purge + embargo >= n:
        raise ValueError("purge and embargo leave no usable observations")
    group_size = n // n_splits
    if group_size < 2:
        raise ValueError("index too short for n_splits")

    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('unique ordered index required')
    if label_start is None or label_end is None:
        if label_start is not None or label_end is not None or one_bar_labels is not True:
            raise ValueError('label intervals or explicit one-bar labels required')
        label_start = label_end = index
    starts, ends = pd.Index(label_start), pd.Index(label_end)
    if len(starts) != n or len(ends) != n or starts.hasnans or ends.hasnans or not starts.is_monotonic_increasing or any(starts > ends):
        raise ValueError('invalid label intervals')
    groups: list[pd.Index] = []
    for i in range(n_splits):
        start = i * group_size
        end = start + group_size if i < n_splits - 1 else n
        groups.append(index[start:end])

    # Test group count = floor(n_splits / 2)
    test_group_count = n_splits // 2
    if test_group_count < 1 or test_group_count >= n_splits:
        raise ValueError("invalid test_group_count derived from n_splits")

    for test_combo in combinations(range(n_splits), test_group_count):
        test_set = pd.Index([])
        for g in test_combo:
            test_set = test_set.union(groups[g])

        train_set = pd.Index([])
        for g in range(n_splits):
            if g not in test_combo:
                train_set = train_set.union(groups[g])

        if train_set.empty or test_set.empty:
            continue

        train_positions = index.get_indexer(train_set)
        excluded: set[int] = set()
        for group_number in test_combo:
            start = group_number * group_size
            end = (group_number + 1) * group_size if group_number < n_splits - 1 else n
            excluded.update(range(max(0, start - purge), min(n, end + embargo)))
            # Multi-bar test labels can extend past the final test row.
            # The embargo begins after that information horizon, not its row.
            horizon_end = ends[start:end].max()
            after = int(starts.searchsorted(horizon_end, side='right'))
            excluded.update(range(after, min(n, after + embargo)))
        for test_position in index.get_indexer(test_set):
            excluded.update(i for i in train_positions
                            if starts[i] <= ends[test_position] and ends[i] >= starts[test_position])
        valid_train_positions = [position for position in train_positions if position not in excluded]
        if not valid_train_positions:
            continue
        if train_size is not None and len(valid_train_positions) < train_size:
            continue
        if test_size is not None and len(test_set) < test_size:
            continue
        train_idx = index[valid_train_positions]
        test_idx = test_set

        yield CPCVResult(train_indices=train_idx, test_indices=test_idx)


purged_splits = combinatorial_purged_cv_splits
