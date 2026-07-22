from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def train_index(self, index: pd.Index) -> pd.Index:
        return index[self.train_start : self.train_end]

    def test_index(self, index: pd.Index) -> pd.Index:
        return index[self.test_start : self.test_end]


def walk_forward_splits(
    index: pd.Index,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    purge: int = 0,
    embargo: int = 0,
) -> Iterable[WalkForwardSplit]:
    """Yield chronological train/test windows with purge and embargo gaps."""

    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if test_size <= 0:
        raise ValueError("test_size must be positive")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")

    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be positive")

    cursor = train_size
    n = len(index)
    while cursor + test_size <= n:
        train_start = max(0, cursor - train_size)
        train_end = max(train_start, cursor - purge)
        test_start = cursor
        test_end = cursor + test_size
        yield WalkForwardSplit(train_start, train_end, test_start, test_end)
        cursor = test_end + embargo + (step - test_size)
