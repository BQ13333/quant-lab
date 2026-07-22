from __future__ import annotations

from typing import Any

import pandas as pd


class HoldoutManager:
    """Manages a frozen final holdout set to prevent leakage into optimisation."""

    def __init__(self, index: pd.Index, holdout_fraction: float = 0.2) -> None:
        if not 0.0 < holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be between 0 and 1")
        self._index = index
        self._holdout_fraction = holdout_fraction
        split_point = int(len(index) * (1 - holdout_fraction))
        self._holdout_index = index[split_point:]
        self._train_index = index[:split_point]

    @property
    def holdout_index(self) -> pd.Index:
        return self._holdout_index

    @property
    def train_index(self) -> pd.Index:
        return self._train_index

    def is_holdout(self, date: pd.Timestamp | str) -> bool:
        """Return True if *date* lies in the holdout set."""
        ts = pd.Timestamp(date)
        return ts in self._holdout_index

    def assert_no_holdout_optimization(self, search_results: list[dict[str, Any]]) -> None:
        """Raise RuntimeError if any search result was validated on holdout data."""

        for result in search_results:
            fold_dates = result.get("fold_dates", [])
            for d in fold_dates:
                if isinstance(d, pd.Timestamp) and d in self._holdout_index:
                    raise RuntimeError(
                        f"Holdout leakage detected: result {result.get('params', {})} used holdout date {d}"
                    )
                if isinstance(d, (list, tuple, pd.DatetimeIndex)):
                    for sub in d:
                        if pd.Timestamp(sub) in self._holdout_index:
                            raise RuntimeError(
                                f"Holdout leakage detected: result {result.get('params', {})} used holdout date {sub}"
                            )
