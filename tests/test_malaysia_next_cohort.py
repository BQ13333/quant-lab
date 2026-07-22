from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.research.malaysia_next_cohort import (
    _opening_gap_portfolio,
    calendar_contrast_series,
    sampled_top_fraction_weights,
    turn_of_month_session_mask,
)


def test_sampled_weights_are_lagged_one_session_after_signal_date():
    index = pd.bdate_range("2026-01-05", periods=10)
    columns = [f"T{number:02d}" for number in range(10)]
    score = pd.DataFrame(
        np.tile(np.arange(10, dtype=float), (len(index), 1)), index=index, columns=columns
    )
    weights, diagnostics = sampled_top_fraction_weights(
        score, frequency="weekly", minimum_selected=2
    )
    first_friday = index[4]
    assert float(weights.loc[first_friday].abs().sum()) == 0.0
    assert float(weights.loc[index[5]].abs().sum()) == 1.0
    assert set(weights.loc[index[5]].loc[lambda row: row > 0.0].index) == {"T08", "T09"}
    assert diagnostics.signal_dates == 2


def test_turn_of_month_mask_is_exactly_first_three_and_last_session():
    sessions = pd.bdate_range("2026-01-01", "2026-02-27")
    mask = turn_of_month_session_mask(sessions, sessions)
    expected = {
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-05"),
        pd.Timestamp("2026-01-30"),
        pd.Timestamp("2026-02-02"),
        pd.Timestamp("2026-02-03"),
        pd.Timestamp("2026-02-04"),
        pd.Timestamp("2026-02-27"),
    }
    assert set(mask.index[mask]) == expected


def test_turn_of_month_mask_does_not_mislabel_partial_boundary_months():
    sessions = pd.bdate_range("2026-01-15", "2026-02-10")
    mask = turn_of_month_session_mask(sessions, sessions)
    assert pd.Timestamp("2026-01-15") not in set(mask.index[mask])
    assert pd.Timestamp("2026-01-30") in set(mask.index[mask])
    assert pd.Timestamp("2026-02-02") in set(mask.index[mask])
    assert pd.Timestamp("2026-02-10") not in set(mask.index[mask])


def test_opening_gap_charges_entry_and_exit_and_selects_most_negative_names():
    index = pd.DatetimeIndex(["2026-01-05"])
    columns = [f"T{number:02d}" for number in range(10)]
    gap = pd.DataFrame([np.linspace(-0.10, 0.08, 10)], index=index, columns=columns)
    intraday = pd.DataFrame(0.02, index=index, columns=columns)
    valid = pd.DataFrame(True, index=index, columns=columns)
    result = _opening_gap_portfolio(
        gap,
        intraday,
        valid,
        base_cost_bps_one_way=45.0,
        stress_cost_bps_one_way=65.0,
        minimum_selected=2,
    )
    weights, turnover = result[3], result[4]
    assert set(weights.loc[index[0]].loc[lambda row: row > 0.0].index) == {"T00", "T01"}
    assert float(turnover.iloc[0]) == 2.0
    assert np.isclose(float(result[1].iloc[0]), 0.02 - 2.0 * 45.0 / 10_000.0)
    assert np.isclose(float(result[2].iloc[0]), 0.02 - 2.0 * 65.0 / 10_000.0)


def test_calendar_contrast_mean_equals_inside_minus_outside_mean():
    index = pd.bdate_range("2026-01-01", periods=40)
    inside = pd.Series([True] * 20 + [False] * 20, index=index)
    returns = pd.Series([0.01] * 20 + [0.002] * 20, index=index)
    contrast = calendar_contrast_series(returns, inside)
    assert np.isclose(float(contrast.mean()), 0.008)
