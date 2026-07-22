from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.research.malaysia_v2_experiment import (
    MalaysiaV2PairedEngine,
    hysteresis_unit_weights,
    paired_moving_block_bootstrap,
    return_series_hash,
)
from quant_lab.research.v2_overlay_protocol import V2_DRAFT02_PROTOCOL_ID


def test_hysteresis_is_deterministic_retains_incumbents_and_respects_capacity():
    index = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    columns = [f"T{item}.KL" for item in range(20)]
    scores = pd.DataFrame(
        [
            np.arange(20, dtype=float),
            np.arange(20, dtype=float)[::-1],
            np.ones(20),
        ],
        index=index,
        columns=columns,
    )
    first, counts, diagnostics = hysteresis_unit_weights(
        scores, protocol_id=V2_DRAFT02_PROTOCOL_ID
    )
    second, second_counts, second_diagnostics = hysteresis_unit_weights(
        scores, protocol_id=V2_DRAFT02_PROTOCOL_ID
    )
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_series_equal(counts, second_counts)
    assert diagnostics == second_diagnostics
    assert counts.tolist() == [4, 4, 4]
    assert (first.sum(axis=1) == 1.0).all()
    assert not diagnostics.insufficient_diversification_dates


def test_hysteresis_fails_closed_when_valid_universe_cannot_supply_four_names():
    scores = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0]],
        index=pd.to_datetime(["2025-01-31"]),
        columns=[f"T{item}.KL" for item in range(5)],
    )
    weights, counts, diagnostics = hysteresis_unit_weights(
        scores, protocol_id=V2_DRAFT02_PROTOCOL_ID
    )
    assert counts.iloc[0] == 1
    assert weights.iloc[0].sum() == 0.0
    assert diagnostics.insufficient_diversification_dates == (
        "2025-01-31T00:00:00",
    )


def test_paired_bootstrap_is_deterministic_and_uses_frozen_parameters():
    rng = np.random.default_rng(31)
    index = pd.bdate_range("2022-01-03", periods=504)
    control = pd.Series(rng.normal(0.0002, 0.01, len(index)), index=index)
    challenger = control * 0.90 + 0.0001
    first = paired_moving_block_bootstrap(control, challenger)
    second = paired_moving_block_bootstrap(control, challenger)
    assert first == second
    assert first.replications == 10_000
    assert first.finite_margin_draws == 10_000
    assert 0.0 <= first.noninferiority_probability <= 1.0
    with pytest.raises(ValueError, match="frozen"):
        paired_moving_block_bootstrap(control, challenger, block_length_sessions=21)


def test_return_series_hash_is_identity_free_and_value_sensitive():
    index = pd.bdate_range("2025-01-02", periods=20)
    first = pd.Series(0.001, index=index, name="first")
    second = first.rename("second")
    assert return_series_hash(first) == return_series_hash(second)
    second.iloc[-1] = 0.002
    assert return_series_hash(first) != return_series_hash(second)


def test_challenger_exposure_is_lagged_and_same_day_return_cannot_change_it():
    index = pd.bdate_range("2023-01-02", periods=180)
    columns = [f"T{item:02d}.KL" for item in range(20)]
    rng = np.random.default_rng(7)
    score = pd.DataFrame(rng.normal(size=(len(index), len(columns))), index=index, columns=columns)
    returns = pd.DataFrame(rng.normal(0.0, 0.01, size=score.shape), index=index, columns=columns)
    engine = object.__new__(MalaysiaV2PairedEngine)
    engine.protocol = {"protocol_id": V2_DRAFT02_PROTOCOL_ID}
    baseline = engine._challenger_portfolio(score, returns)
    changed = returns.copy()
    changed.iloc[-1] = 1.0
    perturbed = engine._challenger_portfolio(score, changed)
    assert baseline[5].iloc[-1] == perturbed[5].iloc[-1]
    assert baseline[0].iloc[-1] != perturbed[0].iloc[-1]
