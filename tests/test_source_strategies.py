from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.models.source_strategies import (
    MovingAverageCrossoverModel,
    build_source_strategy_model_suite,
    build_source_strategy_specs,
    source_strategy_manifest,
)


def test_source_strategy_specs_cover_implemented_and_data_blocked_ideas():
    specs = build_source_strategy_specs()
    statuses = {spec.implementation_status for spec in specs}
    ids = {spec.strategy_id for spec in specs}

    assert "implemented" in statuses
    assert "requires_pit_fundamentals" in statuses
    assert "kakushadze_price_momentum" in ids
    assert "kakushadze_earnings_momentum" in ids
    assert "worldquant_alpha_process" in ids


def test_source_strategy_model_suite_is_broad_and_source_backed():
    models = build_source_strategy_model_suite(prefix="test")
    names = [model.name for model in models]

    assert len(models) >= 45
    assert any("ewmac" in name for name in names)
    assert any("regtrend" in name for name in names)
    assert any("resid_mom" in name for name in names)
    assert any("alpha_combo" in name for name in names)
    assert any(hasattr(model, "source_spec") for model in models if "trend_lowvol" in model.name)


def test_source_strategy_manifest_is_tabular():
    manifest = source_strategy_manifest()

    assert not manifest.empty
    assert {"strategy_id", "source_file", "implementation_status"}.issubset(manifest.columns)


def test_moving_average_crossover_emits_source_metadata():
    dates = pd.bdate_range("2025-01-01", periods=120)
    rng = np.random.default_rng(123)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(len(dates), 5)),
        index=dates,
        columns=[f"S{i}" for i in range(5)],
    )
    prices = 100.0 * (1.0 + returns).cumprod()
    spec = next(spec for spec in build_source_strategy_specs() if spec.strategy_id == "kakushadze_moving_average")
    model = MovingAverageCrossoverModel(name="test_ma", fast=10, slow=40, source_spec=spec)

    signal = model.predict(prices, 60, 90)

    assert signal.scores.shape == (30, 5)
    assert np.isfinite(signal.scores.to_numpy()).all()
    assert signal.metadata["source_strategy_id"] == "kakushadze_moving_average"
    assert signal.metadata["family"] == "moving_average_crossover"
