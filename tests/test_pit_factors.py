from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.models.pit_factors import PITFactorModel, build_pit_factor_model_suite


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=80)
    return pd.DataFrame(
        {
            "AAA.KL": np.linspace(10.0, 12.0, len(dates)),
            "BBB.KL": np.linspace(20.0, 19.0, len(dates)),
            "CCC.KL": np.linspace(30.0, 32.0, len(dates)),
        },
        index=dates,
    )


def test_pit_factor_model_predicts_price_aligned_scores():
    prices = _prices()
    factor = pd.DataFrame(
        {
            "AAA.KL": np.linspace(1.0, 2.0, len(prices)),
            "BBB.KL": np.linspace(2.0, 1.0, len(prices)),
            "CCC.KL": np.linspace(1.5, 1.7, len(prices)),
        },
        index=prices.index,
    )
    model = PITFactorModel(
        name="pit_test",
        factor_matrices={"pit_fundamentals__book_to_price": factor},
        factor_weights={"pit_fundamentals__book_to_price": 1.0},
        feature_set="fundamentals",
    )

    signal = model.predict(prices, 20, 40)

    assert signal.scores.shape == (20, 3)
    assert set(signal.scores.columns) == set(prices.columns)
    assert signal.metadata["feature_set"] == "fundamentals"
    assert signal.metadata["readiness"] == "research_mode_leakage_qa_passed"


def test_build_pit_factor_model_suite_creates_family_and_composite_models():
    prices = _prices()
    matrices = {
        "pit_fundamentals__book_to_price": prices / prices,
        "pit_ibes__eps_yield": prices / 100.0,
        "pit_liquidity__log_market_cap": prices,
    }
    families = {
        "fundamentals": ("pit_fundamentals__book_to_price",),
        "ibes": ("pit_ibes__eps_yield",),
        "liquidity": ("pit_liquidity__log_market_cap",),
        "events": (),
    }

    models = build_pit_factor_model_suite(matrices, families, prefix="pitunit")
    names = {model.name for model in models}

    assert "pitunit_fundamentals" in names
    assert "pitunit_ibes" in names
    assert "pitunit_all_equal" in names
