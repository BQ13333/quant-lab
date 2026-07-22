from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_lab.data.pit_feature_selectors import (
    build_default_pit_factor_matrices,
    load_pit_feature_bundle,
    select_factor_columns,
)


def _panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=90)
    rows = []
    for ticker in ["AAA.KL", "BBB.KL"]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "requested_ric": ticker,
                    "decision_date": date,
                    "liquidity_market_cap_free_float__market_cap": 1000.0 + index,
                    "liquidity_market_cap_free_float__free_float": 60.0,
                    "fundamentals_pit__book_value": 2.0,
                    "ibes_estimates_revisions__eps_estimate": 0.2 + index * 0.001,
                    "ibes_estimates_revisions__revenue_estimate": 10.0 + index * 0.01,
                    "ibes_estimates_revisions__recommendation": 2.0,
                    "earnings_events_actuals__earnings_release": date - pd.Timedelta(days=5),
                }
            )
    return pd.DataFrame(rows)


def test_build_default_pit_factor_matrices_derives_expected_families():
    dates = pd.bdate_range("2024-01-01", periods=90)
    prices = pd.DataFrame(
        {
            "AAA.KL": np.linspace(10, 12, len(dates)),
            "BBB.KL": np.linspace(20, 21, len(dates)),
        },
        index=dates,
    )

    bundle = build_default_pit_factor_matrices(_panel(), prices, revision_window=21, event_window=30)

    assert "pit_fundamentals__book_to_price" in bundle.matrices
    assert "pit_ibes__eps_revision_63d" in bundle.matrices
    assert "pit_events__earnings_recency_30d" in bundle.matrices
    assert set(select_factor_columns(["fundamentals", "ibes"])).issuperset(
        {"pit_fundamentals__book_to_price", "pit_ibes__eps_yield"}
    )
    assert bundle.matrices["pit_events__earnings_recency_30d"].max().max() > 0


def test_load_pit_feature_bundle_requires_pass(tmp_path):
    artifact = tmp_path / "qa"
    artifact.mkdir()
    panel_path = artifact / "decision_lagged_features.parquet"
    _panel().to_parquet(panel_path, index=False)
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "status": "fail",
                "total_leakage_violations": 1,
                "artifacts": {"decision_lagged_features": str(panel_path)},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="did not pass"):
        load_pit_feature_bundle(artifact, require_pass=True)
