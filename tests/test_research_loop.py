from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant_lab.data.global_universe import global_universe_metadata
from quant_lab.research import (
    ResearchLoopConfig,
    evaluate_promotion_decisions,
    run_global_research_loop_from_prices,
)


def _prices(n_days: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    tickers = ["ACWI", "SPY", "EFA", "EEM", "EWJ", "EWU", "EWZ", "INDA", "EWS"]
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    returns = rng.normal(0.0004, 0.012, size=(n_days, len(tickers)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=tickers)


def test_evaluate_promotion_decisions_rejects_benchmark_rows():
    leaderboard = pd.DataFrame(
        [
            {"model_name": "m1", "weighting": "long_only", "sharpe": 1.0, "deflated_sharpe": 0.9, "annual_excess_return": -0.01, "information_ratio": -0.1, "max_drawdown": -0.1, "turnover": 0.1},
            {"model_name": "bench", "weighting": "benchmark", "sharpe": 0.8, "deflated_sharpe": 0.0, "annual_excess_return": 0.0, "information_ratio": 0.0, "max_drawdown": -0.2, "turnover": 0.0},
        ]
    )

    decisions = evaluate_promotion_decisions(leaderboard)

    assert decisions["model_name"].tolist() == ["m1"]
    assert decisions["promoted"].tolist() == [False]
    assert "annual excess return" in decisions["reasons"].iloc[0]


def test_run_global_research_loop_from_prices_writes_persistent_artifacts(tmp_path):
    prices = _prices()
    metadata = global_universe_metadata()
    metadata = metadata[metadata["ticker"].isin(prices.columns)].reset_index(drop=True)

    result = run_global_research_loop_from_prices(
        prices=prices,
        metadata=metadata,
        artifact_root=tmp_path / "loops",
        vault_root=tmp_path / "vault",
        config=ResearchLoopConfig(run_id="LOOP-TEST", signal_profile="lite"),
        cost_bps_grid=(0.0, 5.0),
        train_size=80,
        test_size=40,
        embargo=2,
        min_rows=220,
    )

    assert result.artifact_dir.joinpath("loop_manifest.json").exists()
    assert result.artifact_dir.joinpath("promotion_decisions.csv").exists()
    assert result.artifact_dir.joinpath("signal_manifest.csv").exists()
    assert result.note_path.exists()
    assert not result.promotion_decisions.empty

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["config"]["run_id"] == "LOOP-TEST"
    assert manifest["leaderboard_rows"] == len(result.research_result.leaderboard)
