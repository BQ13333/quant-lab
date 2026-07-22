from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.data.global_universe import global_universe_metadata
from quant_lab.research.global_markets import run_global_research_from_prices


def _synthetic_global_prices(n_days: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    tickers = ["ACWI", "SPY", "EFA", "EEM", "EWJ", "EWU", "EWZ", "INDA", "EWS"]
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    returns = rng.normal(0.0004, 0.012, size=(n_days, len(tickers)))
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=tickers)
    return prices


def test_run_global_research_from_prices_writes_artifacts(tmp_path):
    prices = _synthetic_global_prices()
    metadata = global_universe_metadata()
    metadata = metadata[metadata["ticker"].isin(prices.columns)].reset_index(drop=True)

    result = run_global_research_from_prices(
        prices=prices,
        metadata=metadata,
        artifact_dir=tmp_path / "global",
        vault_root=tmp_path / "vault",
        run_id="GLOBAL-TEST",
        cost_bps_grid=(0.0, 5.0),
        baseline_cost_bps=5.0,
        train_size=80,
        test_size=40,
        embargo=2,
        min_rows=200,
    )

    assert result.artifact_dir.joinpath("leaderboard.csv").exists()
    assert result.artifact_dir.joinpath("cost_sensitivity.csv").exists()
    assert result.artifact_dir.joinpath("eligible_usd_prices.parquet").exists()
    assert not result.leaderboard.empty
    assert not result.cost_sensitivity.empty
    assert set(result.cost_sensitivity["cost_bps"]) == {0.0, 5.0}
    assert result.note_paths
