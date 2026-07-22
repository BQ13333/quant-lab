from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.research.malaysia import run_malaysia_research_from_asean_artifact


def test_run_malaysia_research_from_asean_artifact_smoke(tmp_path):
    source = tmp_path / "asean"
    source.mkdir()
    tickers = [f"{1000 + i}.KL" for i in range(6)]
    dates = pd.bdate_range("2020-01-01", periods=220)
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    prices = 100 * (1 + returns).cumprod()
    prices["EWM"] = 100 * (1 + returns.mean(axis=1)).cumprod()
    prices.to_parquet(source / "daily_usd_prices.parquet")

    metadata = pd.DataFrame(
        [{"ticker": ticker, "name": ticker, "country": "Malaysia", "currency": "MYR", "exchange": "Bursa Malaysia", "tier": "stock"} for ticker in tickers]
        + [{"ticker": "EWM", "name": "iShares MSCI Malaysia ETF", "country": "Malaysia", "currency": "USD", "exchange": "NYSE Arca", "tier": "regional_proxy"}]
    )
    metadata.to_csv(source / "universe.csv", index=False)

    result = run_malaysia_research_from_asean_artifact(
        asean_artifact_dir=source,
        artifact_root=tmp_path / "malaysia",
        vault_root=tmp_path / "vault",
        run_id="MALAYSIA-TEST",
        cost_bps_grid=(0.0,),
        baseline_cost_bps=0.0,
        train_size=80,
        test_size=20,
        embargo=2,
        min_rows=200,
    )

    assert result.artifact_dir.exists()
    assert result.eligible_prices.shape == (220, 6)
    assert result.artifact_dir.joinpath("leaderboard.csv").exists()
    assert result.note_paths
