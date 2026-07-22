from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.data.global_universe import global_universe_metadata
from quant_lab.research import ResearchLoopConfig, run_global_research_loop_from_prices


def _synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    tickers = ["ACWI", "SPY", "EFA", "EEM", "EWJ", "EWU", "EWZ", "INDA", "EWS"]
    dates = pd.date_range("2021-01-01", periods=320, freq="B")
    returns = rng.normal(0.00035, 0.012, size=(len(dates), len(tickers)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + returns, axis=0), index=dates, columns=tickers)


def main() -> None:
    prices = _synthetic_prices()
    metadata = global_universe_metadata()
    metadata = metadata[metadata["ticker"].isin(prices.columns)].reset_index(drop=True)

    result = run_global_research_loop_from_prices(
        prices=prices,
        metadata=metadata,
        artifact_root="artifacts/research_loops",
        vault_root="vault",
        config=ResearchLoopConfig(run_id="LOOP-SYNTHETIC-DEMO", signal_profile="lite"),
        cost_bps_grid=(0.0, 5.0),
        train_size=100,
        test_size=40,
        embargo=2,
        min_rows=250,
    )

    print("Artifact dir:", result.artifact_dir)
    print("Manifest:", result.manifest_path)
    print("Loop note:", result.note_path)
    print(result.promotion_decisions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
