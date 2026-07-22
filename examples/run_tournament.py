from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quant_lab.models.baselines import MeanReversionModel, MomentumModel
from quant_lab.research.tournament import ModelTournament


def make_synthetic_prices(rows: int = 756, assets: int = 12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    columns = [f"asset_{i:02d}" for i in range(assets)]

    market = rng.normal(0.0002, 0.008, size=(rows, 1))
    idiosyncratic = rng.normal(0.0, 0.015, size=(rows, assets))
    trend = np.linspace(-0.0002, 0.0005, assets)
    returns = market + idiosyncratic + trend

    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=columns)


def main() -> None:
    prices = make_synthetic_prices()
    tournament = ModelTournament(
        models=[
            MomentumModel(name="mom_21", lookback=21),
            MomentumModel(name="mom_63", lookback=63),
            MeanReversionModel(name="rev_5", lookback=5),
        ],
        target_volatility=0.12,
        cost_bps=5.0,
    )
    leaderboard = tournament.run(prices, train_size=252, test_size=63, embargo=5)
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
