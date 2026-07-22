from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.research.diagnostics import (
    generate_cost_sensitivity,
    generate_country_attribution,
    generate_diagnostics_summary,
    generate_fx_attribution,
    generate_leaderboard,
    generate_liquidity_report,
    generate_model_cards,
    generate_monthly_attribution,
    generate_turnover_report,
)


def _make_leaderboard(n_models: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_name": [f"model_{i}" for i in range(n_models)],
            "sharpe": [1.0 + i * 0.1 for i in range(n_models)],
            "sortino": [0.8 + i * 0.1 for i in range(n_models)],
            "calmar": [0.5 + i * 0.1 for i in range(n_models)],
            "max_drawdown": [-0.1 - i * 0.02 for i in range(n_models)],
            "annual_return": [0.1 + i * 0.01 for i in range(n_models)],
            "annual_volatility": [0.1 + i * 0.01 for i in range(n_models)],
            "turnover": [1.0 + i * 0.1 for i in range(n_models)],
            "deflated_sharpe": [0.9 - i * 0.05 for i in range(n_models)],
            "annual_excess_return": [0.02 + i * 0.01 for i in range(n_models)],
            "tracking_error": [0.05 + i * 0.01 for i in range(n_models)],
            "information_ratio": [0.4 + i * 0.1 for i in range(n_models)],
            "splits": [10] * n_models,
        }
    )


def _make_returns_weights(n: int = 100, n_assets: int = 3):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = pd.DataFrame(
        rng.normal(0.001, 0.01, (n, n_assets)),
        index=dates,
        columns=["A", "B", "C"],
    )
    weights = pd.DataFrame(
        np.ones((n, n_assets)) / n_assets, index=dates, columns=["A", "B", "C"]
    )
    return returns, weights


def test_generate_leaderboard_sorts_by_deflated_sharpe():
    lb = _make_leaderboard()
    result = generate_leaderboard(lb)
    assert not result.empty
    assert list(result.columns) == list(lb.columns)
    assert result.iloc[0]["deflated_sharpe"] >= result.iloc[1]["deflated_sharpe"]


def test_generate_leaderboard_save_roundtrip(tmp_path):
    lb = _make_leaderboard()
    result = generate_leaderboard(lb, save_dir=tmp_path)
    csv_path = tmp_path / "leaderboard.csv"
    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    pd.testing.assert_frame_equal(result, loaded)


def test_generate_leaderboard_empty():
    result = generate_leaderboard(pd.DataFrame())
    assert result.empty


def test_generate_diagnostics_summary_columns():
    lb = _make_leaderboard()
    benchmarks = pd.DataFrame(
        {
            "model_name": ["benchmark_ew"],
            "sharpe": [0.8],
            "sortino": [0.6],
            "calmar": [0.4],
            "max_drawdown": [-0.15],
            "annual_return": [0.08],
            "annual_volatility": [0.10],
            "turnover": [0.0],
            "deflated_sharpe": [0.5],
            "annual_excess_return": [0.0],
            "tracking_error": [0.0],
            "information_ratio": [0.0],
            "splits": [1],
        }
    )
    result = generate_diagnostics_summary(lb, benchmarks)
    assert "gate_passed" in result.columns
    assert "gate_reasons" in result.columns
    assert "promote_passed" in result.columns
    assert "promote_reasons" in result.columns
    assert len(result) == len(lb) + len(benchmarks)


def test_generate_diagnostics_summary_save(tmp_path):
    lb = _make_leaderboard()
    result = generate_diagnostics_summary(lb, save_dir=tmp_path)
    assert (tmp_path / "diagnostics_summary.csv").exists()


def test_generate_cost_sensitivity_columns():
    returns, weights = _make_returns_weights()
    result = generate_cost_sensitivity(returns["A"], weights, cost_grid=(0.0, 5.0, 10.0))
    assert not result.empty
    assert "cost_bps" in result.columns
    assert "net_sharpe" in result.columns
    assert "net_return" in result.columns
    assert "max_drawdown" in result.columns


def test_generate_cost_sensitivity_save(tmp_path):
    returns, weights = _make_returns_weights()
    result = generate_cost_sensitivity(returns["A"], weights, save_dir=tmp_path)
    assert (tmp_path / "cost_sensitivity.csv").exists()


def test_generate_country_attribution_columns():
    returns, weights = _make_returns_weights()
    group_map = pd.Series({"A": "USA", "B": "UK", "C": "USA"})
    result = generate_country_attribution(returns, weights, group_map)
    assert not result.empty
    assert "country" in result.columns
    assert "contribution" in result.columns
    assert "avg_weight" in result.columns
    assert "avg_return" in result.columns
    assert "selection_effect" in result.columns
    assert "allocation_effect" in result.columns
    assert set(result["country"]) == {"UK", "USA"}


def test_generate_country_attribution_save(tmp_path):
    returns, weights = _make_returns_weights()
    group_map = pd.Series({"A": "USA", "B": "UK", "C": "USA"})
    result = generate_country_attribution(returns, weights, group_map, save_dir=tmp_path)
    assert (tmp_path / "country_attribution.csv").exists()


def test_generate_monthly_attribution_columns():
    returns, weights = _make_returns_weights(n=100)
    result = generate_monthly_attribution(returns, weights, cost_bps=5.0)
    assert not result.empty
    assert "year" in result.columns
    assert "month" in result.columns
    assert "return" in result.columns
    assert "volatility" in result.columns
    assert "drawdown" in result.columns
    assert "turnover" in result.columns
    assert "cost" in result.columns


def test_generate_monthly_attribution_save(tmp_path):
    returns, weights = _make_returns_weights(n=100)
    result = generate_monthly_attribution(returns, weights, save_dir=tmp_path)
    assert (tmp_path / "monthly_attribution.csv").exists()


def test_generate_fx_attribution_columns():
    returns, weights = _make_returns_weights()
    fx_returns = pd.Series(
        np.random.default_rng(7).normal(0.0001, 0.005, len(returns)),
        index=returns.index,
    )
    result = generate_fx_attribution(returns, fx_returns, weights)
    assert not result.empty
    assert "asset" in result.columns
    assert "fx_contribution" in result.columns
    assert "local_contribution" in result.columns
    assert "total_contribution" in result.columns
    assert len(result) == returns.shape[1]


def test_generate_fx_attribution_save(tmp_path):
    returns, weights = _make_returns_weights()
    fx_returns = pd.Series(
        np.random.default_rng(7).normal(0.0001, 0.005, len(returns)),
        index=returns.index,
    )
    result = generate_fx_attribution(returns, fx_returns, weights, save_dir=tmp_path)
    assert (tmp_path / "fx_attribution.csv").exists()


def test_generate_turnover_report_columns():
    returns, weights = _make_returns_weights()
    metadata = pd.DataFrame(
        {"country": ["USA", "UK", "USA"]}, index=["A", "B", "C"]
    )
    result = generate_turnover_report(weights, metadata=metadata)
    assert not result.empty
    assert "category" in result.columns
    assert "subcategory" in result.columns
    assert "mean_turnover" in result.columns
    assert "max_turnover" in result.columns
    categories = set(result["category"])
    assert "overall" in categories
    assert "month" in categories
    assert "country" in categories
    assert "top_asset" in categories


def test_generate_turnover_report_save(tmp_path):
    returns, weights = _make_returns_weights()
    result = generate_turnover_report(weights, save_dir=tmp_path)
    assert (tmp_path / "turnover_report.csv").exists()


def test_generate_liquidity_report_columns():
    returns, weights = _make_returns_weights()
    volumes = pd.DataFrame(
        np.ones((len(weights), 3)) * 1e6,
        index=weights.index,
        columns=["A", "B", "C"],
    )
    result = generate_liquidity_report(weights, volumes, metadata=None)
    assert not result.empty
    assert "asset" in result.columns
    assert "adv" in result.columns
    assert "max_participation" in result.columns
    assert "avg_participation" in result.columns
    assert "capacity_score" in result.columns


def test_generate_liquidity_report_save(tmp_path):
    returns, weights = _make_returns_weights()
    volumes = pd.DataFrame(
        np.ones((len(weights), 3)) * 1e6,
        index=weights.index,
        columns=["A", "B", "C"],
    )
    result = generate_liquidity_report(weights, volumes, save_dir=tmp_path)
    assert (tmp_path / "liquidity_report.csv").exists()


def test_generate_model_cards_shape():
    lb = _make_leaderboard(n_models=2)
    returns_dict = {
        f"model_{i}": pd.Series(
            np.random.default_rng(i).normal(0.001, 0.01, 100)
        )
        for i in range(2)
    }
    weights_dict = {
        f"model_{i}": pd.DataFrame(np.ones((100, 3)) / 3, columns=["A", "B", "C"])
        for i in range(2)
    }
    cards = generate_model_cards(lb, returns_dict, weights_dict)
    assert len(cards) == 2
    for name, card in cards.items():
        assert not card.empty
        assert card.shape[0] == 1
        assert "gate_passed" in card.columns
        assert "promote_passed" in card.columns
        assert "cost_sharpe_0bps" in card.columns


def test_generate_model_cards_save(tmp_path):
    lb = _make_leaderboard(n_models=2)
    returns_dict = {
        f"model_{i}": pd.Series(
            np.random.default_rng(i).normal(0.001, 0.01, 100)
        )
        for i in range(2)
    }
    weights_dict = {
        f"model_{i}": pd.DataFrame(np.ones((100, 3)) / 3, columns=["A", "B", "C"])
        for i in range(2)
    }
    cards = generate_model_cards(lb, returns_dict, weights_dict, save_dir=tmp_path)
    assert (tmp_path / "model_cards").exists()
    for name in cards:
        assert (tmp_path / "model_cards" / f"{name}.csv").exists()
