from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_lab.agents.institutional_alpha_factory import (
    apply_institutional_promotion_gate,
    build_mutation_queue,
    load_accepted_data_quality_manifest,
    materialize_accepted_return_l2_proxy,
    prefix_candidate_namespace,
)


def test_load_accepted_data_quality_manifest_requires_5_of_5(tmp_path):
    root = tmp_path / "dq"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "accepted": False,
                "overall_score": 4.8,
                "max_score": 5.0,
                "critical_blockers": ["security_master"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="blocked"):
        load_accepted_data_quality_manifest(root)

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "accepted": True,
                "overall_score": 5.0,
                "max_score": 5.0,
                "critical_blockers": [],
                "paths": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = load_accepted_data_quality_manifest(root)
    assert manifest["accepted"] is True


def test_apply_institutional_promotion_gate_requires_holdout_and_overfit_pass():
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "winner",
                "net_sharpe": 3.2,
                "deflated_sharpe": 0.96,
                "net_ir": 1.1,
                "score": 1.1,
                "holdout_net_sharpe": 1.4,
                "passed_all": True,
                "constraint_max_drawdown_passed": True,
            },
            {
                "candidate_id": "weak_holdout",
                "net_sharpe": 3.4,
                "deflated_sharpe": 0.97,
                "net_ir": 1.2,
                "score": 1.2,
                "holdout_net_sharpe": 0.2,
                "passed_all": True,
                "constraint_max_drawdown_passed": True,
            },
            {
                "candidate_id": "below_sharpe_three",
                "net_sharpe": 2.9,
                "deflated_sharpe": 0.99,
                "net_ir": 1.0,
                "score": 1.0,
                "holdout_net_sharpe": 2.0,
                "passed_all": True,
                "constraint_max_drawdown_passed": True,
            },
        ]
    )

    ledger = apply_institutional_promotion_gate(
        candidates,
        overfit_summary={"pbo_even_odd_proxy": 0.10, "warnings": [], "effective_independent_trials": 5.0},
        min_net_sharpe=3.0,
        min_deflated_sharpe=0.95,
        min_holdout_net_sharpe=1.0,
        max_pbo=0.30,
    )
    by_id = ledger.set_index("candidate_id")

    assert bool(by_id.loc["winner", "institutional_gate_pass"]) is True
    assert by_id.loc["winner", "production_state"] == "paper_trade_candidate_pending_human_approval"
    assert bool(by_id.loc["weak_holdout", "institutional_gate_pass"]) is False
    assert by_id.loc["weak_holdout", "promotion_state"] == "killed_by_holdout_gate"
    assert bool(by_id.loc["below_sharpe_three", "institutional_gate_pass"]) is False

    failed = apply_institutional_promotion_gate(
        candidates.iloc[[0]],
        overfit_summary={"pbo_even_odd_proxy": 0.80, "warnings": ["high_even_odd_pbo_proxy"]},
        min_net_sharpe=3.0,
        min_deflated_sharpe=0.95,
        min_holdout_net_sharpe=1.0,
        max_pbo=0.30,
    )
    assert bool(failed.loc[0, "institutional_gate_pass"]) is False
    assert failed.loc[0, "promotion_state"] == "killed_by_overfit_audit"


def test_prefix_candidate_namespace_keeps_returns_aligned(tmp_path):
    ledger = pd.DataFrame({"candidate_id": ["a", "b"], "net_sharpe": [1.0, 2.0]})
    returns = pd.DataFrame(
        {"a": [0.01, 0.02], "b": [0.0, -0.01]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )

    out, out_returns = prefix_candidate_namespace(ledger, returns, "pit_source_ensemble", 2, tmp_path)

    assert out["candidate_id"].tolist() == [
        "pit_source_ensemble_iter02::a",
        "pit_source_ensemble_iter02::b",
    ]
    assert out["raw_candidate_id"].tolist() == ["a", "b"]
    assert out_returns.columns.tolist() == [
        "pit_source_ensemble_iter02::a",
        "pit_source_ensemble_iter02::b",
    ]


def test_materialize_accepted_return_l2_proxy_replaces_price_authority(tmp_path):
    l2 = tmp_path / "l2"
    l2.mkdir()
    raw = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=2).tolist() * 2,
            "ticker": ["AAA.KL", "AAA.KL", "BBB.KL", "BBB.KL"],
            "sid": ["a", "a", "b", "b"],
            "local_symbol": ["AAA", "AAA", "BBB", "BBB"],
            "name": ["A", "A", "B", "B"],
            "country": ["Malaysia", "Malaysia", "Malaysia", "Malaysia"],
            "exchange": ["XKLS", "XKLS", "XKLS", "XKLS"],
            "currency": ["MYR", "MYR", "MYR", "MYR"],
            "adj_close": [10.0, 11.0, 20.0, 19.0],
        }
    )
    raw_path = l2 / "daily_features.parquet"
    raw.to_parquet(raw_path, index=False)
    (l2 / "eligible_universe_tiers.csv").write_text(
        "ticker,country,u2_backtest_eligible\nAAA.KL,Malaysia,True\nBBB.KL,Malaysia,True\n",
        encoding="utf-8",
    )
    (l2 / "manifest.json").write_text(
        json.dumps({"long_ohlcv_root": str(raw_path), "date_min": "2024-01-01", "date_max": "2024-01-02"}),
        encoding="utf-8",
    )

    adjusted_root = tmp_path / "adjusted"
    adjusted_root.mkdir()
    adjusted = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=2).tolist() * 2,
            "ticker": ["AAA.KL", "AAA.KL", "BBB.KL", "BBB.KL"],
            "close": [10.0, 11.0, 20.0, 19.0],
            "adj_close_total_return_net": [100.0, 101.0, 200.0, 198.0],
        }
    )
    adjusted_path = adjusted_root / "daily_adjusted_returns.parquet"
    adjusted.to_parquet(adjusted_path, index=False)
    (adjusted_root / "manifest.json").write_text(
        json.dumps(
            {
                "production_adjusted_return_ready": True,
                "artifacts": {"daily_adjusted_returns": str(adjusted_path)},
            }
        ),
        encoding="utf-8",
    )

    proxy = materialize_accepted_return_l2_proxy(l2, adjusted_root, tmp_path / "proxy")
    manifest = json.loads((proxy / "manifest.json").read_text(encoding="utf-8"))
    proxied = pd.read_parquet(manifest["long_ohlcv_root"])

    assert manifest["price_authority"] == "accepted_adjusted_return_reconstruction"
    assert proxied["adj_close"].tolist() == [100.0, 101.0, 200.0, 198.0]
    assert set(["country", "exchange", "currency"]).issubset(proxied.columns)


def test_build_mutation_queue_detects_portfolio_and_overfit_workstreams():
    ledger = pd.DataFrame(
        [
            {
                "candidate_id": "best",
                "net_sharpe": 2.4,
                "holdout_net_sharpe": 3.1,
                "institutional_gate_pass": False,
                "constraint_deflated_sharpe_passed": False,
                "constraint_max_country_weight_passed": False,
            },
            {
                "candidate_id": "second",
                "net_sharpe": 1.0,
                "holdout_net_sharpe": 0.5,
                "institutional_gate_pass": False,
                "constraint_deflated_sharpe_passed": False,
                "constraint_max_country_weight_passed": False,
            },
        ]
    )

    queue = build_mutation_queue(
        ledger,
        overfit_summary={"effective_independent_trials": 1.5},
    )

    assert "portfolio_pm" in set(queue["owner_agent"])
    assert "olympiad_math_validator" in set(queue["owner_agent"])
    assert "ml_ai_researcher" in set(queue["owner_agent"])
    assert "risk_manager" in set(queue["owner_agent"])
