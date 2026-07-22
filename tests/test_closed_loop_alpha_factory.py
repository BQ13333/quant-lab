from __future__ import annotations

import pandas as pd

from quant_lab.agents.closed_loop_alpha_factory import (
    apply_pm_sharpe_gate,
    closed_loop_allocator_configs,
)


def test_apply_pm_sharpe_gate_requires_sharpe_three_and_deflated_sharpe():
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "winner",
                "net_sharpe": 3.25,
                "deflated_sharpe": 0.97,
                "net_ir": 1.1,
                "passed_all": True,
                "constraint_max_drawdown_passed": True,
            },
            {
                "candidate_id": "soft_gate_only",
                "net_sharpe": 2.95,
                "deflated_sharpe": 0.99,
                "net_ir": 1.0,
                "passed_all": True,
                "constraint_max_drawdown_passed": True,
            },
            {
                "candidate_id": "failed_constraint",
                "net_sharpe": 3.50,
                "deflated_sharpe": 0.99,
                "net_ir": 1.0,
                "passed_all": False,
                "constraint_max_drawdown_passed": False,
            },
            {
                "candidate_id": "weak_dsr",
                "net_sharpe": 3.50,
                "deflated_sharpe": 0.80,
                "net_ir": 1.0,
                "passed_all": True,
                "constraint_max_drawdown_passed": True,
            },
        ]
    )

    ledger = apply_pm_sharpe_gate(candidates, min_net_sharpe=3.0, min_deflated_sharpe=0.95)
    by_id = ledger.set_index("candidate_id")

    assert bool(by_id.loc["winner", "pm_gate_pass"]) is True
    assert by_id.loc["winner", "promotion_state"] == "promotion_candidate_pending_holdout_and_human_approval"

    assert bool(by_id.loc["soft_gate_only", "pm_gate_pass"]) is False
    assert by_id.loc["soft_gate_only", "promotion_state"] == "research_survivor_below_sharpe3_goal"
    assert "net_sharpe_below_3" in by_id.loc["soft_gate_only", "kill_reason"]

    assert bool(by_id.loc["failed_constraint", "pm_gate_pass"]) is False
    assert by_id.loc["failed_constraint", "promotion_state"] == "killed_by_pm_gate"
    assert "max_drawdown" in by_id.loc["failed_constraint", "kill_reason"]

    assert bool(by_id.loc["weak_dsr", "pm_gate_pass"]) is False
    assert "deflated_sharpe_below_0.95" in by_id.loc["weak_dsr", "kill_reason"]


def test_closed_loop_allocator_configs_include_capped_daily_weekly_monthly_mix():
    configs = closed_loop_allocator_configs(max_asset_weight=0.08)

    assert any(config["weighting"] == "long_only_capped" for config in configs)
    assert any(config["weighting"] == "group_neutral_long_only_capped" for config in configs)
    assert any(config["weighting"] == "group_neutral_dollar_neutral" for config in configs)
    assert {"D", "W", "M"}.issubset({config["rebalance_frequency"] for config in configs})
    assert all(config.get("max_weight", 0.08) <= 0.08 for config in configs)
