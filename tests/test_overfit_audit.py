from __future__ import annotations

import numpy as np
import pandas as pd

from quant_lab.research.overfit_audit import build_overfit_audit


def test_build_overfit_audit_reports_core_warning_light_metrics():
    dates = pd.bdate_range("2026-01-01", periods=80)
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(len(dates), 4)),
        index=dates,
        columns=[f"c{i}" for i in range(4)],
    )
    scoreboard = pd.DataFrame(
        {
            "candidate_id": returns.columns,
            "model_name": [f"m{i}" for i in range(4)],
            "family": ["momentum", "momentum", "low_volatility", "alpha_combo"],
            "meta_source_strategy_id": ["s1", "s1", "s2", "s3"],
            "net_ir": [1.0, 0.5, 0.2, -0.1],
            "net_sharpe": [1.2, 0.7, 0.4, -0.2],
            "deflated_sharpe": [0.6, 0.3, 0.1, 0.0],
            "score": [1.0, 0.5, 0.2, -0.1],
            "passed_all": [True, False, False, False],
            "constraint_deflated_sharpe_passed": [True, False, False, False],
        }
    )

    audit = build_overfit_audit(scoreboard, returns)

    assert audit.summary["candidate_count"] == 4
    assert audit.summary["feasible_candidates"] == 1
    assert "pbo_even_odd_proxy" in audit.summary
    assert not audit.top_family_summary.empty
    assert not audit.correlation_summary.empty
