from __future__ import annotations

import hashlib
import json

import pandas as pd

from quant_lab.data.japan_g3_sector_gap_executor import execute_japan_g3_sector_gap_plan
from quant_lab.data.vendor_adapters import RefinitivRetryableProviderError


def _inputs(tmp_path, plan_rows):
    gap = tmp_path / "gap"
    baseline = tmp_path / "baseline"
    g1 = tmp_path / "g1"
    g2 = tmp_path / "g2"
    for directory in (gap, baseline, g1, g2):
        directory.mkdir()
    plan = pd.DataFrame(plan_rows)
    plan.to_csv(gap / "sector_gap_plan.csv", index=False)
    missing = sum(int(row["ric_count"]) for row in plan_rows)
    (gap / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete_plan",
                "selected_rics": 1 + missing,
                "existing_profile_rics": 1,
                "missing_rics": missing,
                "plan_rows": len(plan),
                "denominator_reconciles": True,
            }
        ), encoding="utf-8"
    )
    pd.DataFrame(
        {"ric": ["AAA.T"], "fundamental_route_class": ["general"]}
    ).to_csv(baseline / "sector_routing_profile.csv", index=False)
    (g1 / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass", "scorecard_eligible": True, "evidence_scope": "full",
                "independent_audit": True, "validator_type": "independent_process",
                "source_checksums_verified": True, "critical_blocker_count": 0,
            }
        ), encoding="utf-8"
    )
    (g2 / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass", "admission_passed": True, "g2_complete": True,
                "scorecard_eligible": True, "source_checksums_verified": True,
                "critical_blocker_count": 0, "check_count": 72, "passed_check_count": 72,
            }
        ), encoding="utf-8"
    )
    digest = hashlib.sha256((gap / "sector_gap_plan.csv").read_bytes()).hexdigest()
    return gap, baseline, g1, g2, digest


def test_sector_gap_executor_completes_exact_denominator(tmp_path):
    rows = [{"plan_row_id": "gap-1", "ric_count": 2, "ric_list": "BBB.T,CCC.T"}]
    gap, baseline, g1, g2, digest = _inputs(tmp_path, rows)

    class Client:
        def get_data(self, universe, fields):
            return pd.DataFrame(
                {
                    "Instrument": list(universe),
                    "TRBC Business Sector Code": ["5010"] * len(universe),
                }
            )

    result = execute_japan_g3_sector_gap_plan(
        Client(), gap_plan_dir=gap, expected_plan_sha256=digest,
        baseline_sector_dir=baseline, g1_validation_dir=g1, g2_admission_dir=g2,
        artifact_root=tmp_path / "out", run_id="RUN", max_plan_rows=1,
    )
    assert result.manifest["status"] == "complete_current_routing"
    assert result.manifest["profile_rics"] == 3
    assert result.manifest["coverage"] == 1.0


def test_sector_gap_executor_halts_on_first_retryable_failure(tmp_path):
    rows = [
        {"plan_row_id": "gap-1", "ric_count": 1, "ric_list": "BBB.T"},
        {"plan_row_id": "gap-2", "ric_count": 1, "ric_list": "CCC.T"},
    ]
    gap, baseline, g1, g2, digest = _inputs(tmp_path, rows)

    class Client:
        def __init__(self):
            self.calls = 0

        def get_data(self, universe, fields):
            self.calls += 1
            raise RefinitivRetryableProviderError("429 Too many requests")

    client = Client()
    result = execute_japan_g3_sector_gap_plan(
        client, gap_plan_dir=gap, expected_plan_sha256=digest,
        baseline_sector_dir=baseline, g1_validation_dir=g1, g2_admission_dir=g2,
        artifact_root=tmp_path / "out", run_id="RUN", max_plan_rows=2,
    )
    assert client.calls == 1
    assert result.row_manifest["status"].tolist() == ["failed_retryable_provider"]
    assert result.manifest["halted_early"] is True
