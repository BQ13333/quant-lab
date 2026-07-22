from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g3_ordinary_executor import execute_japan_g3_ordinary_plan
from quant_lab.data.vendor_adapters import RefinitivRetryableProviderError


class FakeClient:
    def __init__(self):
        self.calls = 0

    def get_data(self, universe, fields, parameters):
        self.calls += 1
        return pd.DataFrame(
            {
                "Instrument": list(universe),
                "Revenue": [100.0] * len(universe),
                "Calc Date": ["2025-05-08"] * len(universe),
                "Period End Date": ["2025-03-31"] * len(universe),
                "Financial Period Absolute": ["FY2025"] * len(universe),
            }
        )


def _plan(path: Path) -> str:
    pd.DataFrame(
        [
            {
                "plan_row_id": "fund-general-0001-01-fy",
                "dataset_id": "fundamentals_pit",
                "route_class": "general",
                "route": "get_data_parameterized",
                "frequency": "FY",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "batch_number": 1,
                "window_number": 1,
                "ric_count": 1,
                "ric_list": "7203.T",
                "source_fields": "TR.Revenue",
                "status": "ready_after_vendor_health",
            }
        ]
    ).to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _g2_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "g2"
    directory.mkdir(exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "admission_passed": True,
                "g2_complete": True,
                "scorecard_eligible": True,
                "source_checksums_verified": True,
                "window_coverage_complete": True,
                "raw_compacted_parity": True,
                "official_corporate_actions_reconciled": True,
                "pit_liquidity_complete": True,
                "critical_blocker_count": 0,
                "check_count": 72,
                "passed_check_count": 72,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_executor_refuses_before_independent_g1(tmp_path: Path) -> None:
    plan = tmp_path / "fundamentals_plan.csv"
    digest = _plan(plan)
    g1 = tmp_path / "g1"
    g1.mkdir()
    (g1 / "manifest.json").write_text(json.dumps({"status": "failed_validation"}), encoding="utf-8")
    client = FakeClient()
    with pytest.raises(PermissionError):
        execute_japan_g3_ordinary_plan(
            client, plan, digest, g1, tmp_path / "missing-g2", tmp_path / "out", "RUN"
        )
    assert client.calls == 0


def test_executor_checks_hash_and_writes_atomic_terminal_evidence(tmp_path: Path) -> None:
    plan = tmp_path / "fundamentals_plan.csv"
    digest = _plan(plan)
    g1 = tmp_path / "g1"
    g1.mkdir()
    (g1 / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "scorecard_eligible": True,
                "evidence_scope": "full",
                "independent_validation": True,
                "source_checksums_verified": True,
                "critical_blocker_count": 0,
            }
        ),
        encoding="utf-8",
    )
    client = FakeClient()
    g2 = _g2_dir(tmp_path)
    result = execute_japan_g3_ordinary_plan(
        client, plan, digest, g1, g2, tmp_path / "out", "RUN", max_plan_rows=1, sleep_seconds=0
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert client.calls == 1
    assert manifest["status"] == "complete"
    assert manifest["scorecard_eligible"] is True
    assert manifest["known_at_coverage"] == 1.0
    assert manifest["pit_timestamp_violations"] == 0
    assert not (result.artifact_dir / ".writer.lock").exists()


def test_executor_refuses_when_strict_g2_is_not_admitted(tmp_path: Path) -> None:
    plan = tmp_path / "fundamentals_plan.csv"
    digest = _plan(plan)
    g1 = tmp_path / "g1"
    g1.mkdir()
    (g1 / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "scorecard_eligible": True,
                "evidence_scope": "full",
                "independent_audit": True,
                "validator_type": "independent_process",
                "source_checksums_verified": True,
                "critical_blocker_count": 0,
            }
        ),
        encoding="utf-8",
    )
    g2 = tmp_path / "g2"
    g2.mkdir()
    (g2 / "manifest.json").write_text(
        json.dumps({"status": "failed_admission", "critical_blocker_count": 1}),
        encoding="utf-8",
    )
    client = FakeClient()
    with pytest.raises(PermissionError, match="strict Japan G2"):
        execute_japan_g3_ordinary_plan(
            client, plan, digest, g1, g2, tmp_path / "out", "RUN"
        )
    assert client.calls == 0


def test_executor_rejects_plan_hash_mismatch(tmp_path: Path) -> None:
    plan = tmp_path / "fundamentals_plan.csv"
    _plan(plan)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        execute_japan_g3_ordinary_plan(
            FakeClient(), plan, "0" * 64, tmp_path / "g1", tmp_path / "g2", tmp_path / "out", "RUN"
        )


def test_executor_halts_after_first_retryable_provider_failure(tmp_path: Path) -> None:
    plan = tmp_path / "fundamentals_plan.csv"
    rows = []
    for index, ric in enumerate(("7203.T", "6758.T"), start=1):
        rows.append(
            {
                "plan_row_id": f"fund-general-{index:04d}-01-fy",
                "dataset_id": "fundamentals_pit",
                "route_class": "general",
                "frequency": "FY",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "ric_list": ric,
                "source_fields": "TR.Revenue",
            }
        )
    pd.DataFrame(rows).to_csv(plan, index=False)
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    g1 = tmp_path / "g1"
    g1.mkdir()
    (g1 / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "scorecard_eligible": True,
                "evidence_scope": "full",
                "independent_audit": True,
                "validator_type": "independent_process",
                "source_checksums_verified": True,
                "critical_blocker_count": 0,
            }
        ),
        encoding="utf-8",
    )

    class RetryableClient:
        def __init__(self):
            self.calls = 0

        def get_data(self, **kwargs):
            self.calls += 1
            raise RefinitivRetryableProviderError("429 Too many requests")

    client = RetryableClient()
    g2 = _g2_dir(tmp_path)
    result = execute_japan_g3_ordinary_plan(
        client, plan, digest, g1, g2, tmp_path / "out", "RUN", max_plan_rows=2, sleep_seconds=0
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert client.calls == 1
    assert result.row_manifest["status"].tolist() == ["failed_retryable_provider"]
    assert manifest["halted_early"] is True
    assert manifest["failed_rows"] == 1
