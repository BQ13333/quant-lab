from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_combined_monthly_membership_qa import (
    run_japan_combined_monthly_membership_qa,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(root: Path, name: str, artifact_type: str, family_name: str) -> Path:
    source = root / name
    normalized = source / "normalized.parquet"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "observation_month": ["2010-10", "2010-10"],
            "code": ["1001", "1002"],
            "known_at_utc": ["2010-11-20T15:00:00+00:00"] * 2,
            "currency": ["JPY", "JPY"],
            "pit_usable": [True, True],
            "source_sha256": ["a" * 64, "a" * 64],
            "issue_name_japanese": ["One", "Two"],
            "source_exchange": [family_name, family_name],
        }
    ).to_parquet(normalized, index=False)
    pd.DataFrame(
        {
            "request_id": ["request"],
            "observation_month": ["2010-10"],
        }
    ).to_csv(source / "request_plan.csv", index=False)
    pd.DataFrame(
        {
            "request_id": ["request"],
            "status": ["completed"],
            "normalized_path": [str(normalized)],
            "normalized_sha256": [_sha256(normalized)],
        }
    ).to_csv(source / "file_status.csv", index=False)
    pd.DataFrame({"request_id": ["request"], "quality_pass": [True]}).to_csv(
        source / "file_quality.csv", index=False
    )
    (source / "manifest.json").write_text(
        json.dumps({"status": "complete_candidate_source", "artifact_type": artifact_type}),
        encoding="utf-8",
    )
    return source


def test_combined_union_passes_integrated_month(tmp_path: Path) -> None:
    tse = _source(tmp_path, "tse", "official_jpx_monthly_quotation_source", "TSE")
    combined = _source(
        tmp_path,
        "combined",
        "official_jasdaq_then_ose_combined_monthly_cash_equity_membership_source",
        "OSE",
    )
    result = run_japan_combined_monthly_membership_qa(
        source_dirs=[tse, combined],
        run_id="QA",
        artifact_root=tmp_path / "out",
        start_month="2010-10",
        end_month="2010-10",
        min_monthly_codes=1,
        required_months=1,
    )
    assert result.manifest["status"] == "pass_candidate_source"
    assert result.manifest["membership_rows"] == 2
    assert pd.read_csv(result.monthly_coverage_path).loc[0, "cross_source_duplicate_codes"] == 2


def test_missing_integrated_source_fails_closed(tmp_path: Path) -> None:
    tse = _source(tmp_path, "tse", "official_jpx_monthly_quotation_source", "TSE")
    result = run_japan_combined_monthly_membership_qa(
        source_dirs=[tse],
        run_id="FAIL",
        artifact_root=tmp_path / "out",
        start_month="2010-10",
        end_month="2010-10",
        min_monthly_codes=1,
        required_months=1,
    )
    assert result.manifest["status"] == "in_progress"
    assert "OSE_COMBINED" in result.anomalies_path.read_text(encoding="utf-8")
