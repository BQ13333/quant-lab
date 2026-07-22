from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from quant_lab.data.jpx_monthly_membership_qa import run_jpx_monthly_membership_qa


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_monthly_membership_qa_checks_hashes_and_continuity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    plan_rows = []
    status_rows = []
    quality_rows = []
    for month, codes in (("2020-01", ["1301", "2301"]), ("2020-02", ["1301", "2301", "3301"])):
        request_id = month.replace("-", "") + "-st"
        path = source / f"{request_id}.parquet"
        pd.DataFrame(
            {
                "code": codes,
                "known_at_utc": ["2020-03-20T15:00:00Z"] * len(codes),
                "currency": ["JPY"] * len(codes),
                "pit_usable": [True] * len(codes),
                "source_sha256": ["a" * 64] * len(codes),
            }
        ).to_parquet(path, index=False)
        plan_rows.append({"request_id": request_id, "observation_month": month})
        status_rows.append(
            {
                "request_id": request_id,
                "status": "completed",
                "normalized_path": str(path),
                "normalized_sha256": _sha(path),
            }
        )
        quality_rows.append({"request_id": request_id, "quality_pass": True})
    pd.DataFrame(plan_rows).to_csv(source / "request_plan.csv", index=False)
    pd.DataFrame(status_rows).to_csv(source / "file_status.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(source / "file_quality.csv", index=False)

    result = run_jpx_monthly_membership_qa(
        source_dir=source,
        run_id="QA",
        artifact_root=tmp_path / "qa",
        max_monthly_churn_rate=0.6,
    )
    assert result.manifest["status"] == "pass_candidate_source"
    assert result.manifest["completed_months"] == 2
    assert result.manifest["anomalies"] == 0


def test_partial_latest_month_is_not_compared_for_churn(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    plan = pd.DataFrame(
        [
            {"request_id": "202001-st", "observation_month": "2020-01"},
            {"request_id": "202002-st1", "observation_month": "2020-02"},
            {"request_id": "202002-st2", "observation_month": "2020-02"},
        ]
    )
    plan.to_csv(source / "request_plan.csv", index=False)
    path = source / "202001.parquet"
    pd.DataFrame(
        {
            "code": ["1301", "2301"],
            "known_at_utc": ["2020-02-20T15:00:00Z"] * 2,
            "currency": ["JPY"] * 2,
            "pit_usable": [True] * 2,
            "source_sha256": ["a" * 64] * 2,
        }
    ).to_parquet(path, index=False)
    pd.DataFrame(
        [
            {
                "request_id": "202001-st",
                "status": "completed",
                "normalized_path": str(path),
                "normalized_sha256": _sha(path),
            }
        ]
    ).to_csv(source / "file_status.csv", index=False)
    pd.DataFrame([{"request_id": "202001-st", "quality_pass": True}]).to_csv(
        source / "file_quality.csv", index=False
    )

    result = run_jpx_monthly_membership_qa(
        source_dir=source, run_id="QA-PARTIAL", artifact_root=tmp_path / "qa"
    )

    assert result.manifest["completed_months"] == 1
    assert pd.read_csv(result.monthly_churn_path).empty
