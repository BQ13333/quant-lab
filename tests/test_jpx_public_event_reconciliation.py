from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.jpx_public_event_reconciliation import write_jpx_public_event_reconciliation


def _seed(root: Path) -> tuple[Path, Path]:
    public = root / "public"
    public.mkdir()
    pd.DataFrame(
        [{"event_date": "2024-02-01", "issue_name": "Old", "code": "1234", "market_segment": "Standard", "reason": "Merger"}]
    ).to_csv(public / "delisting_events.csv", index=False)
    pd.DataFrame(
        [{"listing_date": "2023-01-04", "approval_date": "2022-12-01", "issue_name": "New", "code": "56780", "market_segment": "Growth"}]
    ).to_csv(public / "new_listing_events.csv", index=False)
    pd.DataFrame([{"page_id": "p1"}]).to_csv(public / "request_plan.csv", index=False)
    pd.DataFrame([{"page_id": "p1", "status": "completed"}]).to_csv(public / "page_status.csv", index=False)
    sha = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in public.glob("*.csv")
    }
    (public / "manifest.json").write_text(
        json.dumps({
            "status": "complete_limited_coverage",
            "official_jpx_source": True,
            "coverage": {"planned_pages": 1, "completed_pages": 1},
            "sha256": sha,
        }),
        encoding="utf-8",
    )
    reference = root / "reference.csv"
    pd.DataFrame(
        [
            {"local_code": "1234", "refinitiv_ric": "1234.T", "retire_date": "2024-02-01", "first_trade_date": "2000-01-01"},
            {"local_code": "5678", "refinitiv_ric": "5678.T", "retire_date": "", "first_trade_date": "2023-01-04"},
        ]
    ).to_csv(reference, index=False)
    return public, reference


def test_public_events_cross_validate_recent_refinitiv_dates(tmp_path: Path):
    public, reference = _seed(tmp_path)
    result = write_jpx_public_event_reconciliation(
        run_id="CROSS", public_events_dir=public, reference_map_csv=reference, artifact_root=tmp_path / "out"
    )
    assert result.manifest["status"] == "pass"
    assert result.manifest["counts"]["events"] == 2
    assert result.manifest["counts"]["mapping_matched"] == 2
    assert result.manifest["counts"]["date_exact_match"] == 2
    assert result.manifest["scorecard_eligible"] is False


def test_public_source_hash_tampering_is_rejected(tmp_path: Path):
    public, reference = _seed(tmp_path)
    (public / "delisting_events.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        write_jpx_public_event_reconciliation(
            run_id="BAD", public_events_dir=public, reference_map_csv=reference, artifact_root=tmp_path / "out"
        )


def test_full_security_master_resolves_reference_map_gap_by_exact_event_date(tmp_path: Path):
    public, reference = _seed(tmp_path)
    ref = pd.read_csv(reference)
    ref = ref.loc[~ref["local_code"].astype(str).eq("1234")]
    ref.to_csv(reference, index=False)
    security = tmp_path / "security.csv"
    pd.DataFrame(
        [
            {
                "ticker_symbol": "1234",
                "ric": "1234.T^OLD",
                "exchange_code": "TYO",
                "retire_date": "2024-02-01",
                "first_trade_date": "2000-01-01",
            }
        ]
    ).to_csv(security, index=False)
    result = write_jpx_public_event_reconciliation(
        run_id="RESOLVE",
        public_events_dir=public,
        reference_map_csv=reference,
        security_master_csv=security,
        artifact_root=tmp_path / "out",
    )
    assert result.manifest["counts"]["mapping_matched"] == 2
    rows = pd.read_csv(result.reconciliation_path)
    resolved = rows.loc[rows["code"].astype(str).eq("1234")].iloc[0]
    assert resolved["mapping_status"] == "matched_security_master_exact_event_date"
