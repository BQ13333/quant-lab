from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g1_successor_vendor_plan import (
    write_japan_g1_successor_vendor_plan,
)


def test_successor_vendor_plan_is_exact_hash_verified_and_non_promotable(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, lookup_rows=2)
    result = write_japan_g1_successor_vendor_plan(
        run_id="PLAN-V1",
        transition_evidence_dir=source,
        artifact_root=tmp_path / "output",
        expected_lookup_rows=2,
    )
    plan = pd.read_csv(result.plan_path, dtype=str).fillna("")
    assert list(plan["local_code"]) == ["165A", "9243"]
    assert set(plan["lookup_status"]) == {"planned_not_executed"}
    assert result.manifest["refinitiv_called"] is False
    assert result.manifest["scorecard_eligible"] is False
    assert not any(result.manifest["promotion_policy"].values())
    assert result.manifest["output_sha256"][result.plan_path.name] == _hash(result.plan_path)


def test_successor_vendor_plan_rejects_tampering_and_wrong_denominator(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, lookup_rows=1)
    with pytest.raises(ValueError, match="denominator mismatch"):
        write_japan_g1_successor_vendor_plan(
            run_id="BAD-DENOMINATOR",
            transition_evidence_dir=source,
            artifact_root=tmp_path / "output",
            expected_lookup_rows=2,
        )
    path = source / "admitted_transition_candidates.csv"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="contract invalid"):
        write_japan_g1_successor_vendor_plan(
            run_id="TAMPERED",
            transition_evidence_dir=source,
            artifact_root=tmp_path / "output",
            expected_lookup_rows=1,
        )


def _source(tmp_path: Path, *, lookup_rows: int) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    rows = []
    for index in range(66):
        missing = index < lookup_rows
        code = "9243" if index == 0 else "165A" if index == 1 else f"{index:04d}"
        rows.append(
            {
                "event_segment_admitted": True,
                "successor_primary_rics": "" if missing else f"{code}.T",
                "successor_code": code,
                "old_code": f"{9000 + index:04d}"[-4:],
                "evidence_class": (
                    "tpm_official" if index == 0 else "public_event_only" if missing else "identity_anchored"
                ),
                "successor_issue_name": f"Successor {index}",
                "official_listing_date": "2021-10-01",
                "listing_source_url": "https://www.jpx.co.jp/official.html",
                "old_event_date": "2021-09-29",
                "old_event_source_url": "https://www.jpx.co.jp/delisted.html",
                "name_match_basis": "reason_target_exact_canonical",
                "gap_days": 2,
            }
        )
    candidates = root / "admitted_transition_candidates.csv"
    pd.DataFrame(rows).to_csv(candidates, index=False)
    manifest = {
        "artifact_type": "japan_g1_official_corporate_transition_evidence_candidate",
        "status": "complete_candidate",
        "denominator_valid": True,
        "scorecard_eligible": False,
        "independent_validation_required": True,
        "counts": {
            "transition_denominator": 66,
            "event_segments_admitted_candidate": 66,
            "unresolved_candidates": 0,
        },
        "output_sha256": {candidates.name: _hash(candidates)},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
