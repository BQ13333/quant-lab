from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g4_halt_denominator_recovery import (
    DELISTING_CHANNEL,
    HALT_CHANNEL,
    discover_lifecycle_urls,
    inspect_lifecycle_text,
    run_japan_g4_halt_denominator_recovery,
    validate_recovery_artifacts,
)


PDF_TEXT = """17 Changes in Listed Companies and Issues, Etc. (Jan. 2016)
Delisting
1st Section 2016.01.21 9870 Example
Suspension of Trading
1st Section 2016.01.27 7262 Example 8:20-10:30
"""


def test_discovery_accepts_only_exact_lifecycle_links() -> None:
    html = b'<a href="/x/17_idou1601.pdf">ok</a><a href="/x/16_allse1601.pdf">no</a>'
    assert discover_lifecycle_urls(html, "https://www.jpx.co.jp/archive.html") == {
        "2016-01": "https://www.jpx.co.jp/x/17_idou1601.pdf"
    }


def test_text_inspection_requires_exact_month_and_explicit_sections() -> None:
    inspected = inspect_lifecycle_text(PDF_TEXT, "2016-01")
    assert inspected["exact_month"]
    assert inspected["sections"][DELISTING_CHANNEL]["event_rows"] == 1
    assert inspected["sections"][HALT_CHANNEL]["event_rows"] == 1
    missing = inspect_lifecycle_text(
        "Changes in Listed Companies and Issues, Etc. (Jan. 2016)\nNew Listings\n2016.01.05 1234 X",
        "2016-01",
    )
    assert not missing["sections"][HALT_CHANNEL]["explicit"]
    assert not missing["sections"][DELISTING_CHANNEL]["explicit"]


def test_legacy_section_stops_at_next_numbered_report() -> None:
    inspected = inspect_lifecycle_text(
        PDF_TEXT
        + "\n21 Ex-New, Ex-Rights, Etc. (Jan. 2016)\n"
        + "2016.01.30 9999 Must Not Count\n",
        "2016-01",
    )
    assert inspected["sections"][HALT_CHANNEL]["event_rows"] == 1


def test_runner_is_fail_closed_and_hash_validation_detects_tamper(tmp_path: Path) -> None:
    gaps = tmp_path / "gaps.csv"
    pd.DataFrame(
        [
            {"gap_id": "h", "gap_type": "source_denominator_uncovered", "scope": HALT_CHANNEL, "month": "2016-01"},
            {"gap_id": "d", "gap_type": "source_denominator_uncovered", "scope": DELISTING_CHANNEL, "month": "2016-01"},
            {"gap_id": "z", "gap_type": "source_denominator_uncovered", "scope": HALT_CHANNEL, "month": "2016-02"},
        ]
    ).to_csv(gaps, index=False)
    archive_html = b'<a href="/reports/17_idou1601.pdf">Jan</a><a href="/reports/17_idou1602.pdf">Feb</a>'

    def fetch(url: str):
        if url.endswith(".html"):
            return 200, archive_html, {}
        return 200, b"%PDF fixture", {}

    result = run_japan_g4_halt_denominator_recovery(
        run_id="TEST",
        unresolved_gaps_path=gaps,
        artifact_root=tmp_path / "artifacts",
        end_year=2016,
        fetcher=fetch,
        text_extractor=lambda body: PDF_TEXT if b"fixture" in body else "",
        require_expected_gaps=False,
        require_oracle=False,
    )
    coverage = pd.read_csv(result.coverage_path)
    assert coverage.loc[coverage["month"].eq("2016-01"), "covered"].all()
    assert not coverage.loc[coverage["month"].eq("2016-02"), "covered"].any()
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["absence_is_zero"] is False
    assert validate_recovery_artifacts(result.artifact_dir) == []
    result.coverage_path.write_text("tampered\n", encoding="utf-8")
    assert "artifact_hash_mismatch:month_channel_coverage.csv" in validate_recovery_artifacts(result.artifact_dir)
    try:
        run_japan_g4_halt_denominator_recovery(
            run_id="TEST",
            unresolved_gaps_path=gaps,
            artifact_root=tmp_path / "artifacts",
            end_year=2016,
            fetcher=fetch,
            text_extractor=lambda body: PDF_TEXT,
            require_expected_gaps=False,
            require_oracle=False,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("write-once producer must reject an existing run id")


def test_oracle_gate_preserves_and_rejects_unsupported_claim(tmp_path: Path) -> None:
    gaps = tmp_path / "gaps.csv"
    pd.DataFrame([
        {"gap_id": "d", "gap_type": "source_denominator_uncovered", "scope": DELISTING_CHANNEL, "month": "2016-01"},
        {"gap_id": "h", "gap_type": "source_denominator_uncovered", "scope": HALT_CHANNEL, "month": "2016-01"},
    ]).to_csv(gaps, index=False)
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    claims = oracle / "claim_reconciliation.csv"
    pd.DataFrame([
        {"gap_id": "d", "independent_event_rows": 1, "independent_explicit_zero": False, "explicit_section_proven": True, "source_identity_match": True},
        {"gap_id": "h", "independent_event_rows": 0, "independent_explicit_zero": False, "explicit_section_proven": False, "source_identity_match": True},
    ]).to_csv(claims, index=False)
    claim_hash = hashlib.sha256(claims.read_bytes()).hexdigest()
    hashes = oracle / "artifact_hashes.json"
    hashes.write_text(json.dumps({"claim_reconciliation.csv": claim_hash}), encoding="utf-8")
    (oracle / "manifest.json").write_text(
        json.dumps({"status": "failed_validation", "artifact_hashes_sha256": hashlib.sha256(hashes.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )

    result = run_japan_g4_halt_denominator_recovery(
        run_id="ORACLE",
        unresolved_gaps_path=gaps,
        artifact_root=tmp_path / "artifacts",
        end_year=2016,
        fetcher=lambda url: (
            (200, b'<a href="/reports/17_idou1601.pdf">Jan</a>', {})
            if url.endswith(".html") else (200, b"%PDF fixture", {})
        ),
        text_extractor=lambda body: PDF_TEXT,
        require_expected_gaps=False,
        oracle_validation_dir=oracle,
    )
    coverage = pd.read_csv(result.coverage_path).set_index("gap_id")
    assert bool(coverage.loc["d", "covered"])
    assert not bool(coverage.loc["h", "covered"])
    assert coverage.loc["h", "acceptance_reason"] == "rejected_by_independent_oracle_no_explicit_event_or_zero"
