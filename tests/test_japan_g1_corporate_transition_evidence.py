from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import quant_lab.data.japan_g1_corporate_transition_evidence as module
from quant_lab.data.japan_g1_corporate_transition_evidence import (
    PUBLIC_EVENT_ONLY_CODES,
    LITALICO_SUPPLEMENTAL_URL,
    SUPPLEMENTAL_URL,
    TPM_DELISTING_URL,
    TPM_OUTLINE_URL,
    build_technical_listing_notice_plan,
    canonicalize_jpx_name,
    is_stock_technical_listing_title,
    parse_supplemental_transition_page,
    parse_litalico_name_bridge_pdf,
    parse_technical_listing_notice,
    parse_tpm_transition_evidence,
    write_japan_g1_corporate_transition_evidence,
)


def test_exact_canonicalization_and_stock_only_title_classification() -> None:
    assert canonicalize_jpx_name("  ＡＢＣ・Holdings, Inc. ** (Stock Transfer) ") == "ABCHOLDINGSINC"
    assert canonicalize_jpx_name("ABC Holdings Inc.") == "ABCHOLDINGSINC"
    assert is_stock_technical_listing_title(
        "Base prices and daily price limits for newly listing stocks: Alpha"
    )
    assert is_stock_technical_listing_title(
        "Base price and daily price limit for newly listing stock: Alpha"
    )
    assert not is_stock_technical_listing_title(
        "Base prices and daily price limits for newly listing ETF: Alpha"
    )
    assert not is_stock_technical_listing_title(
        "Base prices and daily price limits for newly listing stocks: Convertible Bond"
    )


def test_listing_parser_assigns_exact_dates_in_mixed_date_notice() -> None:
    body = b"""
    <div id="readArea"><p>The initial listing of 2 stocks is scheduled for October 3, 2022
    (1 stock is scheduled for October 1)</p><table>
    <tr><td>Alpha Holdings stocks (Code: 1001)</td></tr>
    <tr><td>Beta Holdings stocks (Code: 1002)</td></tr>
    <tr><td>Gamma Holdings stocks (Code: 1003)</td></tr>
    </table></div>
    """
    rows = parse_technical_listing_notice(
        body,
        publication_date="2022-09-29",
        source_url="https://www.jpx.co.jp/english/news/1030/20220929-03.html",
        title="Base prices and daily price limits for newly listing stocks",
    )
    assert [(row["successor_code"], row["official_listing_date"]) for row in rows] == [
        ("1001", "2022-10-03"),
        ("1002", "2022-10-03"),
        ("1003", "2022-10-01"),
    ]


def test_listing_parser_removes_legacy_no_space_stock_suffix() -> None:
    rows = parse_technical_listing_notice(
        b"""
        <div id="readArea"><p>The initial listing of 1 stock is scheduled for October 1, 2020.</p>
        <table><tr><td>Hirogin Holdings,Inc.stocks (Code: 7337)</td></tr></table></div>
        """,
        publication_date="2020-09-29",
        source_url="https://www.jpx.co.jp/english/news/1030/20200929-03.html",
        title="Base prices and daily price limits for newly listing stocks",
    )
    assert rows[0]["successor_issue_name"] == "Hirogin Holdings,Inc"
    assert rows[0]["official_listing_date"] == "2020-10-01"


def test_supplemental_parser_extracts_both_exact_identifiers() -> None:
    parsed = parse_supplemental_transition_page(
        _supplemental_html(), source_url=SUPPLEMENTAL_URL
    )
    assert parsed == {
        "old_code": "3398",
        "old_issue_name": "KUSURI NO AOKI CO.,LTD.",
        "old_event_date": "2016-11-16",
        "old_event_reason": "Stock Swap",
        "successor_code": "3549",
        "successor_issue_name": "KUSURI NO AOKI HOLDINGS CO.,LTD.",
        "official_listing_date": "2016-11-21",
        "source_url": SUPPLEMENTAL_URL,
    }


def test_tpm_parser_reconciles_pdf_and_delisting_page(monkeypatch: pytest.MonkeyPatch) -> None:
    class Page:
        def extract_text(self) -> str:
            return (
                "Aug. 16, 2021 Outline of Company FT Holdings Co.,Inc. (hereinafter the Company), "
                "established by listed company Fuji Techno Solutions Co.,Inc. "
                "(Code: 2336, Market Section: TOKYO PRO Market) via stock transfer. "
                "Scheduled Listing Date Oct. 1, 2021 Category of Industry / Code Services 9243"
            )

    class Reader:
        def __init__(self, _: object) -> None:
            self.pages = [Page()]

    monkeypatch.setattr(module, "PdfReader", Reader)
    parsed = parse_tpm_transition_evidence(b"pdf", _tpm_delisting_html())
    assert parsed["old_code"] == "2336"
    assert parsed["successor_code"] == "9243"
    assert parsed["old_event_date"] == "2021-09-29"
    assert parsed["official_listing_date"] == "2021-10-01"
    assert parsed["transition_mechanism"] == "Stock Transfer"


def test_litalico_name_bridge_parser_requires_exact_codes_and_effective_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Page:
        def extract_text(self) -> str:
            return (
                "LITALICO Inc. (6187) will be deleted. "
                "LITALICO Media & Solutions Inc. (7366), a new parent company, will be added. "
                "LITALICO Media & Solutions Inc. will change its company name to "
                "LITALICO Inc. on April 1, 2021."
            )

    class Reader:
        def __init__(self, _: object) -> None:
            self.pages = [Page()]

    monkeypatch.setattr(module, "PdfReader", Reader)
    parsed = parse_litalico_name_bridge_pdf(b"pdf")
    assert parsed["old_code"] == "6187"
    assert parsed["successor_code"] == "7366"
    assert parsed["name_change_effective_date"] == "2021-04-01"


def test_notice_plan_is_hash_verified_and_excludes_non_stock_page(tmp_path: Path) -> None:
    archive = _archive(tmp_path, include_bond=True)
    plan = build_technical_listing_notice_plan(archive)
    assert len(plan) == 1
    assert plan.iloc[0]["url"].endswith("20210929-01.html")
    with archive.open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_technical_listing_notice_plan(archive)


def test_bounded_writer_covers_three_classes_and_resumes_without_refetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quote = _quote_candidates(tmp_path)
    archive = _archive(tmp_path)
    delistings, successor_rows = _delistings(tmp_path)
    listing_html = _listing_html(successor_rows)
    calls: list[str] = []

    def fetcher(url: str) -> tuple[int, bytes, dict[str, str]]:
        calls.append(url)
        if url == SUPPLEMENTAL_URL:
            return 200, _supplemental_html(), {"Content-Type": "text/html"}
        if url == LITALICO_SUPPLEMENTAL_URL:
            return 200, b"litalico-fixture-pdf", {"Content-Type": "application/pdf"}
        if url == TPM_OUTLINE_URL:
            return 200, b"fixture-pdf", {"Content-Type": "application/pdf"}
        if url == TPM_DELISTING_URL:
            return 200, _tpm_delisting_html(), {"Content-Type": "text/html"}
        return 200, listing_html, {"Content-Type": "text/html"}

    monkeypatch.setattr(
        module,
        "parse_tpm_transition_evidence",
        lambda *_args, **_kwargs: {
            "old_code": "2336",
            "old_issue_name": "Fuji Techno Solutions Co.,Inc.",
            "old_event_date": "2021-09-29",
            "old_event_reason": "Delisted based on application.",
            "successor_code": "9243",
            "successor_issue_name": "FT Holdings Co.,Inc.",
            "official_listing_date": "2021-10-01",
            "transition_mechanism": "Stock Transfer",
            "listing_source_url": TPM_OUTLINE_URL,
            "old_event_source_url": TPM_DELISTING_URL,
        },
    )
    monkeypatch.setattr(
        module,
        "parse_litalico_name_bridge_pdf",
        lambda *_args, **_kwargs: {
            "old_code": "6187",
            "old_issue_name": "LITALICO Inc.",
            "successor_code": "7366",
            "successor_pre_change_name": "LITALICO Media & Solutions Inc.",
            "successor_listing_name": "LITALICO Inc.",
            "name_change_effective_date": "2021-04-01",
            "source_url": LITALICO_SUPPLEMENTAL_URL,
        },
    )
    kwargs = {
        "run_id": "FIXTURE-V1",
        "quote_chain_candidates_path": quote,
        "archive_manifest_path": archive,
        "delisting_events_path": delistings,
        "artifact_root": tmp_path / "output",
        "sleep_seconds": 0,
        "fetcher": fetcher,
        "expected_class_counts": {
            "identity_anchored": 1,
            "public_event_only": 11,
            "tpm_official": 1,
        },
    }
    result = write_japan_g1_corporate_transition_evidence(**kwargs)
    reconciliation = pd.read_csv(result.reconciliation_path, dtype=str).fillna("")
    admitted = pd.read_csv(result.admitted_candidates_path, dtype=str).fillna("")
    assert len(reconciliation) == 13
    assert len(admitted) == 13
    assert reconciliation["evidence_class"].value_counts().to_dict() == {
        "public_event_only": 11,
        "identity_anchored": 1,
        "tpm_official": 1,
    }
    assert result.manifest["counts"]["direct_evidence_preserved"] == 1
    assert result.manifest["counts"]["transition_denominator"] == 13
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["promotion_eligible"] is False
    assert result.manifest["independent_validation_required"] is True
    assert len(calls) == 5

    calls.clear()
    write_japan_g1_corporate_transition_evidence(**kwargs)
    assert calls == []


def test_bounded_run_emits_pending_sources_instead_of_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quote = _quote_candidates(tmp_path)
    archive = _archive(tmp_path)
    delistings, _ = _delistings(tmp_path)
    result = write_japan_g1_corporate_transition_evidence(
        run_id="PENDING-V1",
        quote_chain_candidates_path=quote,
        archive_manifest_path=archive,
        delisting_events_path=delistings,
        artifact_root=tmp_path / "output",
        max_pages=1,
        sleep_seconds=0,
        fetcher=lambda _url: (503, b"unavailable", {}),
        expected_class_counts={
            "identity_anchored": 1,
            "public_event_only": 11,
            "tpm_official": 1,
        },
    )
    state = pd.read_csv(result.artifact_dir / "page_state.csv", dtype=str).fillna("")
    assert state["terminal_status"].eq("pending").sum() == 4
    assert state["terminal_status"].eq("retryable_failure").sum() == 1
    assert any("official_source_fetch_incomplete" in item for item in result.manifest["critical_blockers"])


def _quote_candidates(tmp_path: Path) -> Path:
    root = tmp_path / "quote"
    root.mkdir(exist_ok=True)
    path = root / "quote_chain_candidates.csv"
    pd.DataFrame(
        [
            {
                "identity_row_id": "A",
                "local_code": "1000",
                "successor_primary_rics": "2000.T",
                "successor_codes": "2000",
                "direct_primary_market_rics": "1000.T^A21",
            }
        ]
    ).to_csv(path, index=False)
    _manifest(root, "output_sha256", {path.name: _hash(path)})
    return path


def _archive(tmp_path: Path, include_bond: bool = False) -> Path:
    root = tmp_path / "archive"
    raw = root / "raw.json"
    root.mkdir(exist_ok=True)
    items = [
        {
            "title": "Base prices and daily price limits for newly listing stocks: fixture",
            "url": "/english/news/1030/20210929-01.html",
            "updated_date": {"year": "2021", "month": "09", "day": "29"},
        }
    ]
    if include_bond:
        items.append(
            {
                "title": "Base prices and daily price limits for newly listing stocks: Convertible Bond",
                "url": "/english/news/1030/20210929-02.html",
                "updated_date": {"year": "2021", "month": "09", "day": "29"},
            }
        )
    raw.write_text(json.dumps({"items": items}), encoding="utf-8")
    path = root / "archive_source_manifest.csv"
    pd.DataFrame(
        [
            {
                "source_id": "archive-1",
                "url": "https://www.jpx.co.jp/archive.json",
                "http_status": 200,
                "content_type": "application/json",
                "raw_path": str(raw),
                "sha256": _hash(raw),
                "bytes": raw.stat().st_size,
            }
        ]
    ).to_csv(path, index=False)
    _manifest(root, "output_sha256", {path.name: _hash(path)})
    return path


def _delistings(tmp_path: Path) -> tuple[Path, list[tuple[str, str]]]:
    root = tmp_path / "delist"
    root.mkdir(exist_ok=True)
    successors = [("2000", "Anchor Holdings")]
    rows = [
        {
            "event_date": "2021-09-29",
            "issue_name": "Anchor Co.",
            "code": "1000",
            "reason": "Becoming a wholly owned subsidiary of Anchor Holdings",
            "source_url": "https://www.jpx.co.jp/old",
        }
    ]
    for index, code in enumerate(PUBLIC_EVENT_ONLY_CODES):
        successor_code = str(3000 + index)
        successor_name = f"Public Successor {index} Holdings"
        successors.append((successor_code, successor_name))
        rows.append(
            {
                "event_date": "2021-09-29",
                "issue_name": f"Public Old {index}",
                "code": code,
                "reason": f"Becoming a wholly owned subsidiary of {successor_name}",
                "source_url": "https://www.jpx.co.jp/old",
            }
        )
    path = root / "delisting_events.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    _manifest(root, "sha256", {path.name: _hash(path)})
    return path, successors


def _listing_html(successors: list[tuple[str, str]]) -> bytes:
    cells = "".join(
        f"<tr><td>{name} stocks (Code: {code})</td></tr>" for code, name in successors
    )
    return (
        "<div id='readArea'><p>The initial listing of 12 stocks is scheduled for October 1, 2021.</p>"
        f"<table>{cells}</table></div>"
    ).encode()


def _supplemental_html() -> bytes:
    return b"""
    <div id="readArea"><p>shares to be de-listed on November 16, 2016 and shares to be
    listed on 1st section November 21, 2016 due to stock swap.</p>
    <h3>1. Exclusion</h3><table><tr><td>KUSURI NO AOKI CO.,LTD.</td><td>3398</td></tr></table>
    <h3>2. Inclusion</h3><table><tr><td>KUSURI NO AOKI HOLDINGS CO.,LTD.</td><td>3549</td></tr></table>
    </div>
    """


def _tpm_delisting_html() -> bytes:
    return b"""
    <table><tr><th>Date of Delisting</th><th>Code</th><th>Name of Company</th><th>Reason</th></tr>
    <tr><td>Sep. 29, 2021</td><td>2336</td><td>Fuji Techno Solutions Co.,Inc.</td>
    <td>Delisted based on application.</td></tr></table>
    """


def _manifest(root: Path, field: str, hashes: dict[str, str]) -> None:
    (root / "manifest.json").write_text(json.dumps({field: hashes}), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
