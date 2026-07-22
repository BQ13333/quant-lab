from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader


UTC = timezone.utc
JPX_ROOT = "https://www.jpx.co.jp"
SUPPLEMENTAL_URL = "https://www.jpx.co.jp/english/news/1044/20161025-01.html"
LITALICO_SUPPLEMENTAL_URL = (
    "https://www.jpx.co.jp/english/news/1044/"
    "b5b4pj000003z80o-att/dara2_e.pdf"
)
TPM_OUTLINE_URL = (
    "https://www.jpx.co.jp/english/equities/products/tpm/issues/"
    "b5b4pj0000049po4-att/b5b4pj0000049pq1.pdf"
)
TPM_DELISTING_URL = (
    "https://www.jpx.co.jp/english/equities/products/tpm/delisting/index.html"
)
PUBLIC_EVENT_ONLY_CODES = (
    "3464",
    "3669",
    "3999",
    "6187",
    "6531",
    "6569",
    "7190",
    "7323",
    "7330",
    "7688",
    "9266",
)
EXPECTED_EVIDENCE_CLASS_COUNTS = {
    "identity_anchored": 54,
    "public_event_only": 11,
    "tpm_official": 1,
}

Fetcher = Callable[[str], tuple[int, bytes, dict[str, str]]]


@dataclass(frozen=True)
class JapanG1CorporateTransitionEvidenceArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    reconciliation_path: Path
    admitted_candidates_path: Path
    manifest: dict[str, Any]


def canonicalize_jpx_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(str(value))).upper()
    normalized = re.sub(r"\*\*", "", normalized)
    normalized = re.sub(
        r"\s*\(\s*(?:STOCK\s+TRANSFER|STOCK\s+SWAP)\s*\)\s*$", "", normalized
    )
    return "".join(character for character in normalized if character.isalnum())


def is_stock_technical_listing_title(title: str) -> bool:
    value = " ".join(unicodedata.normalize("NFKC", html.unescape(str(title))).split())
    if re.search(r"\b(?:ETF|ETN|bond|option|warrant)\b", value, re.I):
        return False
    return bool(
        re.match(
            r"^Base prices? and daily price limits? for newly list(?:ing|ed) stocks?\b",
            value,
            re.I,
        )
    )


def build_technical_listing_notice_plan(
    archive_manifest_path: str | Path,
) -> pd.DataFrame:
    archive_path = Path(archive_manifest_path).resolve()
    _verify_manifest_output(archive_path, "output_sha256")
    archive = pd.read_csv(archive_path, dtype=str).fillna("")
    rows: dict[str, dict[str, Any]] = {}
    for source in archive.to_dict("records"):
        raw_path = Path(source["raw_path"])
        if (
            source.get("http_status") != "200"
            or not raw_path.is_file()
            or _sha256(raw_path) != source.get("sha256", "").lower()
        ):
            raise ValueError(f"archive source contract invalid: {source.get('source_id', '')}")
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        for item in _walk_news_items(payload):
            title = " ".join(str(item.get("title", "")).split())
            if not is_stock_technical_listing_title(title):
                continue
            publication = _publication_date(item)
            url = urljoin(JPX_ROOT, str(item.get("url", "")))
            if not publication or not re.fullmatch(
                r"https://www\.jpx\.co\.jp/english/news/"
                r"(?:1030/[0-9]{8}-[0-9]+|detail/detail_[0-9]+)\.html",
                url,
            ):
                raise ValueError(f"selected stock notice is malformed: {url}")
            row = rows.setdefault(
                url,
                {
                    "publication_date": publication.isoformat(),
                    "title": title,
                    "url": url,
                    "archive_source_ids": [],
                },
            )
            row["archive_source_ids"].append(str(source["source_id"]))
    plan = pd.DataFrame(
        [
            {
                **row,
                "archive_source_ids": "|".join(sorted(set(row["archive_source_ids"]))),
            }
            for row in rows.values()
        ]
    )
    plan = plan.sort_values(["publication_date", "url"]).reset_index(drop=True)
    plan.insert(0, "request_id", [f"JCT-LIST-{index:03d}" for index in range(1, len(plan) + 1)])
    plan.insert(1, "source_kind", "technical_listing_notice")
    return plan


def parse_technical_listing_notice(
    body: bytes,
    *,
    publication_date: str,
    source_url: str,
    title: str,
) -> list[dict[str, str]]:
    soup = BeautifulSoup(body, "html.parser")
    area = soup.select_one("#readArea") or soup
    issue_rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for cell in area.find_all(["td", "th"]):
        value = " ".join(cell.stripped_strings)
        match = re.fullmatch(
            r"(.+?)\s+(?:stocks?\s*)?\(Code\s*:\s*([0-9A-Z]{4})\)", value, re.I
        )
        if not match:
            continue
        code = match.group(2).upper()
        if code in seen:
            continue
        name = _strip_notice_stock_suffix(match.group(1))
        issue_rows.append((code, name))
        seen.add(code)
    text = " ".join(area.stripped_strings)
    if not issue_rows:
        for match in re.finditer(
            r"([A-Za-z0-9][^()]{1,180}?)\s+(?:stocks?\s*)?"
            r"\(Code\s*:\s*([0-9A-Z]{4})\)",
            text,
            re.I,
        ):
            code = match.group(2).upper()
            if code not in seen:
                issue_rows.append((code, _strip_notice_stock_suffix(match.group(1))))
                seen.add(code)
    dates = _listing_dates_by_code(text, issue_rows, publication_date)
    return [
        {
            "successor_code": code,
            "successor_issue_name": name,
            "official_listing_date": dates.get(code, ""),
            "listing_publication_date": publication_date,
            "listing_source_url": source_url,
            "listing_source_title": title,
        }
        for code, name in issue_rows
    ]


def parse_supplemental_transition_page(body: bytes, *, source_url: str) -> dict[str, str]:
    soup = BeautifulSoup(body, "html.parser")
    area = soup.select_one("#readArea") or soup
    text = " ".join(" ".join(area.stripped_strings).split())
    exclusion = _table_identity_after_heading(area, "Exclusion")
    inclusion = _table_identity_after_heading(area, "Inclusion")
    delisted = _first_official_date(text, r"de-?listed\s+on\s+({date})", "2016-10-25")
    listed = _first_official_date(
        text,
        r"shares\s+to\s+be\s+listed\s+on(?:\s+1st\s+section)?\s+({date})",
        "2016-10-25",
    )
    if not exclusion or not inclusion or not delisted or not listed:
        raise ValueError("supplemental transition page could not be parsed exactly")
    return {
        "old_code": exclusion[1],
        "old_issue_name": exclusion[0],
        "old_event_date": delisted,
        "old_event_reason": "Stock Swap",
        "successor_code": inclusion[1],
        "successor_issue_name": inclusion[0],
        "official_listing_date": listed,
        "source_url": source_url,
    }


def parse_litalico_name_bridge_pdf(
    body: bytes, *, source_url: str = LITALICO_SUPPLEMENTAL_URL
) -> dict[str, str]:
    text = " ".join(
        (page.extract_text() or "") for page in PdfReader(io.BytesIO(body)).pages
    )
    text = " ".join(text.split())
    old = re.search(r"LITALICO Inc\.\s*\(\s*6187\s*\)", text, re.I)
    successor = re.search(
        r"LITALICO Media\s*&\s*Solutions Inc\.\s*\(\s*7366\s*\)", text, re.I
    )
    rename = re.search(
        r"LITALICO Media\s*&\s*Solutions Inc\.\s+will change its company name to\s+"
        r"LITALICO Inc\.\s+on\s+({date})".format(date=_date_pattern()),
        text,
        re.I,
    )
    if not old or not successor or not rename:
        raise ValueError("LITALICO official name-bridge PDF could not be parsed exactly")
    effective_date = _parse_english_date(rename.group(1), date(2021, 3, 3))
    if effective_date != "2021-04-01":
        raise ValueError("LITALICO name-bridge effective date mismatch")
    return {
        "old_code": "6187",
        "old_issue_name": "LITALICO Inc.",
        "successor_code": "7366",
        "successor_pre_change_name": "LITALICO Media & Solutions Inc.",
        "successor_listing_name": "LITALICO Inc.",
        "name_change_effective_date": effective_date,
        "source_url": source_url,
    }


def parse_tpm_transition_evidence(
    outline_pdf: bytes,
    delisting_html: bytes,
    *,
    outline_url: str = TPM_OUTLINE_URL,
    delisting_url: str = TPM_DELISTING_URL,
) -> dict[str, str]:
    pdf_text = " ".join(
        (page.extract_text() or "") for page in PdfReader(io.BytesIO(outline_pdf)).pages
    )
    pdf_text = " ".join(pdf_text.split())
    old = re.search(r"(Fuji Techno Solutions Co\.,Inc\.)\s*\(Code:\s*(2336)\b", pdf_text, re.I)
    successor_name = re.search(r"Outline of Company\s+(.+?)\s*\(hereinafter", pdf_text, re.I)
    successor_code = re.search(
        r"Category of Industry\s*/?\s*Code\b.{0,80}?\b(9243)\b",
        pdf_text,
        re.I,
    )
    listing_date = _first_official_date(
        pdf_text, r"Scheduled Listing Date\s+({date})", "2021-08-16"
    )
    mechanism = bool(re.search(r"via stock transfer", pdf_text, re.I))

    soup = BeautifulSoup(delisting_html, "html.parser")
    delisting_row: tuple[str, str, str, str] | None = None
    for row in soup.find_all("tr"):
        cells = [" ".join(cell.stripped_strings) for cell in row.find_all(["td", "th"])]
        if len(cells) >= 4 and cells[1].strip() == "2336":
            event_date = _parse_english_date(cells[0], date(2021, 1, 1))
            delisting_row = (event_date, cells[1].strip(), cells[2].strip(), cells[3].strip())
            break
    if not delisting_row:
        page_text = " ".join(soup.stripped_strings)
        match = re.search(
            r"(Sep\.?\s+29,\s+2021)\s+(2336)\s+(Fuji Techno Solutions Co\.,Inc\.)\s+"
            r"(Delisted based on application\.?)",
            page_text,
            re.I,
        )
        if match:
            delisting_row = (
                _parse_english_date(match.group(1), date(2021, 1, 1)),
                match.group(2),
                match.group(3),
                match.group(4),
            )
    if not all((old, successor_name, successor_code, listing_date, mechanism, delisting_row)):
        raise ValueError("TPM transition evidence could not be parsed exactly")
    assert old and successor_name and successor_code and delisting_row
    if old.group(2) != delisting_row[1] or canonicalize_jpx_name(old.group(1)) != canonicalize_jpx_name(delisting_row[2]):
        raise ValueError("TPM PDF and delisting-page identity mismatch")
    return {
        "old_code": old.group(2),
        "old_issue_name": old.group(1),
        "old_event_date": delisting_row[0],
        "old_event_reason": delisting_row[3],
        "successor_code": successor_code.group(1),
        "successor_issue_name": successor_name.group(1).strip(),
        "official_listing_date": listing_date,
        "transition_mechanism": "Stock Transfer",
        "listing_source_url": outline_url,
        "old_event_source_url": delisting_url,
    }


def write_japan_g1_corporate_transition_evidence(
    *,
    run_id: str,
    quote_chain_candidates_path: str | Path,
    archive_manifest_path: str | Path,
    delisting_events_path: str | Path,
    artifact_root: str | Path = "artifacts/japan_g1_corporate_transition_evidence",
    max_pages: int = 0,
    sleep_seconds: float = 0.02,
    fetcher: Fetcher | None = None,
    expected_class_counts: dict[str, int] | None = None,
) -> JapanG1CorporateTransitionEvidenceArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    if max_pages < 0 or sleep_seconds < 0:
        raise ValueError("max_pages and sleep_seconds must be non-negative")
    expected = expected_class_counts or EXPECTED_EVIDENCE_CLASS_COUNTS
    if set(expected) != set(EXPECTED_EVIDENCE_CLASS_COUNTS):
        raise ValueError("expected_class_counts must contain all evidence classes")

    quote_path = Path(quote_chain_candidates_path).resolve()
    archive_path = Path(archive_manifest_path).resolve()
    delisting_path = Path(delisting_events_path).resolve()
    _verify_manifest_output(quote_path, "output_sha256")
    _verify_manifest_output(delisting_path, "sha256")
    quote_candidates = pd.read_csv(quote_path, dtype=str).fillna("")
    anchored = quote_candidates.loc[
        quote_candidates["successor_primary_rics"].str.strip().ne("")
        & quote_candidates["successor_codes"].str.strip().ne("")
    ].copy()
    if len(anchored) != expected["identity_anchored"]:
        raise ValueError(
            f"identity-anchored denominator mismatch: {len(anchored)} != {expected['identity_anchored']}"
        )
    delistings = pd.read_csv(delisting_path, dtype=str).fillna("")
    missing_public = set(PUBLIC_EVENT_ONLY_CODES) - set(delistings["code"])
    if missing_public or len(PUBLIC_EVENT_ONLY_CODES) != expected["public_event_only"]:
        raise ValueError(f"public-event-only denominator mismatch: missing={sorted(missing_public)}")

    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc
    try:
        plan = build_technical_listing_notice_plan(archive_path)
        special = pd.DataFrame(
            [
                {
                    "request_id": "JCT-SUPPLEMENTAL-3398",
                    "source_kind": "supplemental_transition",
                    "publication_date": "2016-10-25",
                    "title": "Calculation of TOPIX in relation to shares of KUSURI NO AOKI HOLDINGS CO.,LTD.",
                    "url": SUPPLEMENTAL_URL,
                    "archive_source_ids": "explicit_official_supplement",
                },
                {
                    "request_id": "JCT-SUPPLEMENTAL-6187",
                    "source_kind": "supplemental_litalico_name_bridge",
                    "publication_date": "2021-03-03",
                    "title": "Change to the JPX-Nikkei Mid and Small Cap Index",
                    "url": LITALICO_SUPPLEMENTAL_URL,
                    "archive_source_ids": "explicit_official_supplement",
                },
                {
                    "request_id": "JCT-TPM-OUTLINE-2336",
                    "source_kind": "tpm_outline_pdf",
                    "publication_date": "2021-08-16",
                    "title": "FT Holdings Co.,Inc. Outline of Company",
                    "url": TPM_OUTLINE_URL,
                    "archive_source_ids": "explicit_official_tpm_source",
                },
                {
                    "request_id": "JCT-TPM-DELISTING-2336",
                    "source_kind": "tpm_delisting_page",
                    "publication_date": "",
                    "title": "TOKYO PRO Market Delisted Companies",
                    "url": TPM_DELISTING_URL,
                    "archive_source_ids": "explicit_official_tpm_source",
                },
            ]
        )
        request_plan = pd.concat([plan, special], ignore_index=True)
        plan_path = root / "official_source_plan.csv"
        _atomic_csv(plan_path, request_plan)
        state_path = root / "page_state.csv"
        state = _fetch_plan(
            request_plan,
            state_path=state_path,
            raw_dir=root / "raw/official_pages",
            max_pages=max_pages,
            sleep_seconds=sleep_seconds,
            fetcher=fetcher or _fetch,
        )

        successes = state.loc[state["terminal_status"].eq("success")].copy()
        listing_rows: list[dict[str, str]] = []
        parse_failures: list[str] = []
        for row in successes.loc[successes["source_kind"].eq("technical_listing_notice")].to_dict("records"):
            parsed = parse_technical_listing_notice(
                Path(row["raw_path"]).read_bytes(),
                publication_date=row["publication_date"],
                source_url=row["url"],
                title=row["title"],
            )
            if not parsed or any(not item["official_listing_date"] for item in parsed):
                parse_failures.append(row["request_id"])
            listing_rows.extend(parsed)
        listing_events = pd.DataFrame(listing_rows)
        if not listing_events.empty:
            listing_events = listing_events.drop_duplicates(
                ["successor_code", "official_listing_date", "listing_source_url"]
            ).sort_values(["official_listing_date", "successor_code"]).reset_index(drop=True)
        listing_events_path = root / "technical_listing_events.csv"
        _atomic_csv(listing_events_path, listing_events)

        supplemental = _parse_special(successes, "supplemental_transition", parse_supplemental_transition_page)
        litalico_bridge = _parse_special(
            successes,
            "supplemental_litalico_name_bridge",
            parse_litalico_name_bridge_pdf,
        )
        tpm_outline = _special_bytes(successes, "tpm_outline_pdf")
        tpm_delisting = _special_bytes(successes, "tpm_delisting_page")
        tpm = (
            parse_tpm_transition_evidence(tpm_outline, tpm_delisting)
            if tpm_outline and tpm_delisting
            else {}
        )

        reconciliation = _reconcile(
            anchored=anchored,
            delistings=delistings,
            listing_events=listing_events,
            supplemental=supplemental,
            litalico_bridge=litalico_bridge,
            tpm=tpm,
        )
        reconciliation_path = root / "transition_reconciliation.csv"
        _atomic_csv(reconciliation_path, reconciliation)
        admitted = reconciliation.loc[reconciliation["event_segment_admitted"].eq(True)].copy()
        admitted_path = root / "admitted_transition_candidates.csv"
        _atomic_csv(admitted_path, admitted)

        class_counts = reconciliation["evidence_class"].value_counts().to_dict()
        denominator_valid = class_counts == expected and len(reconciliation) == sum(expected.values())
        completed = int(state["terminal_status"].eq("success").sum())
        pending = int(state["terminal_status"].eq("pending").sum())
        failed = int(state["terminal_status"].eq("retryable_failure").sum())
        blockers = []
        if not denominator_valid:
            blockers.append("transition_denominator_mismatch")
        if pending or failed:
            blockers.append(f"official_source_fetch_incomplete:pending={pending}:failed={failed}")
        if parse_failures:
            blockers.append(f"technical_listing_parse_failures={len(parse_failures)}")
        unresolved = int(reconciliation["event_segment_admitted"].eq(False).sum())
        if unresolved:
            blockers.append(f"unresolved_transition_candidates={unresolved}")
        manifest = {
            "schema_version": "1.0-immutable-candidate-evidence",
            "run_id": run_id,
            "artifact_type": "japan_g1_official_corporate_transition_evidence_candidate",
            "status": "complete_candidate" if not blockers else "in_progress_candidate",
            "official_jpx_source": True,
            "refinitiv_called": False,
            "immutable_source_hashes": True,
            "scorecard_eligible": False,
            "promotion_eligible": False,
            "independent_validation_required": True,
            "counts": {
                "transition_denominator": len(reconciliation),
                "identity_anchored": int(class_counts.get("identity_anchored", 0)),
                "public_event_only": int(class_counts.get("public_event_only", 0)),
                "tpm_official": int(class_counts.get("tpm_official", 0)),
                "direct_evidence_preserved": int(reconciliation["direct_evidence_preserved"].sum()),
                "event_segments_admitted_candidate": len(admitted),
                "unresolved_candidates": unresolved,
                "later_delisted_successors": int(reconciliation["successor_later_delisted"].sum()),
                "planned_official_sources": len(state),
                "completed_official_sources": completed,
                "pending_official_sources": pending,
                "failed_official_sources": failed,
            },
            "expected_evidence_class_counts": expected,
            "denominator_valid": denominator_valid,
            "critical_blockers": blockers or ["independent_validation_required_before_promotion"],
            "parse_failure_request_ids": parse_failures,
            "input_manifests": {
                "quote_chain_candidates": str(quote_path.parent / "manifest.json"),
                "archive_source_manifest": str(archive_path.parent / "manifest.json"),
                "delisting_events": str(delisting_path.parent / "manifest.json"),
            },
            "input_manifest_sha256": {
                "quote_chain_candidates": _sha256(quote_path.parent / "manifest.json"),
                "archive_source_manifest": _sha256(archive_path.parent / "manifest.json"),
                "delisting_events": _sha256(delisting_path.parent / "manifest.json"),
            },
            "input_file_sha256": {
                "quote_chain_candidates.csv": _sha256(quote_path),
                "archive_source_manifest.csv": _sha256(archive_path),
                "delisting_events.csv": _sha256(delisting_path),
            },
            "artifacts": {
                "official_source_plan": str(plan_path),
                "page_state": str(state_path),
                "technical_listing_events": str(listing_events_path),
                "transition_reconciliation": str(reconciliation_path),
                "admitted_transition_candidates": str(admitted_path),
            },
            "output_sha256": {
                path.name: _sha256(path)
                for path in (
                    plan_path,
                    state_path,
                    listing_events_path,
                    reconciliation_path,
                    admitted_path,
                )
            },
            "promotion_policy": {
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return JapanG1CorporateTransitionEvidenceArtifacts(
            run_id,
            root,
            manifest_path,
            reconciliation_path,
            admitted_path,
            manifest,
        )
    finally:
        lock.rmdir()


def _reconcile(
    *,
    anchored: pd.DataFrame,
    delistings: pd.DataFrame,
    listing_events: pd.DataFrame,
    supplemental: dict[str, str],
    litalico_bridge: dict[str, str],
    tpm: dict[str, str],
) -> pd.DataFrame:
    candidates: list[dict[str, str]] = []
    for row in anchored.to_dict("records"):
        candidates.append(
            {
                "evidence_class": "identity_anchored",
                "identity_row_id": row["identity_row_id"],
                "old_code": row["local_code"],
                "candidate_successor_codes": row["successor_codes"],
                "successor_primary_rics": row["successor_primary_rics"],
                "direct_primary_market_rics": row["direct_primary_market_rics"],
            }
        )
    for code in PUBLIC_EVENT_ONLY_CODES:
        candidates.append(
            {
                "evidence_class": "public_event_only",
                "identity_row_id": "",
                "old_code": code,
                "candidate_successor_codes": "",
                "successor_primary_rics": "",
                "direct_primary_market_rics": "",
            }
        )
    candidates.append(
        {
            "evidence_class": "tpm_official",
            "identity_row_id": "",
            "old_code": "2336",
            "candidate_successor_codes": "9243",
            "successor_primary_rics": "",
            "direct_primary_market_rics": "",
        }
    )

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        evidence_class = candidate["evidence_class"]
        old_code = candidate["old_code"]
        old_event: dict[str, str] = {}
        listing: dict[str, str] = {}
        reason_target = ""
        name_match_basis = ""
        name_bridge_source_url = ""
        if evidence_class == "tpm_official":
            old_event = {
                "event_date": tpm.get("old_event_date", ""),
                "issue_name": tpm.get("old_issue_name", ""),
                "reason": tpm.get("old_event_reason", ""),
                "source_url": tpm.get("old_event_source_url", TPM_DELISTING_URL),
            }
            listing = {
                "successor_code": tpm.get("successor_code", ""),
                "successor_issue_name": tpm.get("successor_issue_name", ""),
                "official_listing_date": tpm.get("official_listing_date", ""),
                "listing_source_url": tpm.get("listing_source_url", TPM_OUTLINE_URL),
            }
            reason_target = listing.get("successor_issue_name", "")
            name_match_basis = "tpm_pdf_explicit_stock_transfer" if tpm else ""
        elif old_code == "3398":
            old_event = {
                "event_date": supplemental.get("old_event_date", ""),
                "issue_name": supplemental.get("old_issue_name", ""),
                "reason": supplemental.get("old_event_reason", ""),
                "source_url": supplemental.get("source_url", SUPPLEMENTAL_URL),
            }
            listing = _one_listing(listing_events, "3549")
            reason_target = supplemental.get("successor_issue_name", "")
            name_match_basis = "supplemental_page_explicit_exclusion_inclusion" if supplemental else ""
        else:
            matching_old = delistings.loc[delistings["code"].eq(old_code)]
            if len(matching_old) == 1:
                old_event = matching_old.iloc[0].to_dict()
            reason_target = _successor_name_from_reason(old_event.get("reason", ""))
            if evidence_class == "identity_anchored":
                codes = [value for value in candidate["candidate_successor_codes"].split("|") if value]
                listing = _one_listing(listing_events, codes[0]) if len(codes) == 1 else {}
            else:
                listing, name_match_basis = _discover_listing(
                    listing_events, old_event, reason_target
                )

        successor_code = listing.get("successor_code", "")
        listing_name = listing.get("successor_issue_name", "")
        if not name_match_basis and reason_target and listing_name:
            if canonicalize_jpx_name(reason_target) == canonicalize_jpx_name(listing_name):
                name_match_basis = "reason_target_exact_canonical"
        if old_code == "6187" and litalico_bridge:
            bridge_valid = (
                litalico_bridge.get("old_code") == old_code
                and litalico_bridge.get("successor_code") == successor_code
                and litalico_bridge.get("name_change_effective_date") == listing.get("official_listing_date", "")
                and canonicalize_jpx_name(reason_target)
                == canonicalize_jpx_name(litalico_bridge.get("successor_pre_change_name", ""))
                and canonicalize_jpx_name(listing_name)
                == canonicalize_jpx_name(litalico_bridge.get("successor_listing_name", ""))
            )
            if bridge_valid:
                name_match_basis = "supplemental_litalico_name_change_exact"
                name_bridge_source_url = litalico_bridge.get("source_url", "")
        exact_name_match = bool(name_match_basis)
        old_date = old_event.get("event_date", "")
        listing_date = listing.get("official_listing_date", "")
        gap_days = _date_gap(old_date, listing_date)
        gap_valid = gap_days is not None and 2 <= gap_days <= 6
        candidate_codes = candidate["candidate_successor_codes"].split("|")
        exact_code_match = (
            bool(successor_code)
            and (
                evidence_class == "public_event_only"
                or successor_code in candidate_codes
            )
        )
        admitted = bool(old_event and listing and exact_code_match and exact_name_match and gap_valid)
        later = _later_delisting(delistings, successor_code, listing_date)
        rows.append(
            {
                "evidence_class": evidence_class,
                "identity_row_id": candidate["identity_row_id"],
                "old_code": old_code,
                "old_issue_name": old_event.get("issue_name", ""),
                "old_event_date": old_date,
                "old_event_reason": old_event.get("reason", ""),
                "old_event_source_url": old_event.get("source_url", ""),
                "reason_target_name": reason_target,
                "successor_code": successor_code,
                "successor_issue_name": listing_name,
                "successor_primary_rics": candidate["successor_primary_rics"],
                "official_listing_date": listing_date,
                "listing_source_url": listing.get("listing_source_url", ""),
                "listing_publication_date": listing.get("listing_publication_date", ""),
                "name_match_basis": name_match_basis,
                "name_bridge_source_url": name_bridge_source_url,
                "exact_name_match": exact_name_match,
                "exact_code_match": exact_code_match,
                "gap_days": gap_days if gap_days is not None else "",
                "gap_rule_valid_2_to_6_days": gap_valid,
                "old_segment_end": old_date,
                "successor_segment_start": listing_date,
                "non_overlap_gap_preserved": gap_valid,
                "direct_primary_market_rics": candidate["direct_primary_market_rics"],
                "direct_evidence_preserved": bool(candidate["direct_primary_market_rics"]),
                "successor_later_delisted": bool(later),
                "successor_later_delisting_date": later.get("event_date", ""),
                "successor_later_delisting_reason": later.get("reason", ""),
                "successor_later_delisting_source_url": later.get("source_url", ""),
                "event_segment_admitted": admitted,
                "candidate_admission_status": (
                    "admitted_candidate_requires_independent_validation"
                    if admitted
                    else "unresolved_exact_evidence_rules"
                ),
                "transition_promoted": False,
                "scorecard_eligible": False,
                "independent_validation_required": True,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["evidence_class", "old_code"], kind="stable"
    ).reset_index(drop=True)


def _discover_listing(
    listing_events: pd.DataFrame, old_event: dict[str, str], reason_target: str
) -> tuple[dict[str, str], str]:
    if listing_events.empty or not old_event:
        return {}, ""
    old_date = old_event.get("event_date", "")
    matches: list[tuple[dict[str, str], str]] = []
    for row in listing_events.to_dict("records"):
        gap = _date_gap(old_date, row.get("official_listing_date", ""))
        if gap is None or not 2 <= gap <= 6:
            continue
        listing_name = row.get("successor_issue_name", "")
        if reason_target and canonicalize_jpx_name(reason_target) == canonicalize_jpx_name(listing_name):
            matches.append((row, "reason_target_exact_canonical"))
        elif canonicalize_jpx_name(old_event.get("issue_name", "")) == canonicalize_jpx_name(listing_name):
            matches.append((row, "old_issue_name_exact_canonical"))
    return matches[0] if len(matches) == 1 else ({}, "")


def _one_listing(listing_events: pd.DataFrame, code: str) -> dict[str, str]:
    if listing_events.empty:
        return {}
    matches = listing_events.loc[listing_events["successor_code"].eq(code)]
    return matches.iloc[0].to_dict() if len(matches) == 1 else {}


def _later_delisting(
    delistings: pd.DataFrame, successor_code: str, listing_date: str
) -> dict[str, str]:
    if not successor_code or not listing_date:
        return {}
    rows = delistings.loc[
        delistings["code"].eq(successor_code)
        & delistings["event_date"].gt(listing_date)
    ].sort_values("event_date")
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _successor_name_from_reason(reason: str) -> str:
    value = re.sub(
        r"\s*\(\s*(?:Stock Transfer|Stock Swap)\s*\)\s*$", "", str(reason), flags=re.I
    )
    match = re.fullmatch(
        r"(?:Becoming a wholly owned subsidiary of|Merging with)\s+(.+)", value, re.I
    )
    return match.group(1).strip() if match else ""


def _strip_notice_stock_suffix(value: str) -> str:
    """Remove only JPX's terminal issue-label suffix, including legacy no-space HTML."""
    return re.sub(r"\.?\s*stocks?$", "", str(value), flags=re.I).strip()


def _fetch_plan(
    plan: pd.DataFrame,
    *,
    state_path: Path,
    raw_dir: Path,
    max_pages: int,
    sleep_seconds: float,
    fetcher: Fetcher,
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = (
        pd.read_csv(state_path, dtype=str).fillna("").set_index("request_id").to_dict("index")
        if state_path.is_file()
        else {}
    )
    rows: list[dict[str, Any]] = []
    new_requests = 0
    for item in plan.to_dict("records"):
        prior = existing.get(item["request_id"], {})
        prior_path = Path(str(prior.get("raw_path", "")))
        if (
            prior.get("terminal_status") == "success"
            and prior.get("url") == item["url"]
            and prior_path.is_file()
            and _sha256(prior_path) == prior.get("sha256", "")
        ):
            rows.append({**item, **prior})
            continue
        extension = (
            ".pdf"
            if item["source_kind"].endswith("_pdf") or str(item["url"]).lower().endswith(".pdf")
            else ".html"
        )
        raw_path = raw_dir / (
            f"{item['request_id']}_{hashlib.sha256(item['url'].encode()).hexdigest()[:12]}{extension}"
        )
        if max_pages and new_requests >= max_pages:
            rows.append(
                {
                    **item,
                    "http_status": 0,
                    "terminal_status": "pending",
                    "content_type": "",
                    "raw_path": "",
                    "sha256": "",
                    "bytes": 0,
                    "error": "bounded_run_limit",
                }
            )
            continue
        try:
            status, body, headers = fetcher(item["url"])
            new_requests += 1
            if status == 200 and body:
                _atomic_bytes(raw_path, body)
                terminal, error = "success", ""
            else:
                terminal, error = "retryable_failure", f"http_status={status}"
        except Exception as exc:
            status, headers = 0, {}
            terminal = "retryable_failure"
            error = f"{type(exc).__name__}:{exc}"[:1000]
        rows.append(
            {
                **item,
                "http_status": status,
                "terminal_status": terminal,
                "content_type": headers.get("Content-Type", headers.get("content-type", "")),
                "raw_path": str(raw_path) if raw_path.is_file() else "",
                "sha256": _sha256(raw_path) if raw_path.is_file() else "",
                "bytes": raw_path.stat().st_size if raw_path.is_file() else 0,
                "error": error,
            }
        )
        if new_requests and new_requests % 25 == 0:
            _atomic_csv(state_path, pd.DataFrame(rows))
        if sleep_seconds and new_requests:
            time.sleep(sleep_seconds)
    state = pd.DataFrame(rows)
    _atomic_csv(state_path, state)
    return state


def _parse_special(
    successes: pd.DataFrame,
    source_kind: str,
    parser: Callable[..., dict[str, str]],
) -> dict[str, str]:
    matches = successes.loc[successes["source_kind"].eq(source_kind)]
    if len(matches) != 1:
        return {}
    row = matches.iloc[0]
    return parser(Path(row["raw_path"]).read_bytes(), source_url=row["url"])


def _special_bytes(successes: pd.DataFrame, source_kind: str) -> bytes:
    matches = successes.loc[successes["source_kind"].eq(source_kind)]
    return Path(matches.iloc[0]["raw_path"]).read_bytes() if len(matches) == 1 else b""


def _listing_dates_by_code(
    text: str, issue_rows: list[tuple[str, str]], publication_date: str
) -> dict[str, str]:
    published = date.fromisoformat(publication_date)
    result: dict[str, str] = {}
    date_pattern = _date_pattern()
    for code, _ in issue_rows:
        match = re.search(
            rf"\(Code\s*:\s*{re.escape(code)}\).*?is scheduled for\s+({date_pattern})",
            text,
            re.I,
        )
        if match:
            result[code] = _parse_english_date(match.group(1), published)
    counted = re.search(
        rf"initial listing of\s+(\d+)\s+stocks?\s+is scheduled for\s+({date_pattern})\s*"
        rf"\((\d+)\s+stocks?\s+is scheduled for\s+({date_pattern})\)",
        text,
        re.I,
    )
    if counted:
        first_count = int(counted.group(1))
        first_date = _parse_english_date(counted.group(2), published)
        second_date = _parse_english_date(counted.group(4), published)
        for index, (code, _) in enumerate(issue_rows):
            result[code] = first_date if index < first_count else second_date
        return result
    general_patterns = (
        rf"scheduled for (?:initial listing|listing) on\s+({date_pattern})",
        rf"scheduled for\s+({date_pattern})",
        rf"newly listed stocks? on\s+({date_pattern})",
    )
    general = ""
    for pattern in general_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            general = _parse_english_date(match.group(1), published)
            break
    for code, _ in issue_rows:
        result.setdefault(code, general)
    return result


def _table_identity_after_heading(area: Any, heading_text: str) -> tuple[str, str] | None:
    heading = next(
        (node for node in area.find_all(["h2", "h3", "h4"]) if heading_text.lower() in node.get_text(" ", strip=True).lower()),
        None,
    )
    table = heading.find_next("table") if heading else None
    if not table:
        return None
    for row in table.find_all("tr"):
        cells = [" ".join(cell.stripped_strings) for cell in row.find_all(["td", "th"])]
        if len(cells) >= 2 and re.fullmatch(r"[0-9A-Z]{4}", cells[1]):
            return cells[0], cells[1].upper()
    return None


def _first_official_date(text: str, pattern: str, publication_date: str) -> str:
    rendered = pattern.format(date=_date_pattern())
    match = re.search(rendered, text, re.I)
    return _parse_english_date(match.group(1), date.fromisoformat(publication_date)) if match else ""


def _date_pattern() -> str:
    return r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2},?(?:\s+\d{4})?"


def _parse_english_date(value: str, published: date) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).replace(",", " ").split())
    match = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2})(?:\s+(\d{4}))?", normalized)
    if not match:
        return ""
    month = datetime.strptime(match.group(1)[:3].title(), "%b").month
    year = int(match.group(3)) if match.group(3) else published.year + int(month < published.month)
    return date(year, month, int(match.group(2))).isoformat()


def _date_gap(old_date: str, listing_date: str) -> int | None:
    try:
        return (date.fromisoformat(listing_date) - date.fromisoformat(old_date)).days
    except ValueError:
        return None


def _verify_manifest_output(path: Path, hash_field: str) -> None:
    manifest_path = path.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"upstream manifest is unreadable: {manifest_path}") from exc
    expected = manifest.get(hash_field, {}).get(path.name, "")
    if not path.is_file() or not expected or _sha256(path) != str(expected).lower():
        raise ValueError(f"upstream output hash mismatch: {path}")


def _walk_news_items(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if "title" in value and "url" in value and "updated_date" in value:
            rows.append(value)
        for child in value.values():
            rows.extend(_walk_news_items(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_walk_news_items(child))
    return rows


def _publication_date(item: dict[str, Any]) -> date | None:
    value = item.get("updated_date", {})
    try:
        return date(int(value["year"]), int(value["month"]), int(value["day"]))
    except (KeyError, TypeError, ValueError):
        return None


def _fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "quant-lab-jpx-corporate-transition-evidence/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(32 * 1024 * 1024 + 1)
            if len(body) > 32 * 1024 * 1024:
                raise ValueError("official source exceeds 32 MiB limit")
            return int(response.status), body, dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
