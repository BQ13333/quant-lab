from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
REGIME_CHANGE_DATE = date(2020, 8, 3)
SOURCE_RUN_ID = "JPN-PRICE-LIMIT-EVENTS-201309-202606-V1"
RECONCILIATION_RUN_ID = (
    "JPN-PRICE-LIMIT-BILINGUAL-RECONCILIATION-201309-202606-V1"
)
SOURCE_OUTPUTS = (
    "notice_plan.csv",
    "page_state.csv",
    "price_limit_events.csv",
    "event_quality.csv",
)
RECONCILIATION_OUTPUTS = (
    "counterpart_fetch_state.csv",
    "reconciliation.csv",
    "corrected_candidate_events.csv",
)
REQUIRED_COMPLETION_BLOCKERS = (
    "official issue-specific price-limit archive before 2013-09 is not yet proven complete",
    "post-2020 return-to-normal dates require admitted daily price/calendar evidence because JPX discontinued removal notices",
)
PROMOTION_POLICY = {
    "alpha": False,
    "historical_backtest": False,
    "production": False,
    "ibkr": False,
    "live_trading": False,
}
EVENT_KEY = ("publication_date", "event_type", "code")
NUMERIC_FIELDS = (
    "base_price_yen",
    "expanded_range_yen",
    "normal_range_yen",
    "observed_multiplier",
)
DETAIL_COLUMNS = (
    "publication_date",
    "event_type",
    "code",
    "source_url",
    "producer_side",
    "source_explicit_side",
    "chain_inferred_side",
    "resolved_side",
    "producer_base_price_yen",
    "producer_expanded_range_yen",
    "producer_normal_range_yen",
    "producer_observed_multiplier",
    "validated_base_price_yen",
    "validated_expanded_range_yen",
    "validated_normal_range_yen",
    "validated_observed_multiplier",
    "authoritative_language",
    "overlay_applied",
    "source_numeric_discrepancy",
    "post_overlay_numeric_discrepancy",
    "post_overlay_regime_discrepancy",
    "expected_multiplier",
)


@dataclass(frozen=True)
class JapanPriceLimitBilingualValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_japan_price_limit_bilingual_validation(
    *,
    run_id: str,
    producer_dir: str | Path,
    reconciliation_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_price_limit_bilingual_validation",
) -> JapanPriceLimitBilingualValidationArtifacts:
    """Validate immutable English/Japanese price-limit artifacts without producer helpers."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")

    producer = Path(producer_dir).resolve()
    reconciliation_root = Path(reconciliation_dir).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    if root in {producer, reconciliation_root} or _is_relative_to(root, producer) or _is_relative_to(
        root, reconciliation_root
    ):
        raise ValueError("validation output must be separate from immutable inputs")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc

    try:
        issues: list[str] = []
        summary_rows: list[dict[str, Any]] = []

        def check(name: str, expected: Any, observed: Any, issue: str) -> bool:
            passed = _same_value(expected, observed)
            summary_rows.append(
                {
                    "check": name,
                    "expected": _display(expected),
                    "observed": _display(observed),
                    "passed": passed,
                }
            )
            if not passed:
                issues.append(issue)
            return passed

        source_manifest_path = producer / "manifest.json"
        overlay_manifest_path = reconciliation_root / "manifest.json"
        source_manifest = _read_json(source_manifest_path, issues, "source_manifest")
        overlay_manifest = _read_json(overlay_manifest_path, issues, "reconciliation_manifest")

        check(
            "independent_run_id",
            False,
            run_id in {SOURCE_RUN_ID, RECONCILIATION_RUN_ID},
            "independence_failure_same_run_id",
        )
        check("source_run_id", SOURCE_RUN_ID, source_manifest.get("run_id"), "source_run_id_mismatch")
        check(
            "reconciliation_run_id",
            RECONCILIATION_RUN_ID,
            overlay_manifest.get("run_id"),
            "reconciliation_run_id_mismatch",
        )
        check(
            "source_artifact_type",
            "japan_exceptional_price_limit_event_history_candidate",
            source_manifest.get("artifact_type"),
            "source_artifact_type_invalid",
        )
        check(
            "reconciliation_artifact_type",
            "japan_exceptional_price_limit_bilingual_reconciliation_candidate",
            overlay_manifest.get("artifact_type"),
            "reconciliation_artifact_type_invalid",
        )
        check(
            "overlay_source_manifest_hash",
            _sha256(source_manifest_path) if source_manifest_path.is_file() else "",
            overlay_manifest.get("source_manifest_sha256", ""),
            "overlay_source_manifest_hash_mismatch",
        )

        source_hashes_valid = _validate_declared_outputs(
            producer,
            source_manifest.get("output_sha256"),
            SOURCE_OUTPUTS,
            issues,
            "source",
        )
        overlay_source_hashes = overlay_manifest.get("source_output_sha256")
        source_hashes_repeated = isinstance(overlay_source_hashes, dict) and all(
            overlay_source_hashes.get(name) == source_manifest.get("output_sha256", {}).get(name)
            for name in SOURCE_OUTPUTS
        )
        check(
            "source_output_hashes",
            len(SOURCE_OUTPUTS),
            source_hashes_valid,
            "source_output_hash_denominator_invalid",
        )
        check("overlay_repeats_source_hashes", True, source_hashes_repeated, "overlay_source_output_hashes_mismatch")
        overlay_hashes_valid = _validate_declared_outputs(
            reconciliation_root,
            overlay_manifest.get("output_sha256"),
            RECONCILIATION_OUTPUTS,
            issues,
            "reconciliation",
        )
        check(
            "reconciliation_output_hashes",
            len(RECONCILIATION_OUTPUTS),
            overlay_hashes_valid,
            "reconciliation_output_hash_denominator_invalid",
        )

        source_blockers = set(source_manifest.get("remaining_completion_blockers", []))
        overlay_blockers = set(overlay_manifest.get("remaining_completion_blockers", []))
        check(
            "source_completion_blockers_retained",
            True,
            set(REQUIRED_COMPLETION_BLOCKERS).issubset(source_blockers),
            "source_completion_blockers_missing",
        )
        check(
            "overlay_completion_blockers_retained",
            True,
            set(REQUIRED_COMPLETION_BLOCKERS).issubset(overlay_blockers),
            "overlay_completion_blockers_missing",
        )
        source_policy_valid = _non_promotable_contract(source_manifest)
        overlay_policy_valid = _non_promotable_contract(overlay_manifest)
        check("source_non_promotable_contract", True, source_policy_valid, "source_non_promotable_contract_invalid")
        check(
            "overlay_non_promotable_contract",
            True,
            overlay_policy_valid,
            "overlay_non_promotable_contract_invalid",
        )

        page_state = _read_csv(producer / "page_state.csv", issues, "page_state")
        events = _read_csv(
            producer / "price_limit_events.csv", issues, "price_limit_events", dtype={"code": str}
        )
        quality = _read_csv(
            producer / "event_quality.csv", issues, "event_quality", dtype={"code": str}
        )
        counterpart_state = _read_csv(
            reconciliation_root / "counterpart_fetch_state.csv", issues, "counterpart_fetch_state"
        )
        reconciliation = _read_csv(
            reconciliation_root / "reconciliation.csv", issues, "reconciliation", dtype={"code": str}
        )
        corrections = _read_csv(
            reconciliation_root / "corrected_candidate_events.csv",
            issues,
            "corrected_candidate_events",
            dtype={"code": str},
        )

        valid_english_pages = _validate_raw_state(
            page_state,
            id_column="plan_row_id",
            url_column="url",
            issues=issues,
            issue_prefix="english_raw",
        )
        check(
            "english_raw_page_hashes",
            len(page_state),
            valid_english_pages,
            "english_raw_page_hash_denominator_invalid",
        )
        valid_japanese_pages = _validate_raw_state(
            counterpart_state,
            id_column="counterpart_id",
            url_column="japanese_source_url",
            issues=issues,
            issue_prefix="japanese_raw",
        )
        check(
            "japanese_raw_page_hashes",
            len(counterpart_state),
            valid_japanese_pages,
            "japanese_raw_page_hash_denominator_invalid",
        )

        english_rows, english_parse_issues = _parse_english_pages(page_state)
        issues.extend(english_parse_issues)
        english_rows = _add_chain_inferred_sides(english_rows)
        check(
            "independently_parsed_english_events",
            len(events),
            len(english_rows),
            "independent_english_event_denominator_mismatch",
        )
        check(
            "independent_english_event_keys",
            _key_set(events),
            _key_set(english_rows),
            "independent_english_event_key_mismatch",
        )

        japanese_rows, japanese_parse_issues = _parse_japanese_pages(counterpart_state)
        issues.extend(japanese_parse_issues)
        check(
            "japanese_pages_yield_rows",
            True,
            len(japanese_rows) >= len(counterpart_state),
            "independent_japanese_page_coverage_invalid",
        )

        source_field_issues = _source_field_issues(english_rows, events)
        issues.extend(source_field_issues)
        check("source_event_field_discrepancies", 0, len(source_field_issues), "source_event_field_validation_failed")

        correction_issues, correction_proof = _prove_6557_correction(
            events=events,
            quality=quality,
            reconciliation=reconciliation,
            corrections=corrections,
            english_rows=english_rows,
            japanese_rows=japanese_rows,
            page_state=page_state,
            reconciliation_root=reconciliation_root,
        )
        issues.extend(correction_issues)
        check("single_6557_correction_proven", 0, len(correction_issues), "single_6557_correction_not_proven")

        detail, recompute_issues = _apply_overlay_and_recompute(
            events=events,
            english_rows=english_rows,
            japanese_rows=japanese_rows,
            corrections=corrections,
        )
        issues.extend(recompute_issues)
        post_numeric = int(detail["post_overlay_numeric_discrepancy"].sum()) if not detail.empty else 0
        post_regime = int(detail["post_overlay_regime_discrepancy"].sum()) if not detail.empty else 0
        overlay_count = int(detail["overlay_applied"].sum()) if not detail.empty else 0
        check("overlay_applied_rows", 1, overlay_count, "overlay_application_denominator_invalid")
        check("post_overlay_numeric_discrepancies", 0, post_numeric, "unresolved_post_overlay_numeric_discrepancies")
        check("post_overlay_regime_discrepancies", 0, post_regime, "unresolved_post_overlay_regime_discrepancies")

        issues = list(dict.fromkeys(issues))
        summary = pd.DataFrame(summary_rows, columns=["check", "expected", "observed", "passed"])
        summary_path = root / "validation_summary.csv"
        detail_path = root / "event_validation_detail.csv"
        _atomic_csv(summary_path, summary)
        _atomic_csv(detail_path, detail)
        side_blank_mask = detail["producer_side"].eq("") if not detail.empty else pd.Series(dtype=bool)
        inference_mask = detail["chain_inferred_side"].ne("") if not detail.empty else pd.Series(dtype=bool)
        manifest = {
            "schema_version": "1.0-independent-immutable-partial",
            "run_id": run_id,
            "validator_run_id": run_id,
            "source_run_id": source_manifest.get("run_id"),
            "reconciliation_run_id": overlay_manifest.get("run_id"),
            "artifact_type": "japan_exceptional_price_limit_bilingual_reconciliation_independent_validation",
            "status": "pass_partial_component" if not issues else "failed_validation",
            "component": "temporary_price_limit_events",
            "independent_validation": True,
            "clean_room_parsers": True,
            "network_accessed": False,
            "source_producer_immutable": True,
            "reconciliation_overlay_immutable": True,
            "atomic_writes": True,
            "official_jpx_bilingual_sources": True,
            "research_only": True,
            "promotion_allowed": False,
            "scorecard_eligible": False,
            "component_scorecard_eligible": False,
            "validation_issues": issues,
            "remaining_completion_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
            "critical_blockers": [*issues, *REQUIRED_COMPLETION_BLOCKERS],
            "counts": {
                "source_event_rows": len(events),
                "independently_parsed_english_event_rows": len(english_rows),
                "independently_parsed_japanese_rows": len(japanese_rows),
                "valid_english_raw_pages": valid_english_pages,
                "valid_japanese_raw_pages": valid_japanese_pages,
                "producer_blank_side_rows": int(side_blank_mask.sum()),
                "chain_inferred_side_rows": int(inference_mask.sum()),
                "proven_correction_rows": 1 if not correction_issues else 0,
                "overlay_applied_rows": overlay_count,
                "unresolved_numeric_discrepancies": post_numeric,
                "unresolved_regime_discrepancies": post_regime,
            },
            "correction_proof": correction_proof,
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path) if source_manifest_path.is_file() else "",
            "reconciliation_manifest": str(overlay_manifest_path),
            "reconciliation_manifest_sha256": _sha256(overlay_manifest_path)
            if overlay_manifest_path.is_file()
            else "",
            "artifacts": {
                "validation_summary": str(summary_path),
                "event_validation_detail": str(detail_path),
            },
            "output_sha256": {
                summary_path.name: _sha256(summary_path),
                detail_path.name: _sha256(detail_path),
            },
            "promotion_policy": dict(PROMOTION_POLICY),
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return JapanPriceLimitBilingualValidationArtifacts(run_id, root, manifest_path, manifest)
    finally:
        lock.rmdir()


def _parse_english_pages(state: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    required = {"plan_row_id", "publication_date", "event_type", "url", "title", "terminal_status", "raw_path"}
    if not required.issubset(state.columns):
        return pd.DataFrame(), ["english_page_state_schema_invalid_for_parse"]
    for item in state.loc[state["terminal_status"].eq("success")].to_dict("records"):
        path = Path(str(item["raw_path"]))
        if not path.is_file():
            continue
        parsed = _parse_english_notice(
            path.read_bytes(),
            publication_date=str(item["publication_date"]),
            event_type=str(item["event_type"]),
            source_url=str(item["url"]),
            title=str(item["title"]),
        )
        if not parsed:
            issues.append(f"independent_english_page_parse_failure:{item['plan_row_id']}")
        rows.extend(parsed)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates([*EVENT_KEY, "source_url"])
        frame = frame.sort_values(["publication_date", "code", "event_type", "source_url"]).reset_index(drop=True)
    return frame, issues


def _parse_english_notice(
    body: bytes,
    *,
    publication_date: str,
    event_type: str,
    source_url: str,
    title: str,
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    paragraphs = [_normalize(" ".join(node.stripped_strings)) for node in soup.find_all("p")]
    text = " ".join(value for value in paragraphs if value) or _normalize(soup.get_text(" ", strip=True))
    matches = list(
        re.finditer(
            r"(?:\(\s*)?Code\s*[:：]\s*((?=[0-9A-Z]{4}\b)(?=[0-9A-Z]*\d)[0-9A-Z]{4})(?:\s*\))?",
            text,
            re.I,
        )
    )
    unique: list[re.Match[str]] = []
    seen: set[str] = set()
    for match in matches:
        code = match.group(1).upper()
        if code not in seen:
            unique.append(match)
            seen.add(code)
    archive_date = _date_value(publication_date)
    published = _url_date(source_url) or archive_date
    if published is None:
        return []
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(unique):
        end = unique[index + 1].start() if index + 1 < len(unique) else len(text)
        segment = text[match.end() : end]
        context = segment if len(unique) > 1 else text
        side_context = text[max(0, match.start() - 300) : end] if len(unique) > 1 else text
        limits = re.search(
            r"\b(Upper|Lower) daily price limit\s*:\s*([\d,]+)\s*yen\s*"
            r"\((?:Upper|Lower) limit will remain\s*([\d,]+)\s*yen\s*as per normal\)",
            segment,
            re.I,
        )
        if limits:
            side = limits.group(1).upper()
            expanded = _integer(limits.group(2))
            normal = _integer(limits.group(3))
        else:
            side = _explicit_english_side(side_context)
            expanded = _extract_integer(context, r"(?:Upper|Lower) daily price limit\s*:\s*([\d,]+)\s*yen\s*\(")
            normal = _extract_integer(context, r"(?:Upper|Lower) limit will remain\s*([\d,]+)\s*yen")
        base = _extract_integer(context, r"Base price\s*:\s*([\d,]+)\s*yen")
        multiplier = expanded / normal if expanded is not None and normal else _text_multiplier(context)
        effective = _effective_date(text, published)
        rows.append(
            {
                "publication_date": published.isoformat(),
                "archive_publication_date": archive_date.isoformat() if archive_date else "",
                "effective_date": effective.isoformat(),
                "event_type": event_type,
                "code": match.group(1).upper(),
                "source_explicit_side": side,
                "base_price_yen": base,
                "expanded_range_yen": expanded,
                "normal_range_yen": normal,
                "observed_multiplier": multiplier,
                "source_url": source_url,
                "title": title,
            }
        )
    return rows


def _parse_japanese_pages(state: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    required = {"counterpart_id", "japanese_source_url", "terminal_status", "raw_path"}
    if not required.issubset(state.columns):
        return pd.DataFrame(), ["japanese_page_state_schema_invalid_for_parse"]
    for item in state.loc[state["terminal_status"].eq("success")].to_dict("records"):
        path = Path(str(item["raw_path"]))
        if not path.is_file():
            continue
        parsed = _parse_japanese_notice(path.read_bytes(), source_url=str(item["japanese_source_url"]))
        if not parsed:
            issues.append(f"independent_japanese_page_parse_failure:{item['counterpart_id']}")
        rows.extend(parsed)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(["publication_date", "code", "source_url"])
        frame = frame.sort_values(["publication_date", "code", "source_url"]).reset_index(drop=True)
    return frame, issues


def _parse_japanese_notice(body: bytes, *, source_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    text = _normalize(soup.get_text(" ", strip=True))
    published = _url_date(source_url) or _japanese_page_date(soup, text)
    matches = list(re.finditer(r"(?:銘柄)?コード\s*[:：]?\s*([0-9A-Z]{4})", text, re.I))
    rows: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.end() : end]
        limits = re.search(
            r"制限値幅\s*\(\s*(上限|下限)\s*\)\s*:\s*([\d,]+)\s*円\s*"
            r"\(\s*(?:上限|下限)は通常(?:どお|通)り\s*([\d,]+)\s*円\s*\)",
            segment,
        )
        expanded = _integer(limits.group(2)) if limits else None
        normal = _integer(limits.group(3)) if limits else None
        base = _extract_integer(segment, r"基準値段\s*:\s*([\d,]+)\s*円")
        rows.append(
            {
                "publication_date": published.isoformat() if published else "",
                "code": match.group(1).upper(),
                "source_explicit_side": {"上限": "UPPER", "下限": "LOWER"}.get(
                    limits.group(1), ""
                )
                if limits
                else "",
                "base_price_yen": base,
                "expanded_range_yen": expanded,
                "normal_range_yen": normal,
                "observed_multiplier": expanded / normal if expanded is not None and normal else None,
                "source_url": source_url,
            }
        )
    return rows


def _add_chain_inferred_sides(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.assign(chain_inferred_side=pd.Series(dtype=str), resolved_side=pd.Series(dtype=str))
    frame = events.copy()
    frame["chain_inferred_side"] = ""
    for _, indexes in frame.sort_values(["publication_date", "effective_date", "source_url"]).groupby(
        "code", sort=False
    ).groups.items():
        active_side = ""
        for index in indexes:
            family = str(frame.at[index, "event_type"])
            explicit = str(frame.at[index, "source_explicit_side"])
            if explicit and family in {"PRELIMINARY", "BROADENING", "CONTINUATION"}:
                active_side = explicit
            elif not explicit and family in {"CANCELLED", "RETURN_NORMAL"} and active_side:
                frame.at[index, "chain_inferred_side"] = active_side
            if family in {"CANCELLED", "RETURN_NORMAL"}:
                active_side = ""
    frame["resolved_side"] = frame["source_explicit_side"].where(
        frame["source_explicit_side"].ne(""), frame["chain_inferred_side"]
    )
    return frame


def _source_field_issues(expected: pd.DataFrame, observed: pd.DataFrame) -> list[str]:
    required = {*EVENT_KEY, "source_url", "source_explicit_side", *NUMERIC_FIELDS}
    if not required.issubset(expected.columns) or not {*EVENT_KEY, "source_url", "side", *NUMERIC_FIELDS}.issubset(
        observed.columns
    ):
        return ["source_event_schema_invalid"]
    expected_by_key = {_row_key(row): row for row in expected.to_dict("records")}
    observed_by_key = {_row_key(row): row for row in observed.to_dict("records")}
    issues: list[str] = []
    for key in sorted(set(expected_by_key) & set(observed_by_key)):
        parsed = expected_by_key[key]
        producer = observed_by_key[key]
        label = ":".join(key)
        if str(parsed["source_url"]) != str(producer["source_url"]):
            issues.append(f"source_url_mismatch:{label}")
        family = str(producer["event_type"])
        parsed_side = str(parsed.get("source_explicit_side", ""))
        producer_side = str(producer.get("side", "")).upper()
        if family in {"BROADENING", "CONTINUATION"}:
            if not parsed_side or producer_side != parsed_side:
                issues.append(f"source_explicit_side_mismatch:{label}")
        elif producer_side and parsed_side and producer_side != parsed_side:
            issues.append(f"source_side_mismatch:{label}")
        for field in NUMERIC_FIELDS:
            if not _numeric_equal(parsed.get(field), producer.get(field)):
                issues.append(f"source_numeric_mismatch:{field}:{label}")
    return issues


def _prove_6557_correction(
    *,
    events: pd.DataFrame,
    quality: pd.DataFrame,
    reconciliation: pd.DataFrame,
    corrections: pd.DataFrame,
    english_rows: pd.DataFrame,
    japanese_rows: pd.DataFrame,
    page_state: pd.DataFrame,
    reconciliation_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    expected_key = ("2024-07-23", "BROADENING", "6557")
    mismatches = []
    for row in english_rows.loc[english_rows["event_type"].isin({"BROADENING", "CONTINUATION"})].to_dict(
        "records"
    ):
        published = _date_value(row.get("publication_date"))
        expected = 2.0 if published and published < REGIME_CHANGE_DATE else 4.0
        observed = _number(row.get("observed_multiplier"))
        if observed is None or not _close(observed, expected):
            mismatches.append(_row_key(row))
    if mismatches != [expected_key]:
        issues.append(f"independent_pre_overlay_mismatch_set_invalid:{mismatches}")
    if len(reconciliation) != 1 or len(corrections) != 1:
        issues.append("overlay_correction_denominator_invalid")

    english = _one_by_key(english_rows, expected_key)
    source = _one_by_key(events, expected_key)
    corrected = _one_by_key(corrections, expected_key)
    reconciled = _one_by_key(reconciliation, expected_key)
    quality_row = _one_by_key(quality, expected_key)
    japanese_matches = japanese_rows.loc[
        japanese_rows.get("publication_date", pd.Series(dtype=str)).eq(expected_key[0])
        & japanese_rows.get("code", pd.Series(dtype=str)).eq(expected_key[2])
    ]
    japanese = japanese_matches.iloc[0].to_dict() if len(japanese_matches) == 1 else {}
    if not all((english, source, corrected, reconciled, quality_row, japanese)):
        issues.append("correction_required_rows_missing_or_ambiguous")

    expected_english = {
        "source_explicit_side": "UPPER",
        "base_price_yen": 746.0,
        "expanded_range_yen": 300.0,
        "normal_range_yen": 150.0,
        "observed_multiplier": 2.0,
    }
    expected_japanese = {
        "source_explicit_side": "UPPER",
        "base_price_yen": 1620.0,
        "expanded_range_yen": 1600.0,
        "normal_range_yen": 400.0,
        "observed_multiplier": 4.0,
    }
    for field, value in expected_english.items():
        if not _field_equal(english.get(field), value):
            issues.append(f"independent_english_6557_mismatch:{field}")
    for field, value in expected_japanese.items():
        if not _field_equal(japanese.get(field), value):
            issues.append(f"independent_japanese_6557_mismatch:{field}")
    for field in NUMERIC_FIELDS:
        if not _numeric_equal(corrected.get(field), expected_japanese[field]):
            issues.append(f"corrected_6557_value_mismatch:{field}")
    if str(corrected.get("side", "")) != "UPPER":
        issues.append("corrected_6557_side_mismatch")
    if not _to_bool(reconciled.get("code_match")) or not _to_bool(reconciled.get("date_match")):
        issues.append("declared_6557_date_or_code_match_invalid")
    if not _to_bool(reconciled.get("side_match")) or not _to_bool(reconciled.get("japanese_regime_match")):
        issues.append("declared_6557_side_or_regime_match_invalid")
    if not _to_bool(reconciled.get("correction_eligible")):
        issues.append("declared_6557_correction_not_eligible")
    if _to_bool(quality_row.get("regime_multiplier_valid")) is not False:
        issues.append("source_6557_quality_flag_not_false")

    english_raw = Path(str(reconciled.get("english_raw_path", "")))
    japanese_raw = Path(str(reconciled.get("japanese_raw_path", "")))
    source_state = page_state.loc[page_state.get("url", pd.Series(dtype=str)).eq(source.get("source_url", ""))]
    retained_english_valid = (
        len(source_state) == 1
        and english_raw.is_file()
        and _is_relative_to(english_raw, reconciliation_root)
        and _sha256(english_raw) == str(reconciled.get("english_sha256", ""))
        and _sha256(english_raw) == str(source_state.iloc[0].get("sha256", ""))
    )
    retained_japanese_valid = (
        japanese_raw.is_file()
        and _is_relative_to(japanese_raw, reconciliation_root)
        and _sha256(japanese_raw) == str(reconciled.get("japanese_sha256", ""))
    )
    if not retained_english_valid:
        issues.append("retained_english_6557_hash_mismatch")
    if not retained_japanese_valid:
        issues.append("retained_japanese_6557_hash_mismatch")

    proof = {
        "publication_date": expected_key[0],
        "event_type": expected_key[1],
        "code": expected_key[2],
        "english_side": english.get("source_explicit_side", ""),
        "japanese_side": japanese.get("source_explicit_side", ""),
        "date_match": english.get("publication_date") == japanese.get("publication_date") == expected_key[0],
        "code_match": english.get("code") == japanese.get("code") == expected_key[2],
        "side_match": english.get("source_explicit_side") == japanese.get("source_explicit_side") == "UPPER",
        "japanese_base_price_yen": japanese.get("base_price_yen"),
        "japanese_expanded_range_yen": japanese.get("expanded_range_yen"),
        "japanese_normal_range_yen": japanese.get("normal_range_yen"),
        "japanese_observed_multiplier": japanese.get("observed_multiplier"),
        "retained_english_hash_valid": retained_english_valid,
        "retained_japanese_hash_valid": retained_japanese_valid,
        "proven": not issues,
    }
    return issues, proof


def _apply_overlay_and_recompute(
    *,
    events: pd.DataFrame,
    english_rows: pd.DataFrame,
    japanese_rows: pd.DataFrame,
    corrections: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    english_by_key = {_row_key(row): row for row in english_rows.to_dict("records")}
    correction_by_key = {_row_key(row): row for row in corrections.to_dict("records")}
    japanese_by_key = {
        (str(row.get("publication_date", "")), str(row.get("code", ""))): row
        for row in japanese_rows.to_dict("records")
    }
    details: list[dict[str, Any]] = []
    for producer in events.to_dict("records"):
        key = _row_key(producer)
        parsed = english_by_key.get(key, {})
        correction = correction_by_key.get(key)
        validated = dict(producer)
        authoritative = parsed
        language = "english"
        if correction is not None:
            validated.update(correction)
            authoritative = japanese_by_key.get((key[0], key[2]), {})
            language = "japanese"
            if not authoritative:
                issues.append(f"overlay_authoritative_japanese_row_missing:{':'.join(key)}")

        source_numeric = any(
            not _numeric_equal(producer.get(field), parsed.get(field)) for field in NUMERIC_FIELDS
        )
        post_numeric = any(
            not _numeric_equal(validated.get(field), authoritative.get(field)) for field in NUMERIC_FIELDS
        )
        expected_multiplier: float | None = None
        regime_discrepancy = False
        if str(producer.get("event_type")) in {"BROADENING", "CONTINUATION"}:
            published = _date_value(producer.get("publication_date"))
            expected_multiplier = 2.0 if published and published < REGIME_CHANGE_DATE else 4.0
            expanded = _number(validated.get("expanded_range_yen"))
            normal = _number(validated.get("normal_range_yen"))
            observed = _number(validated.get("observed_multiplier"))
            ratio = expanded / normal if expanded is not None and normal else None
            regime_discrepancy = not _close(ratio, expected_multiplier) or not _close(
                ratio, observed
            )
        details.append(
            {
                "publication_date": key[0],
                "event_type": key[1],
                "code": key[2],
                "source_url": producer.get("source_url", ""),
                "producer_side": str(producer.get("side", "")),
                "source_explicit_side": parsed.get("source_explicit_side", ""),
                "chain_inferred_side": parsed.get("chain_inferred_side", ""),
                "resolved_side": parsed.get("resolved_side", ""),
                **{f"producer_{field}": producer.get(field) for field in NUMERIC_FIELDS},
                **{f"validated_{field}": validated.get(field) for field in NUMERIC_FIELDS},
                "authoritative_language": language,
                "overlay_applied": correction is not None,
                "source_numeric_discrepancy": source_numeric,
                "post_overlay_numeric_discrepancy": post_numeric,
                "post_overlay_regime_discrepancy": regime_discrepancy,
                "expected_multiplier": expected_multiplier,
            }
        )
    detail = pd.DataFrame(details, columns=DETAIL_COLUMNS)
    if not detail.empty:
        detail = detail.sort_values(["publication_date", "code", "event_type"]).reset_index(drop=True)
    return detail, issues


def _validate_declared_outputs(
    root: Path,
    declared: Any,
    required: tuple[str, ...],
    issues: list[str],
    prefix: str,
) -> int:
    if not isinstance(declared, dict):
        issues.append(f"{prefix}_output_hash_map_invalid")
        return 0
    valid = 0
    for name in required:
        path = root / name
        if path.is_file() and declared.get(name) == _sha256(path):
            valid += 1
        else:
            issues.append(f"{prefix}_output_hash_mismatch:{name}")
    return valid


def _validate_raw_state(
    frame: pd.DataFrame,
    *,
    id_column: str,
    url_column: str,
    issues: list[str],
    issue_prefix: str,
) -> int:
    required = {id_column, url_column, "http_status", "terminal_status", "raw_path", "sha256", "bytes"}
    if not required.issubset(frame.columns):
        issues.append(f"{issue_prefix}_state_schema_invalid")
        return 0
    valid = 0
    for row in frame.to_dict("records"):
        path = Path(str(row.get("raw_path", "")))
        row_valid = (
            str(row.get("terminal_status")) == "success"
            and _to_int(row.get("http_status")) == 200
            and path.is_file()
            and _sha256(path) == str(row.get("sha256", ""))
            and path.stat().st_size == _to_int(row.get("bytes"))
        )
        if row_valid:
            valid += 1
        else:
            issues.append(f"{issue_prefix}_invalid:{row.get(id_column, '')}")
    return valid


def _non_promotable_contract(manifest: dict[str, Any]) -> bool:
    return (
        manifest.get("research_only") is True
        and manifest.get("scorecard_eligible") is False
        and manifest.get("component_scorecard_eligible") is False
        and manifest.get("promotion_policy") == PROMOTION_POLICY
        and manifest.get("promotion_allowed", False) is False
    )


def _read_json(path: Path, issues: list[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        issues.append(f"{label}_invalid")
        return {}
    if not isinstance(value, dict):
        issues.append(f"{label}_not_object")
        return {}
    return value


def _read_csv(
    path: Path,
    issues: list[str],
    label: str,
    *,
    dtype: dict[str, Any] | None = None,
) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False, dtype=dtype)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError):
        issues.append(f"{label}_invalid")
        return pd.DataFrame()


def _one_by_key(frame: pd.DataFrame, key: tuple[str, str, str]) -> dict[str, Any]:
    if not set(EVENT_KEY).issubset(frame.columns):
        return {}
    mask = pd.Series(True, index=frame.index)
    for column, value in zip(EVENT_KEY, key):
        mask &= frame[column].astype(str).eq(value)
    return frame.loc[mask].iloc[0].to_dict() if int(mask.sum()) == 1 else {}


def _key_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    if not set(EVENT_KEY).issubset(frame.columns):
        return set()
    return {_row_key(row) for row in frame.to_dict("records")}


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(column, "")) for column in EVENT_KEY)  # type: ignore[return-value]


def _explicit_english_side(text: str) -> str:
    patterns = (
        r"(?:broaden|expand)(?:ed|ing)?\s+(?:only\s+)?(?:its\s+|the\s+)?(upper|lower)\b",
        r"\b(upper|lower) daily price limit\b[^.]{0,100}\b(?:expand|broaden)(?:ed|ing)?\b",
        r"\b(upper|lower) daily price limit\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).upper()
    return ""


def _effective_date(text: str, published: date) -> date:
    patterns = (
        r"next business day\s*\(\s*([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?\s*\)",
        r"return the daily price limits to normal on\s+([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?",
        r"continue to (?:broaden|expand)[^.]{0,100}?\s+on\s+([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?",
        r"continue to be (?:broadened|expanded)\s+on\s+([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        try:
            month = datetime.strptime(match.group(1)[:3], "%b").month
            year = int(match.group(3)) if match.group(3) else published.year + int(month < published.month - 6)
            return date(year, month, int(match.group(2)))
        except (TypeError, ValueError):
            continue
    return published


def _japanese_page_date(soup: BeautifulSoup, text: str) -> date | None:
    for node in soup.find_all("time"):
        parsed = _date_value(str(node.get("datetime", "")) or node.get_text(" ", strip=True))
        if parsed:
            return parsed
    match = re.search(r"(20\d{2})\s*(?:年|[-/.])\s*(\d{1,2})\s*(?:月|[-/.])\s*(\d{1,2})\s*日?", text[:1200])
    if not match:
        return None
    try:
        return date(*(int(value) for value in match.groups()))
    except ValueError:
        return None


def _url_date(url: str) -> date | None:
    match = re.search(r"/(20[0-9]{6})-[0-9]+\.html(?:\?.*)?$", str(url))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _date_value(value: Any) -> date | None:
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _text_multiplier(text: str) -> float | None:
    if re.search(r"\b(?:four times|quadruple)\b", text, re.I):
        return 4.0
    if re.search(r"\b(?:twice|two times)\b", text, re.I):
        return 2.0
    return None


def _extract_integer(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, re.I)
    return _integer(match.group(1)) if match else None


def _integer(value: str) -> int:
    return int(value.replace(",", ""))


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _numeric_equal(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return left_number is right_number
    return _close(left_number, right_number)


def _field_equal(left: Any, right: Any) -> bool:
    if isinstance(right, (int, float)):
        return _numeric_equal(left, right)
    return str(left) == str(right)


def _close(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and abs(left - right) < 1e-9


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", html.unescape(value))).strip()


def _same_value(expected: Any, observed: Any) -> bool:
    if isinstance(expected, set) and isinstance(observed, set):
        return expected == observed
    return expected == observed


def _display(value: Any) -> str:
    if isinstance(value, set):
        return json.dumps(sorted(value), ensure_ascii=False)
    if isinstance(value, (dict, list, tuple, bool)) or value is None:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


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
