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
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
JPX_ROOT = "https://www.jpx.co.jp"
REGIME_CHANGE_DATE = date(2020, 8, 3)
ROLLING_ARCHIVE_START = date(2024, 9, 1)
TITLE_FAMILIES = ("PRELIMINARY", "CANCELLED", "BROADENING", "CONTINUATION", "RETURN_NORMAL")
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


@dataclass(frozen=True)
class JapanPriceLimitEventHistoryValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_japan_price_limit_event_history_validation(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_price_limit_event_history_validation",
) -> JapanPriceLimitEventHistoryValidationArtifacts:
    """Independently validate a producer bundle without calling producer logic."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")

    producer = Path(producer_dir).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    producer_manifest_path = producer / "manifest.json"
    producer_manifest = _read_json(producer_manifest_path)
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

    check("independent_run_id", False, run_id == producer_manifest.get("run_id"), "independence_failure_same_run_id")
    contract_valid = (
        producer_manifest.get("artifact_type") == "japan_exceptional_price_limit_event_history_candidate"
        and producer_manifest.get("status") == "partial_candidate"
        and producer_manifest.get("component") == "temporary_price_limit_events"
        and producer_manifest.get("official_jpx_source") is True
        and producer_manifest.get("research_only") is True
        and producer_manifest.get("scorecard_eligible") is False
        and producer_manifest.get("component_scorecard_eligible") is False
        and producer_manifest.get("independent_validation_required") is True
        and not producer_manifest.get("producer_blockers")
        and producer_manifest.get("promotion_policy") == PROMOTION_POLICY
    )
    check("producer_candidate_contract", True, contract_valid, "producer_partial_contract_invalid")

    producer_blockers = set(producer_manifest.get("remaining_completion_blockers", []))
    blockers_retained = set(REQUIRED_COMPLETION_BLOCKERS).issubset(producer_blockers)
    check("required_completion_blockers_retained", True, blockers_retained, "producer_required_blockers_missing")

    required_outputs = {"notice_plan.csv", "page_state.csv", "price_limit_events.csv", "event_quality.csv"}
    declared_hashes = producer_manifest.get("output_sha256", {})
    check(
        "required_output_hashes_declared",
        True,
        isinstance(declared_hashes, dict) and required_outputs.issubset(declared_hashes),
        "producer_required_output_hashes_missing",
    )
    if isinstance(declared_hashes, dict):
        for filename, expected_digest in sorted(declared_hashes.items()):
            path = producer / str(filename)
            valid = path.is_file() and _sha256(path) == expected_digest
            check(
                f"producer_output_hash:{filename}",
                True,
                valid,
                f"producer_output_hash_mismatch:{filename}",
            )

    archive_manifest_path = Path(str(producer_manifest.get("archive_manifest", "")))
    archive_manifest_hash_valid = (
        archive_manifest_path.is_file()
        and _sha256(archive_manifest_path) == producer_manifest.get("archive_manifest_sha256")
    )
    check(
        "immutable_archive_manifest_hash",
        True,
        archive_manifest_hash_valid,
        "archive_manifest_hash_mismatch",
    )

    archives = _read_csv(archive_manifest_path, issues, "archive_source_manifest")
    cutoff = _parse_date(producer_manifest.get("cutoff_date"))
    archive_shape_valid = _validate_archive_denominator(archives, cutoff, issues)
    check("archive_month_denominator", True, archive_shape_valid, "archive_month_denominator_invalid")
    valid_archive_rows = _validate_archive_sources(archives, issues)
    check(
        "archive_raw_sources_valid",
        len(archives),
        valid_archive_rows,
        "archive_raw_source_denominator_invalid",
    )

    expected_plan, discovery_issues = _enumerate_archive_notices(archives, cutoff)
    issues.extend(discovery_issues)
    plan = _read_csv(producer / "notice_plan.csv", issues, "notice_plan")
    state = _read_csv(producer / "page_state.csv", issues, "page_state")
    events = _read_csv(producer / "price_limit_events.csv", issues, "price_limit_events", dtype={"code": str})
    quality = _read_csv(producer / "event_quality.csv", issues, "event_quality", dtype={"code": str})

    plan_match = _plans_match(expected_plan, plan)
    check("planned_notice_pages", len(expected_plan), len(plan), "notice_plan_page_denominator_mismatch")
    check("notice_plan_exact_archive_enumeration", True, plan_match, "notice_plan_archive_enumeration_mismatch")
    expected_counts = _family_counts(expected_plan)
    plan_counts = _family_counts(plan)
    state_counts = _family_counts(state)
    for family in TITLE_FAMILIES:
        check(
            f"title_family_plan:{family}",
            expected_counts[family],
            plan_counts[family],
            f"notice_plan_title_family_count_mismatch:{family}",
        )
        check(
            f"title_family_page_state:{family}",
            expected_counts[family],
            state_counts[family],
            f"page_state_title_family_count_mismatch:{family}",
        )

    state_set_valid = _state_matches_plan(expected_plan, state)
    check("page_state_exact_plan", True, state_set_valid, "page_state_plan_set_mismatch")
    valid_notice_pages = _validate_notice_pages(state, issues)
    check(
        "notice_page_raw_sources_valid",
        len(expected_plan),
        valid_notice_pages,
        "notice_page_raw_source_denominator_invalid",
    )

    independent_events, parse_issues = _independent_event_parse(state)
    issues.extend(parse_issues)
    event_keys_match = _event_keys(independent_events) == _event_keys(events)
    check("independent_event_rows", len(independent_events), len(events), "event_row_denominator_mismatch")
    check("independent_event_key_set", True, event_keys_match, "independent_event_key_set_mismatch")
    field_issues = _event_field_issues(independent_events, events)
    issues.extend(field_issues)
    check("event_effective_dates", 0, sum("effective_date" in value for value in field_issues), "event_effective_date_validation_failed")
    check("event_side_fields", 0, sum("side" in value for value in field_issues), "event_side_validation_failed")
    numeric_failures = sum("numeric" in value or "multiplier" in value for value in field_issues)
    check("event_numeric_fields", 0, numeric_failures, "event_numeric_validation_failed")

    regime_issues = _regime_multiplier_issues(independent_events)
    issues.extend(regime_issues)
    check("regime_multiplier_2x_then_4x", 0, len(regime_issues), "regime_multiplier_validation_failed")
    quality_issues = _quality_recompute_issues(independent_events, quality)
    issues.extend(quality_issues)
    check("producer_event_quality_recomputed", 0, len(quality_issues), "event_quality_recompute_failed")

    chain_issues, chain_metrics = _event_chain_issues(independent_events)
    issues.extend(chain_issues)
    check("event_chain_transitions", 0, len(chain_issues), "event_chain_validation_failed")
    summary_rows.extend(
        [
            {"check": key, "expected": "observed", "observed": value, "passed": True}
            for key, value in chain_metrics.items()
        ]
    )

    manifest_counts = producer_manifest.get("counts", {})
    count_expectations = {
        "archive_sources": len(archives),
        "planned_notice_pages": len(expected_plan),
        "completed_notice_pages": valid_notice_pages,
        "parsed_event_rows": len(events),
    }
    for key, expected in count_expectations.items():
        check(
            f"producer_manifest_count:{key}",
            expected,
            manifest_counts.get(key),
            f"producer_manifest_count_mismatch:{key}",
        )

    issues = list(dict.fromkeys(issues))
    validation_summary = pd.DataFrame(summary_rows)
    summary_path = root / "validation_summary.csv"
    _atomic_csv(summary_path, validation_summary)
    remaining = list(REQUIRED_COMPLETION_BLOCKERS)
    manifest = {
        "schema_version": "1.0-independent-partial",
        "run_id": run_id,
        "validator_run_id": run_id,
        "producer_run_id": producer_manifest.get("run_id"),
        "artifact_type": "japan_exceptional_price_limit_event_history_independent_validation",
        "status": "pass_partial_component" if not issues else "failed_validation",
        "component": "temporary_price_limit_events",
        "independent_validation": True,
        "official_jpx_source": True,
        "research_only": True,
        "component_scorecard_eligible": False,
        "scorecard_eligible": False,
        "promotion_allowed": False,
        "validation_issues": issues,
        "remaining_completion_blockers": remaining,
        "critical_blockers": [*issues, *remaining],
        "counts": {
            "archive_sources": len(archives),
            "expected_notice_pages": len(expected_plan),
            "valid_notice_pages": valid_notice_pages,
            "independently_parsed_event_rows": len(independent_events),
            "producer_event_rows": len(events),
            **chain_metrics,
        },
        "title_family_counts": expected_counts,
        "producer_manifest": str(producer_manifest_path),
        "producer_manifest_sha256": _sha256(producer_manifest_path) if producer_manifest_path.is_file() else "",
        "archive_manifest": str(archive_manifest_path),
        "archive_manifest_sha256": _sha256(archive_manifest_path) if archive_manifest_path.is_file() else "",
        "artifacts": {"validation_summary": str(summary_path)},
        "output_sha256": {summary_path.name: _sha256(summary_path)},
        "promotion_policy": dict(PROMOTION_POLICY),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanPriceLimitEventHistoryValidationArtifacts(run_id, root, manifest_path, manifest)


def _classify_archive_title(title: str) -> str:
    value = _normalize_text(title)
    if re.search(r"\b(?:ETF|ETN)\b|subscription warrant|securities option", value, re.I):
        return ""
    families = (
        ("PRELIMINARY", r"^(?:Expansion|Broadening) of daily price limits\s*\(Preliminary Notice\)\s*[:\-]"),
        ("CANCELLED", r"^(?:Handling of daily price limits\s*\(Daily price limits will not be expanded\)|Announced Broadening of Daily Price Limits to be Cancelled)\s*[:\-]"),
        ("BROADENING", r"^(?:Expansion of daily price limits\s*\(Daily price limits? will be expanded\)|Daily Price Limits to be Broadened)\s*[:\-]"),
        ("CONTINUATION", r"^(?:Continued expansion of daily price limits\s*\(Daily price limits will continue to be expanded\)|Continue Broadened Daily Price Limits|Daily Price Limits Continue to be Broadened)\s*[:\-]"),
        ("RETURN_NORMAL", r"^(?:Removal of expanded daily price limits\s*\(Return to normal daily price limits\)|Daily Price Limits to be Returned to Normal)\s*[:\-]"),
    )
    for family, pattern in families:
        if re.search(pattern, value, re.I):
            return family
    return ""


def _enumerate_archive_notices(
    archives: pd.DataFrame, cutoff: date | None
) -> tuple[pd.DataFrame, list[str]]:
    columns = ["plan_row_id", "publication_date", "event_type", "title", "url", "archive_source_ids"]
    if cutoff is None:
        return pd.DataFrame(columns=columns), ["producer_cutoff_date_invalid"]
    rows: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for source in archives.to_dict("records"):
        source_id = str(source.get("source_id", ""))
        raw = Path(str(source.get("raw_path", "")))
        if not raw.is_file() or _sha256(raw) != str(source.get("sha256", "")):
            continue
        try:
            payload = json.loads(raw.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            issues.append(f"archive_json_invalid:{source_id}")
            continue
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, list):
                stack.extend(value)
                continue
            if not isinstance(value, dict):
                continue
            title = str(value.get("title", ""))
            target = str(value.get("url", ""))
            stamp = value.get("updated_date")
            family = _classify_archive_title(title)
            published = _archive_item_date(stamp)
            if family and published and published <= cutoff and target:
                url = urljoin(JPX_ROOT, target)
                if not re.fullmatch(
                    r"https://www\.jpx\.co\.jp/english/news/(?:detail/detail_[0-9]+|1030/[A-Za-z0-9_-]+)\.html",
                    url,
                ):
                    issues.append(f"selected_notice_url_invalid:{url}")
                else:
                    current = rows.setdefault(
                        url,
                        {
                            "publication_date": published.isoformat(),
                            "event_type": family,
                            "title": title.strip(),
                            "url": url,
                            "archive_source_ids": [],
                        },
                    )
                    if (
                        current["publication_date"] != published.isoformat()
                        or current["event_type"] != family
                        or _normalize_text(current["title"]) != _normalize_text(title)
                    ):
                        issues.append(f"archive_duplicate_notice_conflict:{url}")
                    current["archive_source_ids"].append(source_id)
            stack.extend(value.values())
    records = []
    for value in rows.values():
        records.append({**value, "archive_source_ids": "|".join(sorted(set(value["archive_source_ids"])))})
    records.sort(key=lambda row: (row["publication_date"], row["url"]))
    for index, row in enumerate(records, 1):
        row["plan_row_id"] = f"JPL-{index:05d}"
    return pd.DataFrame(records, columns=columns), issues


def _independent_event_parse(state: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "publication_date",
        "archive_publication_date",
        "effective_date",
        "event_type",
        "code",
        "side",
        "base_price_yen",
        "expanded_range_yen",
        "normal_range_yen",
        "observed_multiplier",
        "source_url",
        "title",
    ]
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    required = {"plan_row_id", "publication_date", "event_type", "url", "title", "terminal_status", "raw_path"}
    if not required.issubset(state.columns):
        return pd.DataFrame(columns=columns), ["page_state_schema_invalid_for_parse"]
    for item in state.loc[state["terminal_status"].eq("success")].to_dict("records"):
        raw = Path(str(item["raw_path"]))
        parsed = _parse_notice_page(
            raw.read_bytes(),
            publication_date=str(item["publication_date"]),
            event_type=str(item["event_type"]),
            source_url=str(item["url"]),
            title=str(item["title"]),
        ) if raw.is_file() else []
        if not parsed:
            issues.append(f"independent_page_parse_failure:{item['plan_row_id']}")
        rows.extend(parsed)
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.drop_duplicates(["publication_date", "event_type", "code", "source_url"])
        frame = frame.sort_values(["publication_date", "code", "event_type", "source_url"]).reset_index(drop=True)
    return frame, issues


def _parse_notice_page(
    body: bytes,
    *,
    publication_date: str,
    event_type: str,
    source_url: str,
    title: str,
) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    paragraphs = [_normalize_text(" ".join(node.stripped_strings)) for node in soup.find_all("p")]
    text = " ".join(value for value in paragraphs if value) or _normalize_text(soup.get_text(" ", strip=True))
    code_matches = list(re.finditer(r"(?:\(\s*)?Code\s*[:：]\s*((?=[0-9A-Z]{4}\b)(?=[0-9A-Z]*\d)[0-9A-Z]{4})(?:\s*\))?", text, re.I))
    unique_matches: list[re.Match[str]] = []
    seen_codes: set[str] = set()
    for match in code_matches:
        code = match.group(1).upper()
        if code not in seen_codes:
            unique_matches.append(match)
            seen_codes.add(code)
    archive_published = _parse_date(publication_date)
    if archive_published is None:
        return []
    published = _url_publication_date(source_url) or archive_published
    rows = []
    for index, match in enumerate(unique_matches):
        segment_end = unique_matches[index + 1].start() if index + 1 < len(unique_matches) else len(text)
        segment = text[match.end():segment_end]
        context = segment if len(unique_matches) > 1 else text
        side_context = text[max(0, match.start() - 240):segment_end] if len(unique_matches) > 1 else text
        limits = re.search(
            r"\b(Upper|Lower) daily price limit\s*:\s*([\d,]+)\s*yen\s*"
            r"\((?:Upper|Lower) limit will remain\s*([\d,]+)\s*yen\s*as per normal\)",
            segment,
            re.I,
        )
        if limits:
            side = limits.group(1).upper()
            expanded = int(limits.group(2).replace(",", ""))
            normal = int(limits.group(3).replace(",", ""))
        else:
            side = _extract_side(side_context)
            expanded = _extract_yen(context, r"(?:Upper|Lower) daily price limit\s*:\s*([\d,]+)\s*yen\s*\(")
            normal = _extract_yen(context, r"(?:Upper|Lower) limit will remain\s*([\d,]+)\s*yen")
        base = _extract_yen(context, r"Base price\s*:\s*([\d,]+)\s*yen")
        multiplier: float | None = None
        if expanded is not None and normal:
            multiplier = expanded / normal
        else:
            multiplier_text = context.lower()
            if re.search(r"\b(?:four times|quadruple)\b", multiplier_text):
                multiplier = 4.0
            elif re.search(r"\b(?:twice|two times)\b", multiplier_text):
                multiplier = 2.0
        effective = _extract_effective_date(text, published)
        rows.append(
            {
                "publication_date": published.isoformat(),
                "archive_publication_date": archive_published.isoformat(),
                "effective_date": effective.isoformat(),
                "event_type": event_type,
                "code": match.group(1).upper(),
                "side": side,
                "base_price_yen": base,
                "expanded_range_yen": expanded,
                "normal_range_yen": normal,
                "observed_multiplier": multiplier,
                "source_url": source_url,
                "title": title,
            }
        )
    return rows


def _event_field_issues(expected: pd.DataFrame, observed: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    required = {
        "publication_date", "archive_publication_date", "effective_date", "event_type", "code", "side", "base_price_yen",
        "expanded_range_yen", "normal_range_yen", "observed_multiplier", "source_url", "title",
    }
    if not required.issubset(observed.columns):
        return ["event_schema_invalid"]
    if observed.duplicated(["publication_date", "event_type", "code", "source_url"]).any():
        issues.append("event_duplicate_keys")
    expected_by_key = {_event_key(row): row for row in expected.to_dict("records")}
    observed_by_key = {_event_key(row): row for row in observed.to_dict("records")}
    for key in sorted(set(expected_by_key) & set(observed_by_key)):
        wanted = expected_by_key[key]
        actual = observed_by_key[key]
        if str(actual.get("effective_date", "")) != str(wanted["effective_date"]):
            issues.append(f"event_effective_date_mismatch:{_key_label(key)}")
        if str(actual.get("archive_publication_date", "")) != str(wanted["archive_publication_date"]):
            issues.append(f"event_archive_publication_date_mismatch:{_key_label(key)}")
        if str(actual.get("side", "")).upper() != str(wanted["side"]):
            issues.append(f"event_side_mismatch:{_key_label(key)}")
        for field in ("base_price_yen", "expanded_range_yen", "normal_range_yen", "observed_multiplier"):
            if not _numeric_equal(actual.get(field), wanted[field]):
                suffix = "multiplier" if field == "observed_multiplier" else "numeric"
                issues.append(f"event_{suffix}_mismatch:{field}:{_key_label(key)}")
        if _normalize_text(str(actual.get("title", ""))) != _normalize_text(str(wanted["title"])):
            issues.append(f"event_title_mismatch:{_key_label(key)}")
        published = _parse_date(actual.get("publication_date"))
        effective = _parse_date(actual.get("effective_date"))
        if published is None or effective is None or effective < published or (effective - published).days > 14:
            issues.append(f"event_effective_date_out_of_range:{_key_label(key)}")
        if str(actual.get("side", "")).upper() not in {"UPPER", "LOWER"}:
            issues.append(f"event_side_missing_or_invalid:{_key_label(key)}")
        if actual.get("event_type") in {"BROADENING", "CONTINUATION"}:
            numeric = [_to_float(actual.get(field)) for field in ("base_price_yen", "expanded_range_yen", "normal_range_yen")]
            if any(value is None or value <= 0 for value in numeric):
                issues.append(f"event_numeric_required_field_invalid:{_key_label(key)}")
    return issues


def _regime_multiplier_issues(events: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if events.empty:
        return issues
    for row in events.loc[events["event_type"].isin({"BROADENING", "CONTINUATION"})].to_dict("records"):
        published = _parse_date(row.get("publication_date"))
        observed = _to_float(row.get("observed_multiplier"))
        expected = 2.0 if published and published < REGIME_CHANGE_DATE else 4.0
        if observed is None or abs(observed - expected) > 1e-9:
            issues.append(f"regime_multiplier_mismatch:{_key_label(_event_key(row))}:{observed}/{expected}")
    return issues


def _quality_recompute_issues(events: pd.DataFrame, quality: pd.DataFrame) -> list[str]:
    required = {"publication_date", "event_type", "code", "expected_multiplier", "observed_multiplier", "regime_multiplier_valid"}
    if not required.issubset(quality.columns) or len(quality) != len(events):
        return ["event_quality_schema_or_denominator_invalid"]
    issues: list[str] = []
    quality_rows = {
        (str(row["publication_date"]), str(row["event_type"]), str(row["code"])): row
        for row in quality.to_dict("records")
    }
    for event in events.to_dict("records"):
        key = (str(event["publication_date"]), str(event["event_type"]), str(event["code"]))
        row = quality_rows.get(key)
        if row is None:
            issues.append(f"event_quality_row_missing:{'/'.join(key)}")
            continue
        applies = event["event_type"] in {"BROADENING", "CONTINUATION"} and _to_float(event["observed_multiplier"]) is not None
        published = _parse_date(event["publication_date"])
        expected = (2.0 if published and published < REGIME_CHANGE_DATE else 4.0) if applies else None
        valid = abs(float(event["observed_multiplier"]) - float(expected)) < 1e-9 if applies else True
        if not _numeric_equal(row.get("expected_multiplier"), expected):
            issues.append(f"event_quality_expected_multiplier_mismatch:{'/'.join(key)}")
        if not _numeric_equal(row.get("observed_multiplier"), event.get("observed_multiplier")):
            issues.append(f"event_quality_observed_multiplier_mismatch:{'/'.join(key)}")
        if _to_bool(row.get("regime_multiplier_valid")) is not valid:
            issues.append(f"event_quality_validity_mismatch:{'/'.join(key)}")
    return issues


def _event_chain_issues(events: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    issues: list[str] = []
    left_censored = 0
    post_regime_unresolved = 0
    complete_pre_regime = 0
    if events.empty:
        return issues, {
            "complete_pre_2020_event_chains": 0,
            "left_censored_pre_2013_chains": 0,
            "post_2020_chains_requiring_market_data": 0,
        }
    ordered = events.sort_values(["publication_date", "effective_date", "source_url", "code"])
    for code, group in ordered.groupby("code", sort=True):
        state = "IDLE"
        seen = False
        for row in group.to_dict("records"):
            published = _parse_date(row["publication_date"])
            family = row["event_type"]
            label = f"{code}:{row['publication_date']}:{family}"
            if published and published < REGIME_CHANGE_DATE:
                if family == "PRELIMINARY":
                    if state != "IDLE":
                        issues.append(f"event_chain_unexpected_preliminary:{label}:{state}")
                    state = "PRELIMINARY"
                elif family == "CANCELLED":
                    if state != "PRELIMINARY":
                        issues.append(f"event_chain_cancel_without_preliminary:{label}:{state}")
                    state = "IDLE"
                elif family == "BROADENING":
                    if state != "PRELIMINARY":
                        issues.append(f"event_chain_broadening_without_preliminary:{label}:{state}")
                    state = "ACTIVE"
                elif family == "CONTINUATION":
                    if state != "ACTIVE":
                        issues.append(f"event_chain_continuation_without_broadening:{label}:{state}")
                    state = "ACTIVE"
                elif family == "RETURN_NORMAL":
                    if state != "ACTIVE":
                        if not seen and published < date(2013, 10, 1):
                            left_censored += 1
                        else:
                            issues.append(f"event_chain_return_without_broadening:{label}:{state}")
                    else:
                        complete_pre_regime += 1
                    state = "IDLE"
            else:
                if family in {"PRELIMINARY", "CANCELLED", "RETURN_NORMAL"}:
                    issues.append(f"event_chain_legacy_notice_after_regime_change:{label}")
                elif family == "CONTINUATION" and state != "ACTIVE":
                    issues.append(f"event_chain_continuation_without_broadening:{label}:{state}")
                if family == "BROADENING" and state == "ACTIVE":
                    post_regime_unresolved += 1
                if family in {"BROADENING", "CONTINUATION"}:
                    state = "ACTIVE"
            seen = True
        if state == "ACTIVE" and any(
            (_parse_date(value) or date.min) >= REGIME_CHANGE_DATE for value in group["publication_date"]
        ):
            post_regime_unresolved += 1
    return issues, {
        "complete_pre_2020_event_chains": complete_pre_regime,
        "left_censored_pre_2013_chains": left_censored,
        "post_2020_chains_requiring_market_data": post_regime_unresolved,
    }


def _validate_archive_denominator(archives: pd.DataFrame, cutoff: date | None, issues: list[str]) -> bool:
    required = {"source_id", "url", "http_status", "content_type", "raw_path", "sha256", "bytes"}
    if cutoff is None or not required.issubset(archives.columns):
        return False
    fixed_end = min(cutoff.replace(day=1), date(2024, 8, 1))
    fixed_months = _month_sequence(date(2013, 9, 1), fixed_end)
    rolling_months = (
        _month_sequence(ROLLING_ARCHIVE_START, cutoff.replace(day=1))
        if cutoff >= ROLLING_ARCHIVE_START
        else []
    )
    expected_ids = [f"news_ym_{stamp:%Y%m}" for stamp in fixed_months]
    expected_ids.extend(f"news_ym_{index:02d}" for index in range(1, len(rolling_months) + 1))
    expected_urls = [f"{JPX_ROOT}/english/news/news_ym_{stamp:%Y%m}.json" for stamp in fixed_months]
    expected_urls.extend(
        f"{JPX_ROOT}/english/news/news_ym_{index:02d}.json"
        for index in range(1, len(rolling_months) + 1)
    )
    valid = archives["source_id"].astype(str).tolist() == expected_ids and archives["url"].astype(str).tolist() == expected_urls
    if archives["source_id"].duplicated().any() or archives["url"].duplicated().any():
        issues.append("archive_source_duplicate_ids_or_urls")
        valid = False
    return valid


def _validate_archive_sources(archives: pd.DataFrame, issues: list[str]) -> int:
    required = {"source_id", "http_status", "content_type", "raw_path", "sha256", "bytes"}
    if not required.issubset(archives.columns):
        issues.append("archive_source_manifest_schema_invalid")
        return 0
    valid_count = 0
    for row in archives.to_dict("records"):
        source_id = str(row["source_id"])
        path = Path(str(row["raw_path"]))
        valid = (
            _to_int(row["http_status"]) == 200
            and "json" in str(row["content_type"]).lower()
            and path.is_file()
            and _sha256(path) == str(row["sha256"])
            and path.stat().st_size == _to_int(row["bytes"])
        )
        if valid:
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                valid = False
        if valid:
            valid_count += 1
        else:
            issues.append(f"archive_raw_source_invalid:{source_id}")
    return valid_count


def _validate_notice_pages(state: pd.DataFrame, issues: list[str]) -> int:
    required = {"plan_row_id", "http_status", "terminal_status", "content_type", "raw_path", "sha256", "bytes"}
    if not required.issubset(state.columns):
        issues.append("page_state_schema_invalid")
        return 0
    valid_count = 0
    for row in state.to_dict("records"):
        row_id = str(row["plan_row_id"])
        path = Path(str(row["raw_path"]))
        valid = (
            row["terminal_status"] == "success"
            and _to_int(row["http_status"]) == 200
            and "html" in str(row["content_type"]).lower()
            and path.is_file()
            and _sha256(path) == str(row["sha256"])
            and path.stat().st_size == _to_int(row["bytes"])
        )
        if valid:
            valid_count += 1
        else:
            issues.append(f"notice_page_raw_source_invalid:{row_id}")
    return valid_count


def _plans_match(expected: pd.DataFrame, observed: pd.DataFrame) -> bool:
    columns = ["plan_row_id", "publication_date", "event_type", "title", "url", "archive_source_ids"]
    if not set(columns).issubset(observed.columns) or len(expected) != len(observed):
        return False
    left = expected[columns].fillna("").astype(str).reset_index(drop=True)
    right = observed[columns].fillna("").astype(str).reset_index(drop=True)
    return left.equals(right)


def _state_matches_plan(expected: pd.DataFrame, state: pd.DataFrame) -> bool:
    columns = ["plan_row_id", "publication_date", "event_type", "title", "url", "archive_source_ids"]
    if not set(columns).issubset(state.columns) or len(expected) != len(state):
        return False
    return expected[columns].fillna("").astype(str).reset_index(drop=True).equals(
        state[columns].fillna("").astype(str).reset_index(drop=True)
    )


def _family_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["event_type"].value_counts().to_dict() if "event_type" in frame.columns else {}
    return {family: int(counts.get(family, 0)) for family in TITLE_FAMILIES}


def _event_keys(frame: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    required = {"publication_date", "event_type", "code", "source_url"}
    if not required.issubset(frame.columns):
        return set()
    return {_event_key(row) for row in frame.to_dict("records")}


def _event_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("publication_date", "")),
        str(row.get("event_type", "")),
        str(row.get("code", "")),
        str(row.get("source_url", "")),
    )


def _key_label(key: tuple[str, str, str, str]) -> str:
    return ":".join(key[:3])


def _extract_side(text: str) -> str:
    patterns = (
        r"(?:broaden|expand)(?:ed|ing)?\s+(?:only\s+)?(?:its\s+|the\s+)?(upper|lower)\b",
        r"\b(upper|lower) daily price limit\b[^.]{0,80}\b(?:expand|broaden)(?:ed|ing)?\b",
        r"\b(upper|lower) daily price limit\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).upper()
    return ""


def _extract_effective_date(text: str, published: date) -> date:
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


def _extract_yen(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, re.I)
    return int(match.group(1).replace(",", "")) if match else None


def _archive_item_date(value: Any) -> date | None:
    if not isinstance(value, dict):
        return None
    try:
        return date(int(value["year"]), int(value["month"]), int(value["day"]))
    except (KeyError, TypeError, ValueError):
        return None


def _url_publication_date(source_url: str) -> date | None:
    match = re.search(r"/([0-9]{8})-[0-9]+\.html(?:\?.*)?$", source_url)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _month_sequence(start: date, end: date) -> list[date]:
    rows = []
    current = start.replace(day=1)
    while current <= end.replace(day=1):
        rows.append(current)
        current = date(current.year + int(current.month == 12), current.month % 12 + 1, 1)
    return rows


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", html.unescape(value))).strip()


def _numeric_equal(left: Any, right: Any) -> bool:
    left_value = _to_float(left)
    right_value = _to_float(right)
    if left_value is None or right_value is None:
        return left_value is right_value
    return abs(left_value - right_value) < 1e-9


def _to_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None and number.is_integer() else None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _same_value(expected: Any, observed: Any) -> bool:
    if isinstance(expected, bool) or isinstance(observed, bool):
        return expected is observed
    return expected == observed


def _display(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path, issues: list[str], label: str, dtype: dict[str, Any] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=dtype)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        issues.append(f"{label}_missing_or_invalid")
        return pd.DataFrame()


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
