from __future__ import annotations

import hashlib
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
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
REGIME_CHANGE_DATE = date(2020, 8, 3)
MAX_SOURCE_BYTES = 16 * 1024 * 1024
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
SOURCE_OUTPUTS = (
    "notice_plan.csv",
    "page_state.csv",
    "price_limit_events.csv",
    "event_quality.csv",
)
FETCH_STATE_COLUMNS = (
    "counterpart_id",
    "japanese_source_url",
    "http_status",
    "terminal_status",
    "content_type",
    "raw_path",
    "sha256",
    "bytes",
    "error",
)
RECONCILIATION_COLUMNS = (
    "reconciliation_id",
    "publication_date",
    "event_type",
    "code",
    "expected_multiplier",
    "english_regime_mismatch",
    "english_numeric_consistency_mismatch",
    "english_quality_flag_mismatch",
    "english_side",
    "english_base_price_yen",
    "english_expanded_range_yen",
    "english_normal_range_yen",
    "english_observed_multiplier",
    "english_source_url",
    "english_raw_path",
    "english_sha256",
    "english_bytes",
    "japanese_publication_date",
    "japanese_side",
    "japanese_base_price_yen",
    "japanese_expanded_range_yen",
    "japanese_normal_range_yen",
    "japanese_observed_multiplier",
    "japanese_source_url",
    "japanese_raw_path",
    "japanese_sha256",
    "japanese_bytes",
    "code_match",
    "date_match",
    "side_match",
    "japanese_regime_match",
    "bilingual_limit_numeric_mismatch",
    "bilingual_base_price_mismatch",
    "correction_eligible",
    "reconciliation_status",
    "issues",
)


Fetcher = Callable[[str], tuple[int, bytes, dict[str, str]]]


@dataclass(frozen=True)
class JapanPriceLimitBilingualReconciliationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def derive_japanese_counterpart_url(english_url: str) -> str:
    """Derive the official Japanese JPX notice URL without changing its identity."""
    parts = urlsplit(str(english_url).strip())
    if (
        parts.scheme != "https"
        or parts.netloc.lower() != "www.jpx.co.jp"
        or not parts.path.startswith("/english/news/")
        or not parts.path.endswith(".html")
    ):
        raise ValueError(f"unsupported official English JPX notice URL: {english_url}")
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.removeprefix("/english"), parts.query, ""))


def parse_japanese_price_limit_notice(body: bytes, *, source_url: str) -> list[dict[str, Any]]:
    """Parse per-code values from an official Japanese exceptional-limit notice."""
    soup = BeautifulSoup(body, "html.parser")
    text = unicodedata.normalize("NFKC", soup.get_text(" ", strip=True))
    text = re.sub(r"\s+", " ", text)
    publication_date = _japanese_publication_date(soup, text, source_url)
    code_matches = list(re.finditer(r"(?:銘柄)?コード\s*[:：]?\s*([0-9A-Z]{4})", text, re.I))
    rows: list[dict[str, Any]] = []
    for index, code_match in enumerate(code_matches):
        segment_end = code_matches[index + 1].start() if index + 1 < len(code_matches) else len(text)
        segment = text[code_match.end() : segment_end]
        limits = re.search(
            r"制限値幅\s*\(\s*(上限|下限)\s*\)\s*:\s*([\d,]+)\s*円\s*"
            r"\(\s*(?:上限|下限)は通常(?:どお|通)り\s*([\d,]+)\s*円\s*\)",
            segment,
        )
        side = {"上限": "UPPER", "下限": "LOWER"}.get(limits.group(1), "") if limits else ""
        expanded = _integer(limits.group(2)) if limits else None
        normal = _integer(limits.group(3)) if limits else None
        base_match = re.search(r"基準値段\s*:\s*([\d,]+)\s*円", segment)
        base = _integer(base_match.group(1)) if base_match else None
        rows.append(
            {
                "publication_date": publication_date,
                "code": code_match.group(1).upper(),
                "side": side,
                "base_price_yen": base,
                "expanded_range_yen": expanded,
                "normal_range_yen": normal,
                "observed_multiplier": expanded / normal if expanded is not None and normal else None,
                "source_url": source_url,
            }
        )
    return rows


def write_japan_price_limit_bilingual_reconciliation(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_price_limit_bilingual_reconciliation",
    max_pages: int = 0,
    sleep_seconds: float = 0.02,
    fetcher: Fetcher | None = None,
) -> JapanPriceLimitBilingualReconciliationArtifacts:
    """Reconcile English regime failures against immutable Japanese JPX pages."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    if max_pages < 0 or sleep_seconds < 0:
        raise ValueError("max_pages and sleep_seconds must be non-negative")

    producer = Path(producer_dir).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    if root == producer or _is_relative_to(root, producer):
        raise ValueError("reconciliation output must not be inside the source producer directory")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc

    try:
        source_manifest_path = producer / "manifest.json"
        source_manifest = _load_source_manifest(source_manifest_path, producer)
        events = pd.read_csv(producer / "price_limit_events.csv", dtype={"code": str})
        quality = pd.read_csv(producer / "event_quality.csv", dtype={"code": str})
        page_state = pd.read_csv(producer / "page_state.csv")
        mismatch_plan = _build_mismatch_plan(events, quality)

        counterpart_urls = sorted({row["japanese_source_url"] for row in mismatch_plan})
        counterpart_plan = {
            url: f"JPLBR-PAGE-{index:05d}" for index, url in enumerate(counterpart_urls, 1)
        }
        state_path = root / "counterpart_fetch_state.csv"
        existing_state = _read_existing_state(state_path)
        fetch = fetcher or _fetch
        state_rows: list[dict[str, Any]] = []
        new_requests = 0
        for url, counterpart_id in counterpart_plan.items():
            prior = existing_state.get(url, {})
            prior_raw = Path(str(prior.get("raw_path", "")))
            if (
                prior.get("terminal_status") == "success"
                and _is_relative_to(prior_raw, root)
                and prior_raw.is_file()
                and _sha256(prior_raw) == str(prior.get("sha256", ""))
            ):
                state_rows.append({column: prior.get(column, "") for column in FETCH_STATE_COLUMNS})
                continue
            if max_pages and new_requests >= max_pages:
                state_rows.append(
                    _fetch_state_row(counterpart_id, url, terminal_status="pending", error="bounded_run_limit")
                )
                continue
            try:
                new_requests += 1
                status, body, headers = fetch(url)
                if status == 200 and body and len(body) <= MAX_SOURCE_BYTES:
                    digest = hashlib.sha256(body).hexdigest()
                    raw_path = root / "raw" / "japanese" / f"{counterpart_id}_{digest[:16]}.html"
                    if not raw_path.is_file() or _sha256(raw_path) != digest:
                        _atomic_bytes(raw_path, body)
                    state_rows.append(
                        _fetch_state_row(
                            counterpart_id,
                            url,
                            http_status=status,
                            terminal_status="success",
                            content_type=_header(headers, "content-type"),
                            raw_path=str(raw_path),
                            sha256=digest,
                            byte_count=len(body),
                        )
                    )
                else:
                    error = "source_too_large" if len(body) > MAX_SOURCE_BYTES else f"http_status={status}"
                    state_rows.append(
                        _fetch_state_row(
                            counterpart_id,
                            url,
                            http_status=status,
                            terminal_status="retryable_failure",
                            content_type=_header(headers, "content-type"),
                            error=error,
                        )
                    )
            except Exception as exc:
                state_rows.append(
                    _fetch_state_row(
                        counterpart_id,
                        url,
                        terminal_status="retryable_failure",
                        error=f"{type(exc).__name__}:{exc}"[:1000],
                    )
                )
            _atomic_csv(state_path, pd.DataFrame(state_rows, columns=FETCH_STATE_COLUMNS))
            if sleep_seconds and new_requests:
                time.sleep(sleep_seconds)
        fetch_state = pd.DataFrame(state_rows, columns=FETCH_STATE_COLUMNS)
        _atomic_csv(state_path, fetch_state)

        state_by_url = {
            str(row["japanese_source_url"]): row for row in fetch_state.to_dict("records")
        }
        japanese_rows_by_url: dict[str, list[dict[str, Any]]] = {}
        for url, state in state_by_url.items():
            if state["terminal_status"] != "success":
                continue
            raw_path = Path(str(state["raw_path"]))
            japanese_rows_by_url[url] = parse_japanese_price_limit_notice(
                raw_path.read_bytes(), source_url=url
            )

        reconciliations: list[dict[str, Any]] = []
        corrected_events: list[dict[str, Any]] = []
        for candidate in mismatch_plan:
            reconciliation, corrected = _reconcile_candidate(
                candidate,
                page_state=page_state,
                output_root=root,
                counterpart_state=state_by_url.get(candidate["japanese_source_url"], {}),
                japanese_rows=japanese_rows_by_url.get(candidate["japanese_source_url"], []),
            )
            reconciliations.append(reconciliation)
            if corrected is not None:
                corrected_events.append(corrected)

        reconciliation_path = root / "reconciliation.csv"
        corrected_path = root / "corrected_candidate_events.csv"
        reconciliation_frame = pd.DataFrame(reconciliations, columns=RECONCILIATION_COLUMNS)
        corrected_columns = [
            *events.columns,
            *(
                column
                for column in (
                    "reconciliation_id",
                    "candidate_status",
                    "candidate_only",
                    "promotion_allowed",
                    "correction_reason",
                    "english_source_url",
                    "english_source_sha256",
                    "japanese_source_url",
                    "japanese_source_sha256",
                    "original_english_base_price_yen",
                    "original_english_expanded_range_yen",
                    "original_english_normal_range_yen",
                    "original_english_observed_multiplier",
                )
                if column not in events.columns
            ),
        ]
        corrected_frame = pd.DataFrame(corrected_events, columns=corrected_columns)
        _atomic_csv(reconciliation_path, reconciliation_frame)
        _atomic_csv(corrected_path, corrected_frame)

        pending = int(fetch_state["terminal_status"].eq("pending").sum()) if not fetch_state.empty else 0
        failed = int(fetch_state["terminal_status"].eq("retryable_failure").sum()) if not fetch_state.empty else 0
        unresolved = int((~reconciliation_frame["correction_eligible"]).sum()) if not reconciliation_frame.empty else 0
        reconciliation_blockers: list[str] = []
        if failed:
            reconciliation_blockers.append(f"japanese_counterpart_fetch_failures={failed}")
        if unresolved and not pending:
            reconciliation_blockers.append(f"unresolved_bilingual_mismatches={unresolved}")
        status = "in_progress" if pending else "partial_candidate" if not reconciliation_blockers else "failed_candidate"
        output_paths = (state_path, reconciliation_path, corrected_path)
        manifest = {
            "schema_version": "1.0-immutable-resumable-candidate",
            "run_id": run_id,
            "artifact_type": "japan_exceptional_price_limit_bilingual_reconciliation_candidate",
            "status": status,
            "component": "temporary_price_limit_events",
            "official_jpx_bilingual_sources": True,
            "source_producer_immutable": True,
            "atomic_writes": True,
            "single_writer_lock": True,
            "resumable": True,
            "research_only": True,
            "scorecard_eligible": False,
            "component_scorecard_eligible": False,
            "promotion_allowed": False,
            "counts": {
                "source_event_rows": len(events),
                "detected_mismatch_rows": len(mismatch_plan),
                "planned_japanese_counterpart_pages": len(counterpart_plan),
                "successful_japanese_counterpart_pages": int(fetch_state["terminal_status"].eq("success").sum()) if not fetch_state.empty else 0,
                "pending_japanese_counterpart_pages": pending,
                "failed_japanese_counterpart_pages": failed,
                "corrected_candidate_event_rows": len(corrected_events),
                "unresolved_mismatch_rows": unresolved,
            },
            "reconciliation_blockers": reconciliation_blockers,
            "remaining_completion_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
            "critical_blockers": [*reconciliation_blockers, *REQUIRED_COMPLETION_BLOCKERS],
            "source_producer": str(producer),
            "source_producer_run_id": source_manifest.get("run_id"),
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_output_sha256": {
                filename: _sha256(producer / filename) for filename in SOURCE_OUTPUTS
            },
            "artifacts": {
                "counterpart_fetch_state": str(state_path),
                "reconciliation": str(reconciliation_path),
                "corrected_candidate_events": str(corrected_path),
            },
            "output_sha256": {path.name: _sha256(path) for path in output_paths},
            "promotion_policy": dict(PROMOTION_POLICY),
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return JapanPriceLimitBilingualReconciliationArtifacts(run_id, root, manifest_path, manifest)
    finally:
        lock.rmdir()


def _load_source_manifest(path: Path, producer: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"source producer manifest missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"source producer manifest invalid: {path}") from exc
    if manifest.get("artifact_type") != "japan_exceptional_price_limit_event_history_candidate":
        raise ValueError("source producer artifact_type is not an exceptional price-limit candidate")
    declared = manifest.get("output_sha256", {})
    for filename in SOURCE_OUTPUTS:
        source = producer / filename
        if not source.is_file() or declared.get(filename) != _sha256(source):
            raise ValueError(f"immutable source output hash mismatch: {filename}")
    return manifest


def _build_mismatch_plan(events: pd.DataFrame, quality: pd.DataFrame) -> list[dict[str, Any]]:
    quality_flags: dict[tuple[str, str, str], bool] = {}
    for row in quality.to_dict("records"):
        key = (str(row.get("publication_date", "")), str(row.get("event_type", "")), str(row.get("code", "")))
        quality_flags[key] = quality_flags.get(key, True) and _truthy(row.get("regime_multiplier_valid"))

    rows: list[dict[str, Any]] = []
    applicable = events.loc[events["event_type"].isin({"BROADENING", "CONTINUATION"})].copy()
    applicable = applicable.sort_values(["publication_date", "code", "event_type", "source_url"])
    for event in applicable.to_dict("records"):
        published = date.fromisoformat(str(event["publication_date"]))
        expected = 2.0 if published < REGIME_CHANGE_DATE else 4.0
        expanded = _number(event.get("expanded_range_yen"))
        normal = _number(event.get("normal_range_yen"))
        observed = _number(event.get("observed_multiplier"))
        ratio = expanded / normal if expanded is not None and normal else None
        regime_mismatch = ratio is None or not _close(ratio, expected)
        numeric_consistency_mismatch = ratio is None or observed is None or not _close(ratio, observed)
        key = (str(event["publication_date"]), str(event["event_type"]), str(event["code"]))
        quality_mismatch = not quality_flags.get(key, True)
        if not (regime_mismatch or numeric_consistency_mismatch or quality_mismatch):
            continue
        rows.append(
            {
                **event,
                "expected_multiplier": expected,
                "english_regime_mismatch": regime_mismatch,
                "english_numeric_consistency_mismatch": numeric_consistency_mismatch,
                "english_quality_flag_mismatch": quality_mismatch,
                "japanese_source_url": derive_japanese_counterpart_url(str(event["source_url"])),
            }
        )
    for index, row in enumerate(rows, 1):
        row["reconciliation_id"] = f"JPLBR-{index:05d}"
    return rows


def _reconcile_candidate(
    candidate: dict[str, Any],
    *,
    page_state: pd.DataFrame,
    output_root: Path,
    counterpart_state: dict[str, Any],
    japanese_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    issues: list[str] = []
    english = _retain_english_source(candidate, page_state, output_root, issues)
    japanese_matches = [row for row in japanese_rows if row["code"] == str(candidate["code"])]
    code_match = len(japanese_matches) == 1
    japanese = japanese_matches[0] if code_match else {}
    if not code_match:
        issues.append("japanese_code_missing" if not japanese_matches else "japanese_code_ambiguous")

    japanese_date = str(japanese.get("publication_date", ""))
    japanese_side = str(japanese.get("side", ""))
    date_match = bool(japanese_date) and japanese_date == str(candidate["publication_date"])
    side_match = bool(japanese_side) and japanese_side == str(candidate.get("side", ""))
    if code_match and not date_match:
        issues.append("publication_date_mismatch")
    if code_match and not side_match:
        issues.append("side_mismatch")

    jp_expanded = _number(japanese.get("expanded_range_yen"))
    jp_normal = _number(japanese.get("normal_range_yen"))
    jp_multiplier = jp_expanded / jp_normal if jp_expanded is not None and jp_normal else None
    regime_match = jp_multiplier is not None and _close(jp_multiplier, float(candidate["expected_multiplier"]))
    if code_match and not regime_match:
        issues.append("japanese_regime_mismatch")

    english_expanded = _number(candidate.get("expanded_range_yen"))
    english_normal = _number(candidate.get("normal_range_yen"))
    limit_numeric_mismatch = (
        jp_expanded is not None
        and jp_normal is not None
        and (not _close(jp_expanded, english_expanded) or not _close(jp_normal, english_normal))
    )
    base_numeric_mismatch = not _close(
        _number(japanese.get("base_price_yen")), _number(candidate.get("base_price_yen"))
    )
    if code_match and not limit_numeric_mismatch:
        issues.append("no_bilingual_limit_numeric_mismatch")

    fetch_status = str(counterpart_state.get("terminal_status", ""))
    if fetch_status == "pending":
        issues.append("japanese_counterpart_pending")
    elif fetch_status != "success":
        issues.append("japanese_counterpart_fetch_failed")
    correction_eligible = bool(
        not any(value.startswith("english_source_") for value in issues)
        and fetch_status == "success"
        and code_match
        and date_match
        and side_match
        and regime_match
        and limit_numeric_mismatch
    )
    if correction_eligible:
        status = "corrected_candidate"
    elif fetch_status == "pending":
        status = "pending"
    elif fetch_status != "success":
        status = "counterpart_fetch_failed"
    else:
        status = "tie_out_failed"

    reconciliation = {
        "reconciliation_id": candidate["reconciliation_id"],
        "publication_date": candidate["publication_date"],
        "event_type": candidate["event_type"],
        "code": candidate["code"],
        "expected_multiplier": candidate["expected_multiplier"],
        "english_regime_mismatch": candidate["english_regime_mismatch"],
        "english_numeric_consistency_mismatch": candidate["english_numeric_consistency_mismatch"],
        "english_quality_flag_mismatch": candidate["english_quality_flag_mismatch"],
        "english_side": candidate.get("side", ""),
        "english_base_price_yen": candidate.get("base_price_yen"),
        "english_expanded_range_yen": candidate.get("expanded_range_yen"),
        "english_normal_range_yen": candidate.get("normal_range_yen"),
        "english_observed_multiplier": candidate.get("observed_multiplier"),
        "english_source_url": candidate.get("source_url", ""),
        "english_raw_path": english.get("raw_path", ""),
        "english_sha256": english.get("sha256", ""),
        "english_bytes": english.get("bytes", 0),
        "japanese_publication_date": japanese_date,
        "japanese_side": japanese_side,
        "japanese_base_price_yen": japanese.get("base_price_yen"),
        "japanese_expanded_range_yen": japanese.get("expanded_range_yen"),
        "japanese_normal_range_yen": japanese.get("normal_range_yen"),
        "japanese_observed_multiplier": jp_multiplier,
        "japanese_source_url": candidate["japanese_source_url"],
        "japanese_raw_path": counterpart_state.get("raw_path", ""),
        "japanese_sha256": counterpart_state.get("sha256", ""),
        "japanese_bytes": counterpart_state.get("bytes", 0),
        "code_match": code_match,
        "date_match": date_match,
        "side_match": side_match,
        "japanese_regime_match": regime_match,
        "bilingual_limit_numeric_mismatch": limit_numeric_mismatch,
        "bilingual_base_price_mismatch": base_numeric_mismatch,
        "correction_eligible": correction_eligible,
        "reconciliation_status": status,
        "issues": "|".join(issues),
    }
    if not correction_eligible:
        return reconciliation, None

    corrected = dict(candidate)
    for transient in (
        "expected_multiplier",
        "english_regime_mismatch",
        "english_numeric_consistency_mismatch",
        "english_quality_flag_mismatch",
        "japanese_source_url",
        "reconciliation_id",
    ):
        corrected.pop(transient, None)
    corrected.update(
        {
            "base_price_yen": japanese.get("base_price_yen"),
            "expanded_range_yen": jp_expanded,
            "normal_range_yen": jp_normal,
            "observed_multiplier": jp_multiplier,
            "source_url": candidate["japanese_source_url"],
            "reconciliation_id": candidate["reconciliation_id"],
            "candidate_status": "bilingual_corrected_candidate",
            "candidate_only": True,
            "promotion_allowed": False,
            "correction_reason": "official_english_jpx_numeric_values_stale",
            "english_source_url": candidate["source_url"],
            "english_source_sha256": english["sha256"],
            "japanese_source_url": candidate["japanese_source_url"],
            "japanese_source_sha256": counterpart_state["sha256"],
            "original_english_base_price_yen": candidate.get("base_price_yen"),
            "original_english_expanded_range_yen": candidate.get("expanded_range_yen"),
            "original_english_normal_range_yen": candidate.get("normal_range_yen"),
            "original_english_observed_multiplier": candidate.get("observed_multiplier"),
        }
    )
    return reconciliation, corrected


def _retain_english_source(
    candidate: dict[str, Any], page_state: pd.DataFrame, output_root: Path, issues: list[str]
) -> dict[str, Any]:
    matches = page_state.loc[page_state["url"].eq(candidate["source_url"])]
    if len(matches) != 1:
        issues.append("english_source_state_missing" if matches.empty else "english_source_state_ambiguous")
        return {}
    state = matches.iloc[0].to_dict()
    source = Path(str(state.get("raw_path", "")))
    declared_digest = str(state.get("sha256", ""))
    if state.get("terminal_status") != "success" or not source.is_file():
        issues.append("english_source_raw_missing")
        return {}
    actual_digest = _sha256(source)
    if actual_digest != declared_digest:
        issues.append("english_source_hash_mismatch")
        return {}
    destination = (
        output_root
        / "raw"
        / "english"
        / f"{candidate['reconciliation_id']}_{actual_digest[:16]}.html"
    )
    if not destination.is_file() or _sha256(destination) != actual_digest:
        _atomic_bytes(destination, source.read_bytes())
    return {"raw_path": str(destination), "sha256": actual_digest, "bytes": destination.stat().st_size}


def _read_existing_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, keep_default_na=False)
    if not set(FETCH_STATE_COLUMNS).issubset(frame.columns):
        return {}
    return {str(row["japanese_source_url"]): row for row in frame.to_dict("records")}


def _fetch_state_row(
    counterpart_id: str,
    url: str,
    *,
    http_status: int = 0,
    terminal_status: str,
    content_type: str = "",
    raw_path: str = "",
    sha256: str = "",
    byte_count: int = 0,
    error: str = "",
) -> dict[str, Any]:
    return {
        "counterpart_id": counterpart_id,
        "japanese_source_url": url,
        "http_status": http_status,
        "terminal_status": terminal_status,
        "content_type": content_type,
        "raw_path": raw_path,
        "sha256": sha256,
        "bytes": byte_count,
        "error": error,
    }


def _japanese_publication_date(soup: BeautifulSoup, text: str, source_url: str) -> str:
    url_date = _notice_url_date(source_url)
    if url_date:
        return url_date
    for node in soup.find_all("time"):
        parsed = _parse_date_text(str(node.get("datetime", "")) or node.get_text(" ", strip=True))
        if parsed:
            return parsed
    for meta in soup.find_all("meta"):
        if str(meta.get("property", "")).lower() in {"article:published_time", "date"}:
            parsed = _parse_date_text(str(meta.get("content", "")))
            if parsed:
                return parsed
    return _parse_date_text(text[:1000])


def _notice_url_date(source_url: str) -> str:
    match = re.search(r"/(20[0-9]{6})-[0-9]+\.html(?:\?.*)?$", source_url)
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
    except ValueError:
        return ""


def _parse_date_text(value: str) -> str:
    match = re.search(r"(20\d{2})\s*(?:年|[-/.])\s*(\d{1,2})\s*(?:月|[-/.])\s*(\d{1,2})\s*日?", value)
    if not match:
        return ""
    try:
        return date(*(int(part) for part in match.groups())).isoformat()
    except ValueError:
        return ""


def _fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "quant-lab-jpx-price-limit-bilingual-reconciliation/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), response.read(MAX_SOURCE_BYTES + 1), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(MAX_SOURCE_BYTES + 1), dict(exc.headers.items())


def _header(headers: dict[str, str], name: str) -> str:
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    return lowered.get(name.lower(), "")


def _integer(value: str) -> int:
    return int(value.replace(",", ""))


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _close(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and abs(left - right) < 1e-9


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


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
