from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from quant_lab.data.japan_tick_assignment_history import (
    CONCORDIA_URL,
    CORPORATE_EXCEPTION_EVENTS,
    CORPORATE_EXCEPTION_SOURCES,
    CUTOFF_MASTER_URL,
    archive_urls,
)


UTC = timezone.utc


@dataclass(frozen=True)
class JapanTickAssignmentValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_japan_tick_assignment_validation(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_tick_assignment_history_validation",
) -> JapanTickAssignmentValidationArtifacts:
    producer = Path(producer_dir).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    producer_manifest_path = producer / "manifest.json"
    source_manifest = _json(producer_manifest_path)
    issues: list[str] = []
    if run_id == source_manifest.get("run_id"):
        issues.append("independence_failure_same_run_id")
    if not (
        source_manifest.get("status") == "partial_candidate"
        and source_manifest.get("component") == "pit_narrow_tick_assignment"
        and source_manifest.get("official_jpx_source") is True
        and source_manifest.get("scorecard_eligible") is False
        and source_manifest.get("independent_validation_required") is True
        and not source_manifest.get("producer_blockers")
    ):
        issues.append("producer_partial_contract_invalid")
    for filename, digest in source_manifest.get("output_sha256", {}).items():
        path = producer / filename
        if not path.is_file() or _sha256(path) != digest:
            issues.append(f"producer_output_hash_mismatch:{filename}")
    checkpoint_path = Path(str(source_manifest.get("checkpoint_source", "")))
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != source_manifest.get("checkpoint_source_sha256"):
        issues.append("checkpoint_source_hash_mismatch")

    archives = pd.read_csv(producer / "archive_source_manifest.csv")
    notices = pd.read_csv(producer / "notice_source_manifest.csv")
    corporate = pd.read_csv(producer / "corporate_source_manifest.csv")
    produced_events = pd.read_csv(producer / "assignment_events.csv", dtype={"code": str})
    produced_reconciliation = pd.read_csv(producer / "checkpoint_reconciliation.csv", dtype={"missing_codes": str, "unexpected_codes": str})
    produced_state = set(pd.read_csv(producer / "cutoff_assignment_state.csv", dtype={"code": str})["code"])
    expected_archive_urls = archive_urls()
    if len(archives) != 154 or archives["url"].tolist() != expected_archive_urls:
        issues.append("archive_source_url_denominator_or_order_invalid")
    if len(notices) != 41 or notices["url"].nunique() != 41:
        issues.append("notice_source_denominator_invalid")
    if len(corporate) != 8 or corporate["url"].nunique() != 8:
        issues.append("corporate_source_denominator_invalid")
    for frame, label in ((archives, "archive"), (notices, "notice"), (corporate, "corporate")):
        for row in frame.to_dict("records"):
            path = Path(str(row["raw_path"]))
            if int(row["http_status"]) != 200 or not path.is_file() or _sha256(path) != row["sha256"]:
                issues.append(f"{label}_raw_source_invalid:{row['source_id']}")

    discovered = _independent_notice_discovery(archives, source_manifest["cutoff_date"])
    notice_urls = set(notices.loc[notices["url"].ne(CONCORDIA_URL), "url"])
    if discovered != notice_urls or len(discovered) != 40:
        issues.append(f"independent_notice_discovery_mismatch:{len(discovered)}/{len(notice_urls)}")
    independent_events = _independent_event_parse(notices)
    produced_keys = set(
        produced_events.loc[produced_events["source_type"].eq("tick_assignment_notice")]
        .apply(lambda row: (row["effective_date"], row["action"], row["code"], row["source_url"]), axis=1)
    )
    if independent_events != produced_keys:
        issues.append(
            f"independent_notice_event_set_mismatch:{len(independent_events)}/{len(produced_keys)}"
        )
    supplement = produced_events.loc[produced_events["source_url"].eq(CONCORDIA_URL)]
    if not (
        len(supplement) == 1
        and supplement.iloc[0]["effective_date"] == "2016-04-01"
        and supplement.iloc[0]["action"] == "REMOVE"
        and supplement.iloc[0]["code"] == "8332"
    ):
        issues.append("concordia_supplement_invalid")
    expected_exceptions = {
        (effective, action, code, CORPORATE_EXCEPTION_SOURCES[source_id])
        for effective, action, code, _, source_id in CORPORATE_EXCEPTION_EVENTS
    }
    observed_exceptions = set(
        produced_events.loc[produced_events["source_type"].eq("corporate_action_exception")]
        .apply(lambda row: (row["effective_date"], row["action"], row["code"], row["source_url"]), axis=1)
    )
    if observed_exceptions != expected_exceptions:
        issues.append("corporate_exception_event_set_invalid")
    if len(produced_events) != 145 or produced_events.duplicated(["effective_date", "code", "action"]).any():
        issues.append("assignment_event_denominator_or_uniqueness_invalid")

    checkpoints = pd.read_csv(checkpoint_path, dtype={"code": str})
    recomputed_reconciliation, recomputed_state = _independent_replay(produced_events, checkpoints)
    comparison_columns = [
        "effective_date", "assignment_scope", "checkpoint_basis", "expected_names",
        "replayed_names", "missing_names", "unexpected_names", "exact_set_match",
    ]
    if not recomputed_reconciliation[comparison_columns].equals(produced_reconciliation[comparison_columns]):
        issues.append("checkpoint_reconciliation_recompute_mismatch")
    if len(recomputed_reconciliation) != 14 or not recomputed_reconciliation["exact_set_match"].all():
        issues.append("checkpoint_exact_set_validation_failed")
    if recomputed_state != produced_state or len(recomputed_state) != 493:
        issues.append("cutoff_assignment_state_recompute_mismatch")
    master_row = corporate.loc[corporate["url"].eq(CUTOFF_MASTER_URL)]
    master = pd.read_csv(Path(master_row.iloc[0]["raw_path"]), encoding="cp932", dtype={"Code": str}) if len(master_row) == 1 else pd.DataFrame()
    eligible = {"TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"}
    master_codes = set(master.loc[master["New Index Series Code"].isin(eligible), "Code"].dropna()) if not master.empty else set()
    if master_codes != produced_state or len(master_codes) != 493:
        issues.append("official_20260529_master_set_mismatch")

    remaining = list(source_manifest.get("remaining_completion_blockers", []))
    summary = pd.DataFrame(
        [
            {"check": "official_archive_sources", "value": len(archives), "passed": len(archives) == 154},
            {"check": "unique_notice_pages", "value": len(discovered), "passed": len(discovered) == 40},
            {"check": "assignment_events", "value": len(produced_events), "passed": len(produced_events) == 145},
            {"check": "checkpoint_exact_matches", "value": int(recomputed_reconciliation["exact_set_match"].sum()), "passed": bool(recomputed_reconciliation["exact_set_match"].all())},
            {"check": "cutoff_assignment_names", "value": len(recomputed_state), "passed": len(recomputed_state) == 493},
            {"check": "official_20260529_master_exact_match", "value": len(master_codes), "passed": master_codes == produced_state},
        ]
    )
    summary_path = root / "validation_summary.csv"
    _atomic_csv(summary_path, summary)
    manifest = {
        "schema_version": "1.0-independent-partial",
        "run_id": run_id,
        "producer_run_id": source_manifest.get("run_id"),
        "artifact_type": "japan_tick_assignment_history_independent_validation",
        "status": "pass_partial_component" if not issues else "failed_validation",
        "component": "pit_narrow_tick_assignment",
        "independent_validation": True,
        "official_jpx_source": True,
        "component_scorecard_eligible": False,
        "scorecard_eligible": False,
        "validation_issues": issues,
        "remaining_completion_blockers": remaining,
        "critical_blockers": [*issues, *remaining],
        "counts": source_manifest.get("counts", {}),
        "producer_manifest": str(producer_manifest_path),
        "producer_manifest_sha256": _sha256(producer_manifest_path),
        "artifacts": {"validation_summary": str(summary_path)},
        "output_sha256": {summary_path.name: _sha256(summary_path)},
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanTickAssignmentValidationArtifacts(run_id, root, manifest_path, manifest)


def _independent_notice_discovery(archives: pd.DataFrame, cutoff_date: str) -> set[str]:
    cutoff = date.fromisoformat(cutoff_date)
    urls: set[str] = set()
    for row in archives.to_dict("records"):
        payload = json.loads(Path(row["raw_path"]).read_text(encoding="utf-8"))
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, dict):
                title = str(value.get("title", ""))
                target = str(value.get("url", ""))
                stamp = value.get("updated_date", {})
                if re.match(r"^Handling of Tick Sizes from\b", title, re.I) and target:
                    try:
                        published = date(int(stamp["year"]), int(stamp["month"]), int(stamp["day"]))
                    except (KeyError, TypeError, ValueError):
                        published = None
                    if published and published <= cutoff:
                        urls.add("https://www.jpx.co.jp" + target)
                stack.extend(value.values())
    return urls


def _independent_event_parse(notices: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    rows: set[tuple[str, str, str, str]] = set()
    for source in notices.loc[notices["url"].ne(CONCORDIA_URL)].to_dict("records"):
        soup = BeautifulSoup(Path(source["raw_path"]).read_bytes(), "html.parser")
        for table in soup.find_all("table"):
            previous = table.find_previous(["p", "h3", "h4"])
            context = " ".join(previous.stripped_strings).lower() if previous else ""
            action = "REMOVE" if "no longer" in context else "ADD" if ("newly" in context or "will be subject" in context) else ""
            if not action:
                continue
            for tr in table.find_all("tr"):
                cells = [" ".join(cell.stripped_strings) for cell in tr.find_all(["th", "td"])]
                code = cells[0].strip().upper() if cells else ""
                if len(cells) >= 2 and re.fullmatch(r"(?=.*\d)[0-9A-Z]{4}", code):
                    rows.add((source["effective_date"], action, code, source["url"]))
    return rows


def _independent_replay(events: pd.DataFrame, checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    state = set(checkpoints.loc[checkpoints["effective_date"].eq("2014-01-14") & checkpoints["index_scope"].eq("TOPIX100"), "code"])
    pending = events.sort_values(["effective_date", "code", "action"]).to_dict("records")
    rows = []
    dates = sorted(value for value in checkpoints["effective_date"].unique() if value <= "2026-06-30")
    for stamp in dates:
        if stamp == "2023-05-26":
            continue
        basis = "official_full_topix100_snapshot"
        if stamp == "2023-06-05":
            mid = set(checkpoints.loc[checkpoints["effective_date"].eq(stamp) & checkpoints["index_scope"].eq("TOPIX_MID400"), "code"])
            composite = set(state) | mid
            state = set(composite)
            basis = "replayed_topix100_plus_official_mid400_seed"
        for event in pending:
            if event.get("applied") or event["effective_date"] > stamp:
                continue
            state.add(event["code"]) if event["action"] == "ADD" else state.discard(event["code"])
            event["applied"] = True
        scope = "TOPIX500" if stamp >= "2023-06-05" else "TOPIX100"
        expected = composite if scope == "TOPIX500" else set(checkpoints.loc[checkpoints["effective_date"].eq(stamp) & checkpoints["index_scope"].eq("TOPIX100"), "code"])
        rows.append({
            "effective_date": stamp, "assignment_scope": scope, "checkpoint_basis": basis,
            "expected_names": len(expected), "replayed_names": len(state),
            "missing_names": len(expected - state), "unexpected_names": len(state - expected),
            "exact_set_match": state == expected,
        })
    for event in pending:
        if event.get("applied") or event["effective_date"] > "2026-06-30":
            continue
        state.add(event["code"]) if event["action"] == "ADD" else state.discard(event["code"])
    return pd.DataFrame(rows), state


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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
