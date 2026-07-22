from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

from quant_lab.data.japan_market_rule_history import (
    SOURCES,
    MID400_SNAPSHOT_DATES,
    TOPIX100_SNAPSHOT_DATES,
    price_limit_rule_bands,
    price_limit_transition_evidence,
    rule_amendment_ledger,
    tick_regime_windows,
    tick_rule_bands,
    tick_transition_evidence,
)


UTC = timezone.utc
HISTORICAL_SOURCE_SHA256 = {
    "tick_grid_baseline_20060501": "62ee15e4a343b8a8dac6e549620eaa29cf9a4c08df10aeffb67a4b19c33dec1a",
    "tick_grid_current_20060926_partial": "d67697bb6e464e5374ca4cd1470c922b18bbffe1f8333236c5bea8ff3fc700c4",
    "rules_amendment_20070930": "54d3e27581e199501a8757e10e3c2daf508d6aa9c481afe9cbbdff55598e472d",
    "rules_amendment_20080115": "8c44fb9c5908fe52104c254ee3fab25b2270c928409ba6ec161d463ff4f23ebd",
    "tick_revision_proposal_20080129": "04041e5103678f56e09e49d74183e0bc5b7765c982e48bd332ad598764e49d79",
    "price_limit_amendment_20080617": "4bc0bb6a84329cf50323b5f3586c0d71d9b93625598b824aaceec60bbfa4776a",
    "tick_amendment_20080722_summary": "0ae6396f1b0647c251f8c3eefec58531b22f3690512fc21cb79359621e29b2b7",
    "tick_amendment_20080722_rules": "1b98acc7f278028ecc442ee91db002118a90932148897839330d8b65a998e4f2",
    "rules_amendment_20080707": "4ec2b3705675be800ac1b13d8245a82da0d48dddeca0cea694016bd2ef623e95",
    "rules_amendment_20090105": "54339ecc4ce11846056c8fa45cbf7793b32b556d37de3fce2fd060eabe16deec",
    "tick_amendment_20100104_rules": "11166b1d5024f81b5457b1a4352303b037b2088c49e3b8f98d09a31c49b2ec80",
    "price_limit_amendment_20091230": "00d4438d1532405fc60111421d5cb514599c78dd25d6a82df88ae79c05e7a8f2",
    "price_limit_amendment_20060501": "df9efe1a28c11feaba4d8548793b86923e113c78a7f168d8d2f8abfebb4ffb70",
    "price_limit_arrowhead_outline_20100104": "2cbda1037896a725e112a0863e432545c7edf4a42d1265f94a33c36ca298cd8f",
    "price_limit_persistence_20100301": "e4cc3936bff4cd2b974e2b501e2df36d8b50f44cc731e10c8ae7ea3f65824cd1",
}


@dataclass(frozen=True)
class JapanMarketRuleHistoryValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_japan_market_rule_history_validation(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_market_rule_history_validation",
) -> JapanMarketRuleHistoryValidationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    producer = Path(producer_dir).resolve()
    root = Path(artifact_root) / run_id
    if (root / "manifest.json").exists():
        raise FileExistsError(f"immutable artifact already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source_manifest_path = producer / "manifest.json"
    source_manifest = _json(source_manifest_path)
    issues = []
    if run_id == source_manifest.get("run_id"):
        issues.append("independence_failure_same_run_id")
    if not (
        source_manifest.get("status") == "partial_candidate"
        and source_manifest.get("schema_version") == "2.0-source-backed-rule-continuity"
        and source_manifest.get("scorecard_eligible") is False
        and source_manifest.get("independent_validation_required") is True
        and source_manifest.get("component") == "price_limits"
        and not source_manifest.get("producer_blockers")
    ):
        issues.append("producer_partial_contract_invalid")
    for filename, digest in source_manifest.get("output_sha256", {}).items():
        path = producer / filename
        if not path.is_file() or _sha256(path) != digest:
            issues.append(f"producer_output_hash_mismatch:{filename}")
    sources = pd.read_csv(producer / "source_manifest.csv")
    observed_source_urls = dict(zip(sources["source_id"], sources["url"]))
    if observed_source_urls != SOURCES:
        issues.append("official_source_id_or_url_set_mismatch")
    for row in sources.to_dict("records"):
        path = Path(row["raw_path"])
        if not path.is_file() or _sha256(path) != row["sha256"]:
            issues.append(f"raw_source_hash_mismatch:{row['source_id']}")
    issues.extend(_historical_source_hash_issues(sources))
    issues.extend(_source_anchor_issues(sources))

    windows = pd.read_csv(producer / "tick_regime_windows.csv")
    bands = pd.read_csv(producer / "tick_rule_bands.csv")
    transitions = pd.read_csv(producer / "tick_transition_evidence.csv")
    amendment_ledger = pd.read_csv(producer / "rule_amendment_ledger.csv")
    price_limit_transitions = pd.read_csv(producer / "price_limit_transition_evidence.csv")
    assignments = pd.read_csv(producer / "assignment_snapshots.csv", dtype={"code": str})
    price_limits = pd.read_csv(producer / "price_limit_rule_bands.csv")
    if not windows.equals(tick_regime_windows()):
        issues.append("tick_regime_windows_recompute_mismatch")
    expected_bands = tick_rule_bands()
    if list(bands.columns) != list(expected_bands.columns) or not bands.fillna("").equals(expected_bands.fillna("")):
        issues.append("tick_rule_bands_recompute_mismatch")
    expected_transitions = tick_transition_evidence()
    if list(transitions.columns) != list(expected_transitions.columns) or not transitions.equals(expected_transitions):
        issues.append("tick_transition_evidence_recompute_mismatch")
    expected_ledger = rule_amendment_ledger()
    if list(amendment_ledger.columns) != list(expected_ledger.columns) or not amendment_ledger.equals(expected_ledger):
        issues.append("rule_amendment_ledger_recompute_mismatch")
    expected_price_limit_transitions = price_limit_transition_evidence()
    if (
        list(price_limit_transitions.columns) != list(expected_price_limit_transitions.columns)
        or not price_limit_transitions.equals(expected_price_limit_transitions)
    ):
        issues.append("price_limit_transition_evidence_recompute_mismatch")
    counts = assignments.groupby("index_scope").size().to_dict()
    if counts != {
        "TOPIX100": 100 * (2 + len(TOPIX100_SNAPSHOT_DATES)),
        "TOPIX_MID400": 398 * len(MID400_SNAPSHOT_DATES),
    }:
        issues.append(f"assignment_snapshot_denominator_invalid:{counts}")
    if assignments.duplicated(["effective_date", "index_scope", "code"]).any():
        issues.append("assignment_snapshot_duplicate_keys")
    if assignments["code"].str.fullmatch(r"[0-9A-Z]{4}").ne(True).any():
        issues.append("assignment_snapshot_invalid_codes")
    topix_sets = {
        date: set(frame["code"])
        for date, frame in assignments.loc[assignments["index_scope"].eq("TOPIX100")].groupby("effective_date")
    }
    if topix_sets.get("2014-01-14") != topix_sets.get("2014-07-22"):
        issues.append("topix100_baseline_checkpoint_set_mismatch")
    mid_sets = {
        date: set(frame["code"])
        for date, frame in assignments.loc[assignments["index_scope"].eq("TOPIX_MID400")].groupby("effective_date")
    }
    if mid_sets.get("2023-05-26") != mid_sets.get("2023-06-05"):
        issues.append("mid400_checkpoint_set_mismatch")
    issues.extend(_price_limit_table_issues(price_limits))
    issues.extend(_tick_continuity_issues(windows, bands, transitions, amendment_ledger))

    remaining = list(source_manifest.get("remaining_completion_blockers", []))
    manifest = {
        "schema_version": "2.0-independent-rule-continuity",
        "run_id": run_id,
        "validator_run_id": run_id,
        "producer_run_id": source_manifest.get("run_id"),
        "artifact_type": "japan_market_rule_history_independent_rule_continuity_validation",
        "status": "pass_partial_component" if not issues else "failed_validation",
        "component": "price_limits",
        "independent_validation": True,
        "component_scorecard_eligible": False,
        "scorecard_eligible": False,
        "official_jpx_source": True,
        "validated_lanes": list(source_manifest.get("completed_lanes", [])),
        "validation_issues": issues,
        "remaining_completion_blockers": remaining,
        "critical_blockers": [*issues, *remaining],
        "counts": source_manifest.get("counts", {}),
        "producer_manifest": str(source_manifest_path),
        "producer_manifest_sha256": _sha256(source_manifest_path),
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return JapanMarketRuleHistoryValidationArtifacts(run_id, root, manifest_path, manifest)


def _source_anchor_issues(sources: pd.DataFrame) -> list[str]:
    paths = {row["source_id"]: Path(row["raw_path"]) for row in sources.to_dict("records")}
    issues = []
    anchors = {
        "phase_1_2_outline": ("January 2014", "July 2014", "TOPIX100"),
        "phase_3_rules": ("September 24, 2015", "Phase III", "TOPIX100"),
        "phase_summary": ("Phase I", "2014/7/22", "2015/9/24"),
        "price_limit_rules_20141201": (
            "as of December 1, 2014",
            "Rule 2. Price Limits of Stocks",
            "Less than 100 yen",
            "Where the price is more than 50,000,000 yen",
        ),
        "topix100_baseline_20131201": (
            "TOPIX100 Constituents (as of December 1, 2013)",
            "The constituents may change due to events such as delisting",
        ),
        "price_limit_broadening_rules_20200803": (
            "August 3, 2020",
            "quadruple",
        ),
        "tick_amendment_20100104_rules": (
            "呼値に関する規則の一部改正新旧対照表",
            "平成２２年１月４日から施行",
            "第１４条",
            "３，０００円 以下の場合",
            "呼値の制限値幅に関する規則の一部改正新旧対照表",
            "７００円  〃  １，０００円",
        ),
        "price_limit_persistence_20100301": (
            "呼値の制限値幅に関する規則の一部改正新旧対照表",
            "第２条 （略）",
            "平成２２年３月１日から",
        ),
    }
    for source_id, expected in anchors.items():
        path = paths.get(source_id)
        text = ""
        if path and path.is_file():
            text = "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
        for anchor in expected:
            if anchor not in text:
                issues.append(f"official_source_anchor_missing:{source_id}:{anchor}")
    for source_id, anchor in (
        ("mid400_implementation", "June 5, 2023"),
        ("current_tick_rules", "TOPIX500"),
        ("current_price_limits", "Daily Price Limits"),
        ("price_limit_broadening_2020", "Partial Revision to Broadening of Daily Price Limits"),
    ):
        path = paths.get(source_id)
        text = path.read_text(encoding="utf-8", errors="ignore") if path and path.is_file() else ""
        if not re.search(re.escape(anchor), text, re.I):
            issues.append(f"official_source_anchor_missing:{source_id}:{anchor}")
    return issues


def _historical_source_hash_issues(sources: pd.DataFrame) -> list[str]:
    # Several pre-2010 PDFs use embedded font maps that do not yield stable
    # Japanese text. Pin the exact reviewed official files instead.
    observed = dict(zip(sources["source_id"], sources["sha256"]))
    return [
        f"immutable_official_source_hash_mismatch:{source_id}"
        for source_id, expected in HISTORICAL_SOURCE_SHA256.items()
        if observed.get(source_id) != expected
    ]


def _price_limit_table_issues(price_limits: pd.DataFrame) -> list[str]:
    issues = []
    required = {
        "regime_id",
        "instrument_scope",
        "valid_from",
        "valid_to",
        "base_price_lower_inclusive_yen",
        "base_price_upper_exclusive_yen",
        "daily_limit_yen",
        "source_ids",
        "continuity_status",
    }
    if not required.issubset(price_limits.columns) or len(price_limits) != 63:
        return ["price_limit_history_table_schema_or_denominator_invalid"]
    expected = price_limit_rule_bands()
    if list(price_limits.columns) != list(expected.columns) or not price_limits.fillna("").equals(expected.fillna("")):
        issues.append("price_limit_history_table_recompute_mismatch")
    counts = price_limits.groupby("regime_id").size().to_dict()
    if counts != {"arrowhead_20100104": 34, "pre_arrowhead_20060701": 29}:
        issues.append(f"price_limit_regime_denominator_invalid:{counts}")
    windows = price_limits.groupby("regime_id")[["valid_from", "valid_to"]].first().to_dict("index")
    if windows != {
        "arrowhead_20100104": {"valid_from": "2010-01-04", "valid_to": "2026-06-30"},
        "pre_arrowhead_20060701": {"valid_from": "2006-07-01", "valid_to": "2010-01-03"},
    }:
        issues.append(f"price_limit_regime_windows_invalid:{windows}")
    if not price_limits["continuity_status"].eq("verified_official_rule_and_amendment_ledger").all():
        issues.append("price_limit_continuity_status_invalid")
    return issues


def _tick_continuity_issues(
    windows: pd.DataFrame,
    bands: pd.DataFrame,
    transitions: pd.DataFrame,
    amendment_ledger: pd.DataFrame,
) -> list[str]:
    issues = []
    if len(windows) != 8 or not windows["rule_table_verified"].eq(True).all():
        issues.append("tick_continuity_window_denominator_or_status_invalid")
    counts = bands.groupby("regime_id").size().to_dict()
    expected_counts = {
        "legacy_20060501": 9,
        "legacy_20080722": 9,
        "legacy_20100104": 16,
        "other_issues_verified": 16,
        "phase_1": 16,
        "phase_2": 16,
        "phase_3_topix100": 16,
        "phase_3_topix500": 16,
    }
    if counts != expected_counts:
        issues.append(f"tick_regime_band_denominator_invalid:{counts}")
    if not transitions["continuous_interval_proven"].eq(True).all():
        issues.append("tick_transition_continuity_status_invalid")
    ledger_counts = amendment_ledger.groupby("rule_family").size().to_dict()
    if ledger_counts != {"normal_price_limit": 9, "tick_size": 6}:
        issues.append(f"amendment_ledger_family_denominator_invalid:{ledger_counts}")
    if set(amendment_ledger.loc[amendment_ledger["normal_grid_effect"].eq("grid_changed"), "effective_date"]) != {
        "2008-07-22",
        "2010-01-04",
    }:
        issues.append("amendment_ledger_grid_change_dates_invalid")
    return issues


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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
