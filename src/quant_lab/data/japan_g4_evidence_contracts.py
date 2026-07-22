from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any


G4_START = date(2006, 7, 1)
G4_END = date(2026, 6, 30)
PROMOTION_KEYS = ("alpha", "historical_backtest", "production", "ibkr", "live_trading")


def full_component_blockers(
    root: Path,
    manifest: dict[str, Any],
    *,
    expected_component: str,
) -> list[str]:
    """Return fail-closed blockers for a purported full G4 component validation."""

    prefix = expected_component
    blockers: list[str] = []
    promotion = manifest.get("promotion_policy", {})
    critical = manifest.get("critical_blockers", [])
    if not (
        manifest.get("status") == "pass_component"
        and manifest.get("component") == expected_component
        and manifest.get("component_scorecard_eligible") is True
        and manifest.get("scorecard_eligible") is False
        and manifest.get("independent_validation") is True
        and manifest.get("critical_blocker_count") == 0
        and isinstance(critical, list)
        and not critical
        and all(promotion.get(key) is False for key in PROMOTION_KEYS)
    ):
        blockers.append(f"{prefix}_validation_contract_invalid")

    blockers.extend(_coverage_blockers(manifest, prefix))
    blockers.extend(_artifact_hash_blockers(root, manifest, prefix))
    blockers.extend(_producer_lineage_blockers(manifest, prefix))
    blockers.extend(_semantic_blockers(manifest, prefix))
    return list(dict.fromkeys(blockers))


def _coverage_blockers(manifest: dict[str, Any], prefix: str) -> list[str]:
    coverage = manifest.get("coverage", {})
    if not isinstance(coverage, dict) or coverage.get("complete") is not True:
        return [f"{prefix}_coverage_not_complete"]
    start = _date_value(coverage.get("start_date", coverage.get("start")))
    end = _date_value(coverage.get("end_date", coverage.get("end")))
    blockers = []
    if start is None or start > G4_START:
        blockers.append(f"{prefix}_coverage_starts_after_2006_07_01")
    if end is None or end < G4_END:
        blockers.append(f"{prefix}_coverage_ends_before_2026_06_30")
    return blockers


def _artifact_hash_blockers(
    root: Path, manifest: dict[str, Any], prefix: str
) -> list[str]:
    artifacts = manifest.get("artifacts", {})
    hashes = manifest.get("output_sha256", {})
    if not isinstance(artifacts, dict) or not artifacts:
        return [f"{prefix}_artifacts_missing"]
    if not isinstance(hashes, dict) or not hashes:
        return [f"{prefix}_hash_inventory_empty"]

    blockers: list[str] = []
    hashed_paths: set[Path] = set()
    for filename, expected in hashes.items():
        relative = Path(str(filename))
        if relative.is_absolute() or ".." in relative.parts or not _digest(expected):
            blockers.append(f"{prefix}_hash_inventory_invalid:{filename}")
            continue
        path = (root / relative).resolve()
        if not _within(path, root) or not path.is_file() or _sha256(path) != expected:
            blockers.append(f"{prefix}_output_hash_mismatch:{filename}")
            continue
        hashed_paths.add(path)

    for name, value in artifacts.items():
        path = Path(str(value)).resolve()
        if not _within(path, root) or not path.is_file():
            blockers.append(f"{prefix}_artifact_path_invalid:{name}")
        elif path not in hashed_paths:
            blockers.append(f"{prefix}_artifact_not_hash_verified:{name}")
    return blockers


def _producer_lineage_blockers(manifest: dict[str, Any], prefix: str) -> list[str]:
    path = Path(str(manifest.get("producer_manifest", "")))
    expected = manifest.get("producer_manifest_sha256")
    if not path.is_file() or not _digest(expected) or _sha256(path) != expected:
        return [f"{prefix}_producer_lineage_mismatch"]

    producer = _json(path)
    hashes = producer.get("output_sha256", producer.get("artifact_sha256", {}))
    if not isinstance(hashes, dict) or not hashes:
        return [f"{prefix}_producer_hash_inventory_empty"]
    blockers = []
    for filename, digest in hashes.items():
        relative = Path(str(filename))
        candidate = (path.parent / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not _digest(digest)
            or not _within(candidate, path.parent)
            or not candidate.is_file()
            or _sha256(candidate) != digest
        ):
            blockers.append(f"{prefix}_producer_output_hash_mismatch:{filename}")
    return blockers


def _semantic_blockers(manifest: dict[str, Any], component: str) -> list[str]:
    counts = manifest.get("counts", {})
    official = manifest.get("official_jpx_source") is True
    blockers: list[str] = []
    if component == "halt_suspension_events":
        source_rows = _positive_int(counts.get("source_rows"))
        event_rows = _positive_int(counts.get("event_rows"))
        if not (
            official
            and manifest.get("historical_archive_complete") is True
            and manifest.get("source_denominator_exact") is True
            and source_rows is not None
            and source_rows == event_rows
        ):
            blockers.append("halt_suspension_events_semantic_contract_invalid")
    elif component == "jpy_currency_identity":
        if not (
            official
            and manifest.get("currency") == "JPY"
            and manifest.get("currency_identity") == "JPY"
            and _positive_int(counts.get("source_rows")) is not None
        ):
            blockers.append("jpy_currency_identity_semantic_contract_invalid")
    elif component == "tick_size_rules":
        if not (
            official
            and manifest.get("pit_valid_time_complete") is True
            and _positive_int(counts.get("rule_rows")) is not None
        ):
            blockers.append("tick_size_rules_semantic_contract_invalid")
    elif component == "pit_narrow_tick_assignment":
        if not (
            official
            and manifest.get("pit_valid_time_complete") is True
            and _positive_int(counts.get("assignment_rows")) is not None
        ):
            blockers.append("pit_narrow_tick_assignment_semantic_contract_invalid")
    elif component == "normal_price_limits":
        if not (
            official
            and manifest.get("normal_price_limits_complete") is True
            and _positive_int(counts.get("rule_rows")) is not None
        ):
            blockers.append("normal_price_limits_semantic_contract_invalid")
    elif component == "exceptional_price_limits":
        if not (
            official
            and manifest.get("exceptional_price_limits_complete") is True
            and manifest.get("pre_2013_archive_complete") is True
            and manifest.get("post_2020_return_to_normal_reconciled") is True
            and _positive_int(counts.get("event_rows")) is not None
        ):
            blockers.append("exceptional_price_limits_semantic_contract_invalid")
    elif component == "tca_assumptions":
        samples = _positive_int(counts.get("calibration_rows"))
        entities = _positive_int(counts.get("entities"))
        if not (
            manifest.get("currency") == "JPY"
            and manifest.get("calibration_type")
            == "japan_liquidity_aware_market_data_calibration"
            and manifest.get("liquidity_aware") is True
            and manifest.get("minimum_sample_requirements_met") is True
            and samples is not None
            and entities is not None
        ):
            blockers.append("tca_assumptions_semantic_contract_invalid")
    else:
        blockers.append(f"unsupported_full_component:{component}")
    return blockers


def _date_value(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _digest(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _json(path: Path) -> dict[str, Any]:
    import json

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
