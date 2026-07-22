from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
MATERIALIZER_VERSION = "1.0"


@dataclass(frozen=True)
class JapanG1PITMaterializerArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    entities_path: Path
    listing_windows_path: Path
    identifier_history_path: Path
    universe_membership_path: Path
    active_candidate_path: Path
    reference_candidate_path: Path
    reconciliation_path: Path
    manifest: dict[str, Any]


def write_japan_g1_pit_materializer(
    *,
    run_id: str,
    jquants_dir: str | Path,
    enrichment_dir: str | Path,
    valid_time_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g1_pit_materializer",
) -> JapanG1PITMaterializerArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(artifact_dir)
    inputs = {
        "jquants": Path(jquants_dir),
        "enrichment": Path(enrichment_dir),
        "valid_time": Path(valid_time_dir),
    }
    manifests = {name: _read_json(path / "manifest.json") for name, path in inputs.items()}
    input_sha256 = {
        name: _sha256(path / "manifest.json")
        for name, path in inputs.items()
        if (path / "manifest.json").is_file()
    }
    input_manifests = {
        name: str((path / "manifest.json").resolve()) for name, path in inputs.items()
    }
    blockers = _upstream_blockers(inputs, manifests)
    if blockers:
        manifest = _blocked_manifest(
            run_id, artifact_dir, blockers, input_sha256, input_manifests
        )
        _atomic_write_json(paths["manifest"], manifest)
        return _artifacts(run_id, artifact_dir, paths, manifest)

    jq_path = Path(str(manifests["jquants"]["artifacts"]["listed_info"]))
    listed = pd.read_csv(jq_path, low_memory=False)
    reference_path = Path(str(manifests["valid_time"]["inputs"]["reference_map"]["path"]))
    reference = pd.read_csv(reference_path, low_memory=False)
    enrichment_path = Path(str(manifests["enrichment"]["artifacts"]["normalized_responses"]))
    enrichment = pd.read_csv(enrichment_path, low_memory=False)

    normalized, normalization_blockers = _normalize_snapshots(listed)
    mapping, mapping_issues = _build_code_ric_mapping(reference, enrichment)
    windows = _reconstruct_windows(normalized)
    reconciled = windows.merge(mapping, on="local_code", how="left", validate="many_to_one")
    reconciled["mapping_status"] = reconciled["mapping_status"].fillna("unresolved_code")
    reconciled["currency"] = reconciled["price_currency"].where(
        reconciled["price_currency"].astype(str).eq("JPY"), pd.NA
    )
    reconciled["pit_usable"] = (
        reconciled["mapping_status"].eq("matched")
        & reconciled["refinitiv_ric"].notna()
        & reconciled["currency"].eq("JPY")
        & reconciled["known_at"].notna()
    )
    reconciled["blocker_reason"] = ""
    reconciled.loc[~reconciled["mapping_status"].eq("matched"), "blocker_reason"] = reconciled[
        "mapping_status"
    ]
    reconciled.loc[
        reconciled["mapping_status"].eq("matched") & reconciled["currency"].isna(),
        "blocker_reason",
    ] = "missing_source_backed_jpy_currency"

    entities = _entities(reconciled)
    listing_windows = _listing_windows(reconciled)
    identifier_history = _identifier_history(reconciled)
    membership = _membership(reconciled)
    reference_candidate = _symbol_map(reconciled)
    final_date = normalized["effective_date"].max()
    active_candidate = _symbol_map(reconciled.loc[reconciled["listing_end"].eq(final_date)])

    _atomic_write_parquet(paths["entities"], entities)
    _atomic_write_parquet(paths["listing_windows"], listing_windows)
    _atomic_write_parquet(paths["identifier_history"], identifier_history)
    _atomic_write_parquet(paths["universe_membership"], membership)
    _atomic_write_csv(paths["active_candidate"], active_candidate)
    _atomic_write_csv(paths["reference_candidate"], reference_candidate)
    _atomic_write_csv(paths["reconciliation"], reconciled)

    output_files = [
        paths["entities"],
        paths["listing_windows"],
        paths["identifier_history"],
        paths["universe_membership"],
        paths["active_candidate"],
        paths["reference_candidate"],
        paths["reconciliation"],
    ]
    output_sha256 = {path.name: _sha256(path) for path in output_files}
    all_blockers = list(normalization_blockers)
    if not mapping_issues.empty:
        all_blockers.append(f"mapping_issue_rows={len(mapping_issues)}")
    unusable = int((~reconciled["pit_usable"]).sum())
    if unusable:
        all_blockers.append(f"pit_unusable_windows={unusable}")
    manifest = {
        "schema_version": MATERIALIZER_VERSION,
        "run_id": run_id,
        "artifact_type": "japan_g1_pit_materializer_producer",
        "status": "complete_with_blockers" if all_blockers else "complete",
        "producer_run_id": run_id,
        "independent_audit": False,
        "scorecard_eligible": False,
        "evidence_scope": "producer_candidate_only",
        "official_jpx_source": True,
        "historical_pit_ready": not all_blockers,
        "known_at_policy": {
            "method": "official_jpx_daily_delivery_schedule_conservative",
            "timestamp": "08:00 Asia/Tokyo on each effective JPX snapshot date",
            "effective_date_used_directly": False,
            "retrieval_time_used": False,
            "restatement_risk": True,
            "decision_time_rule": "downstream decision timestamp must be on or after known_at",
        },
        "counts": {
            "snapshot_rows": int(len(normalized)),
            "snapshot_dates": int(normalized["effective_date"].nunique()),
            "listing_windows": int(len(reconciled)),
            "entities": int(len(entities)),
            "active_candidate_rows": int(len(active_candidate)),
            "reference_candidate_rows": int(len(reference_candidate)),
            "pit_usable_windows": int(reconciled["pit_usable"].sum()),
            "pit_unusable_windows": unusable,
            "mapping_issue_rows": int(len(mapping_issues)),
        },
        "critical_blockers": all_blockers,
        "input_manifest_sha256": input_sha256,
        "input_manifests": input_manifests,
        "output_sha256": output_sha256,
        "artifacts": {name: str(path.resolve()) for name, path in paths.items() if name != "manifest"},
        "promotion_policy": {
            "alpha": False,
            "historical_backtest": False,
            "production": False,
            "ibkr": False,
            "live_trading": False,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(paths["manifest"], manifest)
    return _artifacts(run_id, artifact_dir, paths, manifest)


def conservative_snapshot_known_at(effective_date: str | date) -> str:
    value = date.fromisoformat(str(effective_date)) if not isinstance(effective_date, date) else effective_date
    local = datetime.combine(value, time(hour=8), tzinfo=JST)
    return local.astimezone(UTC).isoformat()


def _upstream_blockers(inputs: dict[str, Path], manifests: dict[str, dict[str, Any]]) -> list[str]:
    blockers: list[str] = []
    jq = manifests["jquants"]
    quality = jq.get("quality_metrics", {}) if isinstance(jq.get("quality_metrics"), dict) else {}
    planned = int(quality.get("planned_dates", 0) or 0)
    if not (
        jq.get("status") == "complete"
        and jq.get("dataset_id") == "jquants_pro_v2_listed_info"
        and planned > 0
        and int(quality.get("completed_dates", 0) or 0) == planned
        and int(quality.get("pending_dates", -1)) == 0
        and int(quality.get("failed_dates", -1)) == 0
        and float(quality.get("date_coverage_ratio", 0.0)) == 1.0
        and int(quality.get("normalized_rows", 0) or 0) > 0
        and _verify_jquants_hashes(jq)
    ):
        blockers.append("jquants_listed_info_not_complete_and_hash_verified")
    enrichment = manifests["enrichment"]
    counts = enrichment.get("counts", {}) if isinstance(enrichment.get("counts"), dict) else {}
    if not (
        enrichment.get("status") == "complete_current_observation"
        and enrichment.get("current_observation_only") is True
        and int(counts.get("success", 0) or 0) == int(counts.get("source_unique_rics", -1) or -1)
        and int(counts.get("error", -1)) == 0
    ):
        blockers.append("current_enrichment_not_complete")
    valid = manifests["valid_time"]
    if not (
        valid.get("status") == "complete_valid_time_baseline_current_observation_only"
        and valid.get("valid_time_baseline_ready") is True
        and valid.get("historical_pit_ready") is False
    ):
        blockers.append("valid_time_baseline_not_complete")
    for name, path in inputs.items():
        if not (path / "manifest.json").is_file():
            blockers.append(f"{name}_manifest_missing")
    return list(dict.fromkeys(blockers))


def _verify_jquants_hashes(manifest: dict[str, Any]) -> bool:
    root = Path(str(manifest.get("artifact_dir", "")))
    entries = manifest.get("sha256", {})
    if not root.is_dir() or not isinstance(entries, dict) or not entries:
        return False
    for relative, digest in entries.items():
        path = root / str(relative)
        if not path.is_file() or _sha256(path) != str(digest).lower():
            return False
    checksums = root / "checksums.sha256"
    return checksums.is_file() and _sha256(checksums) == str(
        manifest.get("checksums_file_sha256", "")
    ).lower()


def _normalize_snapshots(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = {"effective_date", "code", "market_code"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("listed-info rows or required snapshot columns are missing")
    result = frame.copy()
    result["effective_date"] = pd.to_datetime(result["effective_date"], errors="coerce").dt.date
    result["code"] = result["code"].astype(str).str.strip().str.upper()
    result["local_code"] = result["code"].map(_local_code)
    if result["effective_date"].isna().any() or result["local_code"].eq("").any():
        raise ValueError("snapshot dates/codes are invalid")
    payload_columns = [column for column in result.columns if column not in {"request_date", "observed_at_utc"}]
    conflicts = result.groupby(["effective_date", "local_code"], dropna=False)[payload_columns].nunique(dropna=False)
    conflict_rows = int(conflicts.gt(1).any(axis=1).sum())
    result = result.sort_values(["effective_date", "local_code"]).drop_duplicates(
        ["effective_date", "local_code"], keep="last"
    )
    blockers = [f"conflicting_duplicate_snapshot_keys={conflict_rows}"] if conflict_rows else []
    return result.reset_index(drop=True), blockers


def _local_code(code: str) -> str:
    value = str(code).strip().upper()
    return value[:4] if len(value) == 5 and value.endswith("0") else value


def _build_code_ric_mapping(reference: pd.DataFrame, enrichment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"local_code", "refinitiv_ric"}
    if reference.empty or not required.issubset(reference.columns):
        raise ValueError("reference map lacks local_code/refinitiv_ric")
    base = reference.copy()
    base["local_code"] = base["local_code"].astype(str).str.strip().str.upper()
    base["refinitiv_ric"] = base["refinitiv_ric"].astype(str).str.strip()
    base = base.loc[base["local_code"].ne("") & base["refinitiv_ric"].ne("")]
    counts = base.groupby("local_code")["refinitiv_ric"].nunique()
    ambiguous_codes = set(counts[counts.gt(1)].index)
    unique = base.loc[~base["local_code"].isin(ambiguous_codes)].drop_duplicates("local_code")
    enrich_columns = [column for column in ("ric", "price_currency", "isin", "sedol", "perm_id") if column in enrichment]
    current = enrichment[enrich_columns].drop_duplicates("ric") if "ric" in enrich_columns else pd.DataFrame()
    if not current.empty:
        unique = unique.merge(current, left_on="refinitiv_ric", right_on="ric", how="left", suffixes=("", "_current"))
    if "price_currency" not in unique:
        unique["price_currency"] = pd.NA
    unique["mapping_status"] = "matched"
    mapping = unique[[column for column in (
        "local_code", "refinitiv_ric", "stable_instrument_id", "isin", "sedol", "perm_id", "price_currency", "mapping_status"
    ) if column in unique.columns]].copy()
    for column in ("stable_instrument_id", "isin", "sedol", "perm_id"):
        if column not in mapping:
            mapping[column] = pd.NA
    issues = base.loc[base["local_code"].isin(ambiguous_codes), ["local_code", "refinitiv_ric"]].copy()
    if not issues.empty:
        issues["mapping_status"] = "ambiguous_local_code_to_ric"
    return mapping, issues


def _reconstruct_windows(snapshots: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(snapshots["effective_date"].unique())
    date_index = {value: index for index, value in enumerate(dates)}
    rows: list[dict[str, Any]] = []
    for local_code, group in snapshots.groupby("local_code", sort=True):
        positions = sorted({date_index[value] for value in group["effective_date"]})
        start = previous = positions[0]
        runs: list[tuple[int, int]] = []
        for position in positions[1:]:
            if position != previous + 1:
                runs.append((start, previous))
                start = position
            previous = position
        runs.append((start, previous))
        for ordinal, (left, right) in enumerate(runs, start=1):
            start_date = dates[left]
            end_date = dates[right]
            rows.append(
                {
                    "stable_listing_id": f"JPX:{local_code}:{start_date.isoformat()}:{ordinal}",
                    "local_code": local_code,
                    "listing_start": start_date,
                    "listing_end": end_date,
                    "known_at": conservative_snapshot_known_at(start_date),
                    "known_at_method": "official_jpx_daily_delivery_schedule_conservative",
                    "left_truncated_at_source_start": left == 0,
                    "right_open_at_source_end": right == len(dates) - 1,
                }
            )
    return pd.DataFrame(rows)


def _entities(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["stable_listing_id", "local_code", "refinitiv_ric", "stable_instrument_id", "isin", "sedol", "perm_id"]
    return frame.reindex(columns=columns).drop_duplicates("stable_listing_id").reset_index(drop=True)


def _listing_windows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["stable_listing_id", "refinitiv_ric", "local_code", "listing_start", "listing_end", "known_at", "known_at_method", "currency", "pit_usable"]
    return frame.reindex(columns=columns).reset_index(drop=True)


def _identifier_history(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        for kind, value in (("local_code", row.get("local_code")), ("ric", row.get("refinitiv_ric")), ("isin", row.get("isin")), ("sedol", row.get("sedol"))):
            if value is None or pd.isna(value) or not str(value).strip():
                continue
            rows.append({
                "stable_listing_id": row["stable_listing_id"],
                "identifier_type": kind,
                "identifier_value": str(value),
                "valid_from": row["listing_start"],
                "valid_to": row["listing_end"],
                "known_at": row["known_at"],
            })
    return pd.DataFrame(rows, columns=["stable_listing_id", "identifier_type", "identifier_value", "valid_from", "valid_to", "known_at"])


def _membership(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["stable_listing_id", "refinitiv_ric", "membership_start", "membership_end", "known_at", "known_at_method", "pit_usable"]
    result = frame.rename(columns={"listing_start": "membership_start", "listing_end": "membership_end"})
    return result.reindex(columns=columns).reset_index(drop=True)


def _symbol_map(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["refinitiv_ric", "local_code", "listing_start", "listing_end", "currency", "known_at", "stable_listing_id", "pit_usable", "blocker_reason"]
    return frame.reindex(columns=columns).reset_index(drop=True)


def _blocked_manifest(
    run_id: str,
    artifact_dir: Path,
    blockers: list[str],
    input_sha256: dict[str, str],
    input_manifests: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": MATERIALIZER_VERSION,
        "run_id": run_id,
        "artifact_type": "japan_g1_pit_materializer_producer",
        "status": "blocked_upstream",
        "producer_run_id": run_id,
        "independent_audit": False,
        "scorecard_eligible": False,
        "evidence_scope": "blocked_no_candidate_output",
        "historical_pit_ready": False,
        "critical_blockers": blockers,
        "input_manifest_sha256": input_sha256,
        "input_manifests": input_manifests,
        "artifact_dir": str(artifact_dir.resolve()),
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def _output_paths(root: Path) -> dict[str, Path]:
    return {
        "manifest": root / "manifest.json",
        "entities": root / "entities.parquet",
        "listing_windows": root / "listing_windows.parquet",
        "identifier_history": root / "identifier_history.parquet",
        "universe_membership": root / "universe_membership_pit.parquet",
        "active_candidate": root / "active_symbol_map_candidate.csv",
        "reference_candidate": root / "reference_symbol_map_candidate.csv",
        "reconciliation": root / "reconciliation.csv",
    }


def _artifacts(run_id: str, artifact_dir: Path, paths: dict[str, Path], manifest: dict[str, Any]) -> JapanG1PITMaterializerArtifacts:
    return JapanG1PITMaterializerArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=paths["manifest"],
        entities_path=paths["entities"],
        listing_windows_path=paths["listing_windows"],
        identifier_history_path=paths["identifier_history"],
        universe_membership_path=paths["universe_membership"],
        active_candidate_path=paths["active_candidate"],
        reference_candidate_path=paths["reference_candidate"],
        reconciliation_path=paths["reconciliation"],
        manifest=manifest,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
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


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
