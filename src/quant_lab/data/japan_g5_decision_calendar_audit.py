from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from quant_lab.agents.japan_data_quality_loop import (
    JapanDataQualityPaths,
    required_japan_hash_inputs,
)


_PASS_STATUSES = {"accepted_backtest", "complete", "pass"}
_INVALID_SCOPE_MARKERS = ("partial", "policy-only", "policy_only", "smoke", "stub", "synthetic")
_FEATURE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_VIOLATION_COLUMNS = [
    "violation_id",
    "decision_id",
    "entity",
    "feature",
    "violation_type",
    "decision_time_utc",
    "observed_time_utc",
    "source_row_id",
    "detail",
]


@dataclass(frozen=True)
class JapanG5Input:
    path: Path
    manifest_path: Path
    artifact_key: str


@dataclass(frozen=True)
class JapanG5AuditConfig:
    required_features: tuple[str, ...]
    purge_days: int
    embargo_days: int
    train_end: str
    test_start: str
    test_end: str
    holdout_start: str
    decision_schedule: str


@dataclass(frozen=True)
class JapanG5DecisionCalendarArtifacts:
    run_id: str
    artifact_dir: Path
    decision_panel_path: Path
    leakage_violations_path: Path
    survivorship_reconciliation_path: Path
    split_ledger_path: Path
    feature_spec_path: Path
    audit_config_path: Path
    immutable_hashes_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable JSON manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return payload


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    try:
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"unreadable tabular input: {path}") from exc
    raise ValueError(f"unsupported tabular input extension for {path}; use CSV or Parquet")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _first_column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"{label} is missing; expected one of {list(candidates)}")


def _strict_utc(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    parsed: list[pd.Timestamp | pd.NaT] = []
    valid: list[bool] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
            is_valid = not pd.isna(timestamp) and timestamp.tzinfo is not None
            parsed.append(timestamp.tz_convert("UTC") if is_valid else pd.NaT)
            valid.append(bool(is_valid))
        except (TypeError, ValueError, OverflowError):
            parsed.append(pd.NaT)
            valid.append(False)
    return pd.Series(parsed, index=values.index, dtype="datetime64[ns, UTC]"), pd.Series(valid, index=values.index)


def _utc_scalar(value: str, label: str) -> pd.Timestamp:
    parsed, valid = _strict_utc(pd.Series([value]))
    if not bool(valid.iloc[0]):
        raise ValueError(f"{label} must be an explicit timezone-aware timestamp")
    return parsed.iloc[0]


def _listing_dates(values: pd.Series) -> pd.Series:
    def convert(value: Any) -> pd.Timestamp | pd.NaT:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return pd.NaT
        if pd.isna(timestamp):
            return pd.NaT
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Tokyo").tz_localize(None)
        return timestamp.normalize()

    return values.map(convert)


def _bool_series(values: pd.Series, default: bool) -> pd.Series:
    truthy = {"1", "true", "yes", "y"}
    falsy = {"0", "false", "no", "n"}

    def convert(value: Any) -> bool:
        if pd.isna(value):
            return default
        if isinstance(value, bool):
            return value
        label = str(value).strip().lower()
        if label in truthy:
            return True
        if label in falsy:
            return False
        return default

    return values.map(convert).astype(bool)


def _manifest_is_full(manifest: Mapping[str, Any]) -> bool:
    if not (
        str(manifest.get("status", "")).lower() in _PASS_STATUSES
        and manifest.get("scorecard_eligible") is True
        and manifest.get("evidence_scope") == "full"
    ):
        return False
    if any(
        manifest.get(flag) is True
        for flag in (
            "current_survivor",
            "is_policy_stub",
            "is_smoke_test",
            "partial",
            "policy_stub",
            "smoke",
            "synthetic",
        )
    ):
        return False
    labels = " ".join(
        str(manifest.get(key, ""))
        for key in ("artifact_type", "mode", "run_id", "scope", "scope_profile")
    ).lower()
    return not any(marker in labels for marker in _INVALID_SCOPE_MARKERS)


def _resolve_declared_path(raw_path: Any, manifest_path: Path) -> Path:
    path = Path(str(raw_path))
    return (path if path.is_absolute() else manifest_path.parent / path).resolve()


def _source_provenance(source: JapanG5Input) -> dict[str, Any]:
    path = source.path.resolve()
    manifest_path = source.manifest_path.resolve()
    manifest = _read_json(manifest_path)
    digest = _sha256(path)
    artifacts = manifest.get("artifacts", {})
    hashes = manifest.get("artifact_sha256", {})
    reasons: list[str] = []
    declared_path: Path | None = None
    expected_digest = ""

    if isinstance(artifacts, dict) and source.artifact_key in artifacts:
        declared_path = _resolve_declared_path(artifacts[source.artifact_key], manifest_path)
    else:
        reasons.append("artifact_path_not_declared")
    if isinstance(hashes, dict) and source.artifact_key in hashes:
        expected_digest = str(hashes[source.artifact_key]).lower()
    else:
        reasons.append("artifact_hash_not_declared")
    if not _manifest_is_full(manifest):
        reasons.append("manifest_not_full_scope_scorecard_evidence")
    if declared_path is not None and declared_path != path:
        reasons.append("declared_artifact_path_mismatch")
    if expected_digest != digest:
        reasons.append("declared_artifact_hash_mismatch")

    return {
        "artifact_key": source.artifact_key,
        "path": str(path),
        "manifest_path": str(manifest_path),
        "sha256": digest,
        "manifest_run_id": str(manifest.get("run_id", "")),
        "manifest_status": str(manifest.get("status", "")),
        "manifest_scorecard_eligible": manifest.get("scorecard_eligible") is True,
        "manifest_evidence_scope": str(manifest.get("evidence_scope", "")),
        "verified": not reasons,
        "reasons": reasons,
    }


def _upstream_admission(paths: JapanDataQualityPaths) -> dict[str, dict[str, Any]]:
    candidates = {
        "g1_security_master": paths.security_master_dir / "manifest.json",
        "g1_universe_reconciliation": (paths.universe_reconciliation_dir or Path("/__missing__")) / "manifest.json",
        "g1_active_symbol_map": (paths.active_symbol_map_dir or Path("/__missing__")) / "manifest.json",
        "g1_reference_partition": (paths.reference_symbol_map_dir or Path("/__missing__")) / "manifest.json",
        "g1_independent_validation": (paths.g1_validation_dir or Path("/__missing__")) / "manifest.json",
        "g2_strict_admission": (paths.g2_admission_dir or Path("/__missing__")) / "manifest.json",
        "g3_strict_admission": (paths.g3_admission_dir or Path("/__missing__")) / "manifest.json",
        "g4_fx_calendar_market_rules_tca": (paths.fx_calendar_tca_dir or Path("/__missing__")) / "manifest.json",
    }
    if paths.full_reference_symbol_map_dir is not None:
        candidates["g1_full_reference_symbol_map"] = (
            paths.full_reference_symbol_map_dir / "manifest.json"
        )
    admission: dict[str, dict[str, Any]] = {}
    for component, raw_path in candidates.items():
        path = raw_path.resolve()
        manifest: dict[str, Any] = {}
        if path.is_file():
            try:
                manifest = _read_json(path)
            except ValueError:
                manifest = {}
        full = _manifest_is_full(manifest)
        admission[component] = {
            "manifest_path": str(path),
            "manifest_sha256": _sha256(path) if path.is_file() else "",
            "hash_verified": path.is_file(),
            "status": str(manifest.get("status", "missing")),
            "scorecard_eligible": manifest.get("scorecard_eligible") is True,
            "evidence_scope": str(manifest.get("evidence_scope", "")),
            "admitted": bool(full and path.is_file()),
        }
    return admission


def _canonical_decisions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    required = {
        "decision_id",
        "entity",
        "decision_time_utc",
        "label_start_utc",
        "label_end_utc",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"decision calendar missing required columns: {sorted(missing)}")
    decisions = frame[
        ["decision_id", "entity", "decision_time_utc", "label_start_utc", "label_end_utc"]
    ].copy()
    decisions["source_decision_row_number"] = range(1, len(decisions) + 1)
    raw_ids = decisions["decision_id"]
    raw_entities = decisions["entity"]
    decisions["decision_id"] = raw_ids.astype("string").str.strip()
    decisions["entity"] = raw_entities.astype("string").str.strip()
    parsed, valid_time = _strict_utc(decisions["decision_time_utc"])
    decisions["decision_time_utc"] = parsed
    label_start, valid_label_start = _strict_utc(decisions["label_start_utc"])
    label_end, valid_label_end = _strict_utc(decisions["label_end_utc"])
    decisions["label_start_utc"] = label_start
    decisions["label_end_utc"] = label_end
    violations: list[dict[str, Any]] = []

    invalid_id = raw_ids.isna() | decisions["decision_id"].isna() | decisions["decision_id"].eq("")
    invalid_entity = raw_entities.isna() | decisions["entity"].isna() | decisions["entity"].eq("")
    duplicates = decisions["decision_id"].duplicated(keep=False) & ~invalid_id
    invalid_label_order = (
        valid_time
        & valid_label_start
        & valid_label_end
        & (
            decisions["label_start_utc"].lt(decisions["decision_time_utc"])
            | decisions["label_end_utc"].lt(decisions["label_start_utc"])
        )
    )
    invalid_rows = (
        invalid_id
        | invalid_entity
        | ~valid_time
        | ~valid_label_start
        | ~valid_label_end
        | invalid_label_order
        | duplicates
    )
    for index in decisions.index[invalid_rows]:
        if invalid_id.loc[index]:
            violation_type = "invalid_decision_id"
        elif invalid_entity.loc[index]:
            violation_type = "invalid_decision_entity"
        elif not valid_time.loc[index]:
            violation_type = "invalid_or_naive_decision_time"
        elif not valid_label_start.loc[index] or not valid_label_end.loc[index]:
            violation_type = "invalid_or_naive_label_interval"
        elif invalid_label_order.loc[index]:
            violation_type = "inconsistent_label_interval"
        else:
            violation_type = "duplicate_decision_id"
        violations.append(
            _violation(
                decision_id=decisions.at[index, "decision_id"],
                entity=decisions.at[index, "entity"],
                violation_type=violation_type,
                decision_time=decisions.at[index, "decision_time_utc"],
                detail=(
                    f"source_decision_row_number={decisions.at[index, 'source_decision_row_number']}; "
                    "required ordering is decision_time_utc <= label_start_utc <= label_end_utc"
                ),
            )
        )
    return decisions, violations


def _canonical_listings(frame: pd.DataFrame) -> pd.DataFrame:
    entity_col = _first_column(frame, ("entity", "refinitiv_ric", "ric", "requested_ric"), "listing entity")
    start_col = _first_column(frame, ("listing_start", "listing_start_utc"), "listing start")
    end_col = _first_column(frame, ("listing_end", "listing_end_utc"), "listing end")
    known_col = _first_column(frame, ("listing_known_at_utc", "known_at"), "listing known-at")
    listings = pd.DataFrame(
        {
            "entity": frame[entity_col].astype("string").str.strip(),
            "listing_start": _listing_dates(frame[start_col]),
            "listing_end": _listing_dates(frame[end_col]),
        }
    )
    listings["listing_known_at_utc"], listings["listing_known_at_valid"] = _strict_utc(frame[known_col])
    end_known_col = next(
        (column for column in ("listing_end_known_at_utc", "window_end_known_at") if column in frame),
        None,
    )
    if end_known_col is not None:
        end_known, _ = _strict_utc(frame[end_known_col])
        listings["listing_end_known_at_utc"] = end_known
    else:
        listings["listing_end_known_at_utc"] = pd.NaT
    listings["right_open"] = (
        _bool_series(frame["right_open"], default=False)
        if "right_open" in frame
        else False
    )
    listings["pit_usable"] = (
        _bool_series(frame["pit_usable"], default=False)
        if "pit_usable" in frame
        else False
    )
    listing_id_col = next(
        (column for column in ("stable_listing_id", "listing_id") if column in frame),
        None,
    )
    listings["listing_id"] = (
        frame[listing_id_col].astype("string").str.strip()
        if listing_id_col is not None
        else pd.Series([pd.NA] * len(frame), dtype="string")
    )
    return listings


def _canonical_features(frame: pd.DataFrame) -> pd.DataFrame:
    entity_col = _first_column(frame, ("entity", "refinitiv_ric", "ric", "requested_ric"), "feature entity")
    known_col = _first_column(frame, ("source_known_at_utc", "available_time_utc"), "feature known-at")
    asof_col = _first_column(
        frame,
        ("source_asof_time_utc", "observation_date", "asof_date", "calc_date", "estimate_date", "period_end_date"),
        "feature source as-of",
    )
    value_col = _first_column(frame, ("value", "feature_value", "estimate_value"), "feature value")
    if "feature" in frame:
        feature = frame["feature"].astype("string").str.strip()
    elif {"dataset_id", "field_role"}.issubset(frame.columns):
        feature = (
            frame["dataset_id"].astype("string").str.strip()
            + "__"
            + frame["field_role"].astype("string").str.strip()
        )
    else:
        raise ValueError("feature observations require feature or dataset_id plus field_role")
    source_asof = pd.to_datetime(frame[asof_col], errors="coerce", utc=True)
    source_known, source_known_valid = _strict_utc(frame[known_col])
    observations = pd.DataFrame(
        {
            "entity": frame[entity_col].astype("string").str.strip(),
            "feature": feature,
            "feature_value": frame[value_col],
            "source_asof_time_utc": source_asof,
            "source_known_at_utc": source_known,
            "source_known_at_valid": source_known_valid,
            "source_row_id": frame["source_row_id"].astype("string").str.strip()
            if "source_row_id" in frame
            else pd.Series([f"source-row-{index + 1}" for index in range(len(frame))], dtype="string"),
        }
    )
    if "dataset_id" in frame:
        observations["dataset_id"] = frame["dataset_id"].astype("string").str.strip()
    else:
        observations["dataset_id"] = observations["feature"].str.split("__", n=1).str[0]
    return observations


def _violation(
    *,
    violation_type: str,
    decision_id: Any = "",
    entity: Any = "",
    feature: Any = "",
    decision_time: Any = "",
    observed_time: Any = "",
    source_row_id: Any = "",
    detail: str = "",
) -> dict[str, Any]:
    return {
        "violation_id": "",
        "decision_id": "" if pd.isna(decision_id) else str(decision_id),
        "entity": "" if pd.isna(entity) else str(entity),
        "feature": "" if pd.isna(feature) else str(feature),
        "violation_type": violation_type,
        "decision_time_utc": "" if pd.isna(decision_time) else str(decision_time),
        "observed_time_utc": "" if pd.isna(observed_time) else str(observed_time),
        "source_row_id": "" if pd.isna(source_row_id) else str(source_row_id),
        "detail": detail,
    }


def _reconcile_listings(
    decisions: pd.DataFrame,
    listings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    panel = decisions.copy()
    panel["listing_id"] = pd.Series([pd.NA] * len(panel), dtype="string")
    panel["listing_start"] = pd.NaT
    panel["listing_end"] = pd.NaT
    panel["listing_known_at_utc"] = pd.Series(
        pd.NaT,
        index=panel.index,
        dtype="datetime64[ns, UTC]",
    )
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    grouped = {str(entity): group for entity, group in listings.groupby("entity", dropna=False, sort=False)}

    for index, decision in panel.iterrows():
        decision_time = decision["decision_time_utc"]
        entity = str(decision["entity"])
        reason = ""
        matched: pd.DataFrame = pd.DataFrame()
        if pd.isna(decision_time):
            reason = "invalid_decision_time"
        else:
            local_date = decision_time.tz_convert("Asia/Tokyo").tz_localize(None).normalize()
            candidates = grouped.get(entity, pd.DataFrame()).copy()
            if not candidates.empty:
                started = candidates["listing_start"].notna() & candidates["listing_start"].le(local_date)
                ended = candidates["listing_end"].notna()
                inside_closed = ~ended | candidates["listing_end"].ge(local_date)
                inside_open = ~ended | candidates["listing_end"].gt(local_date)
                inside = inside_closed.where(~candidates["right_open"], inside_open)
                matched = candidates.loc[started & inside]
            if matched.empty:
                reason = "outside_listing_window"
            elif len(matched) != 1:
                reason = "ambiguous_overlapping_listing_windows"
            else:
                window = matched.iloc[0]
                panel.at[index, "listing_id"] = window["listing_id"]
                panel.at[index, "listing_start"] = window["listing_start"]
                panel.at[index, "listing_end"] = window["listing_end"]
                panel.at[index, "listing_known_at_utc"] = window["listing_known_at_utc"]
                if not bool(window["listing_known_at_valid"]):
                    reason = "invalid_listing_known_at"
                elif window["listing_known_at_utc"] > decision_time:
                    reason = "listing_window_not_known_at_decision"
                elif not bool(window["pit_usable"]):
                    reason = "listing_window_not_pit_usable"
        status = "pass" if not reason else "fail"
        rows.append(
            {
                "decision_id": decision["decision_id"],
                "entity": decision["entity"],
                "decision_time_utc": decision_time,
                "listing_id": panel.at[index, "listing_id"],
                "listing_start": panel.at[index, "listing_start"],
                "listing_end": panel.at[index, "listing_end"],
                "listing_known_at_utc": panel.at[index, "listing_known_at_utc"],
                "status": status,
                "reason": reason,
            }
        )
        if reason:
            violations.append(
                _violation(
                    decision_id=decision["decision_id"],
                    entity=decision["entity"],
                    violation_type=reason,
                    decision_time=decision_time,
                    observed_time=panel.at[index, "listing_known_at_utc"],
                    detail="decision failed PIT listing-window reconciliation",
                )
            )
    decision_reconciliation = pd.DataFrame(rows)
    panel["membership_valid_at_decision"] = decision_reconciliation["status"].eq("pass").to_numpy()
    reconciliation_rows: list[dict[str, Any]] = []
    for entity, group in decision_reconciliation.groupby("entity", dropna=False, sort=False):
        failed = group["status"].ne("pass")
        reconciliation_rows.append(
            {
                "entity": entity,
                "status": "pass" if not failed.any() else "fail",
                "decision_rows": int(len(group)),
                "passing_decision_rows": int((~failed).sum()),
                "failing_decision_rows": int(failed.sum()),
                "listing_windows_used": int(group["listing_id"].nunique(dropna=True)),
                "failure_reasons": "|".join(
                    sorted(set(group.loc[failed, "reason"].dropna().astype(str)))
                ),
            }
        )
    reconciliation = pd.DataFrame(
        reconciliation_rows,
        columns=[
            "entity",
            "status",
            "decision_rows",
            "passing_decision_rows",
            "failing_decision_rows",
            "listing_windows_used",
            "failure_reasons",
        ],
    )
    return panel, reconciliation, violations


def _attach_features(
    panel: pd.DataFrame,
    observations: pd.DataFrame,
    features: Sequence[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]], int, int]:
    output = panel.copy()
    violations: list[dict[str, Any]] = []
    selected_known = 0
    selected_values = 0
    required = observations[observations["feature"].isin(features)].copy()
    invalid = (
        required["entity"].isna()
        | required["entity"].eq("")
        | required["feature"].isna()
        | required["feature"].eq("")
        | ~required["source_known_at_valid"]
        | required["source_asof_time_utc"].isna()
    )
    future_asof = (
        ~invalid
        & required["source_asof_time_utc"].gt(required["source_known_at_utc"])
    )
    duplicates = required.duplicated(["entity", "feature", "source_known_at_utc"], keep=False) & ~invalid
    for index in required.index[invalid | future_asof | duplicates]:
        row = required.loc[index]
        if invalid.loc[index]:
            violation_type = "invalid_feature_source_timestamp_or_key"
        elif future_asof.loc[index]:
            violation_type = "source_asof_after_source_known_at"
        else:
            violation_type = "duplicate_feature_known_at_key"
        violations.append(
            _violation(
                entity=row["entity"],
                feature=row["feature"],
                violation_type=violation_type,
                observed_time=row["source_known_at_utc"],
                source_row_id=row["source_row_id"],
                detail="feature observation is not admissible for deterministic PIT selection",
            )
        )
    usable = required.loc[~invalid & ~future_asof & ~duplicates].copy()

    for feature in features:
        feature_rows = usable[usable["feature"].eq(feature)]
        parts: list[pd.DataFrame] = []
        for entity, left in output.groupby("entity", dropna=False, sort=False):
            left_part = left[["source_decision_row_number", "decision_time_utc"]].copy()
            left_part = left_part.sort_values("decision_time_utc")
            right = feature_rows[feature_rows["entity"].eq(entity)][
                ["feature_value", "source_asof_time_utc", "source_known_at_utc", "source_row_id", "dataset_id"]
            ].sort_values("source_known_at_utc")
            if right.empty:
                merged = left_part.copy()
                for column in ("feature_value", "source_asof_time_utc", "source_known_at_utc", "source_row_id", "dataset_id"):
                    merged[column] = pd.NA
            else:
                merged = pd.merge_asof(
                    left_part,
                    right,
                    left_on="decision_time_utc",
                    right_on="source_known_at_utc",
                    direction="backward",
                    allow_exact_matches=True,
                )
            parts.append(merged)
        selected = pd.concat(parts, ignore_index=True).set_index("source_decision_row_number")
        selected = selected.reindex(output["source_decision_row_number"])
        value_col = feature
        asof_col = f"{feature}__source_asof_time_utc"
        known_col = f"{feature}__source_known_at_utc"
        row_col = f"{feature}__source_row_id"
        output[value_col] = selected["feature_value"].to_numpy()
        output[asof_col] = selected["source_asof_time_utc"].to_numpy()
        output[known_col] = selected["source_known_at_utc"].to_numpy()
        output[row_col] = selected["source_row_id"].to_numpy()
        selected_known += int(selected["source_known_at_utc"].notna().sum())
        selected_values += int(selected["feature_value"].notna().sum())
        missing = selected["source_known_at_utc"].isna() | selected["feature_value"].isna()
        for decision_row_number in selected.index[missing]:
            decision = output.loc[output["source_decision_row_number"].eq(decision_row_number)].iloc[0]
            violations.append(
                _violation(
                    decision_id=decision["decision_id"],
                    entity=decision["entity"],
                    feature=feature,
                    violation_type="missing_feature_at_decision",
                    decision_time=decision["decision_time_utc"],
                    detail="no non-null observation was known by the decision time",
                )
            )
    known_columns = [f"{feature}__source_known_at_utc" for feature in features]
    known_times = pd.concat(
        [pd.to_datetime(output[column], errors="coerce", utc=True) for column in known_columns],
        axis=1,
    )
    output["max_feature_available_time_utc"] = known_times.max(axis=1)
    selected_after_decision = output["max_feature_available_time_utc"].gt(output["decision_time_utc"])
    for index in output.index[selected_after_decision]:
        violations.append(
            _violation(
                decision_id=output.at[index, "decision_id"],
                entity=output.at[index, "entity"],
                violation_type="max_feature_available_time_after_decision",
                decision_time=output.at[index, "decision_time_utc"],
                observed_time=output.at[index, "max_feature_available_time_utc"],
                detail="selected feature availability exceeds the exact decision timestamp",
            )
        )
    return output, violations, selected_known, selected_values


def _split_ledger(
    decisions: pd.DataFrame,
    config: JapanG5AuditConfig,
) -> tuple[pd.DataFrame, list[str], int]:
    train_end = _utc_scalar(config.train_end, "train_end")
    test_start = _utc_scalar(config.test_start, "test_start")
    test_end = _utc_scalar(config.test_end, "test_end")
    holdout_start = _utc_scalar(config.holdout_start, "holdout_start")
    blockers: list[str] = []
    if config.purge_days < 0 or config.embargo_days < 0:
        blockers.append("purge_days_and_embargo_days_must_be_nonnegative")
    if not train_end < test_start <= test_end < holdout_start:
        blockers.append("split_boundaries_are_not_strictly_ordered")
    if test_start - train_end < pd.Timedelta(days=max(0, config.purge_days)):
        blockers.append("configured_train_test_gap_is_shorter_than_purge_days")
    if holdout_start - test_end < pd.Timedelta(days=max(0, config.embargo_days)):
        blockers.append("configured_test_holdout_gap_is_shorter_than_embargo_days")

    times = decisions["decision_time_utc"]
    labels = pd.Series("invalid", index=decisions.index, dtype="string")
    valid = times.notna()
    labels.loc[valid & times.le(train_end)] = "train"
    labels.loc[valid & times.gt(train_end) & times.lt(test_start)] = "purged"
    labels.loc[valid & times.ge(test_start) & times.le(test_end)] = "test"
    labels.loc[valid & times.gt(test_end) & times.lt(holdout_start)] = "embargo"
    labels.loc[valid & times.ge(holdout_start)] = "holdout"
    train_label_overlap = labels.eq("train") & decisions["label_end_utc"].ge(test_start)
    test_label_overlap = labels.eq("test") & decisions["label_end_utc"].ge(holdout_start)
    labels.loc[train_label_overlap] = "purged"
    labels.loc[test_label_overlap] = "embargo"
    ledger = pd.DataFrame(
        {
            "decision_id": decisions["decision_id"],
            "entity": decisions["entity"],
            "decision_time_utc": times,
            "split": labels,
            "purge_days": config.purge_days,
            "embargo_days": config.embargo_days,
            "label_start_utc": decisions["label_start_utc"],
            "label_end_utc": decisions["label_end_utc"],
        }
    )
    required_labels = {"train", "test", "holdout"}
    missing_labels = required_labels - set(labels)
    if missing_labels:
        blockers.append(f"decision_calendar_missing_required_splits:{','.join(sorted(missing_labels))}")
    if labels.eq("invalid").any():
        blockers.append("invalid_decision_times_are_not_assignable_to_a_split")
    remaining_overlap = int(
        (
            labels.eq("train") & decisions["label_end_utc"].ge(test_start)
            | labels.eq("test") & decisions["label_end_utc"].ge(holdout_start)
        ).sum()
    )
    return ledger, blockers, remaining_overlap


def _deduplicated_paths(paths: Iterable[Path]) -> list[Path]:
    unique: dict[Path, None] = {}
    for path in paths:
        unique[path.resolve()] = None
    return list(unique)


def _projected_command_log(path: Path | None, command: str | None) -> tuple[bytes | None, bytes]:
    append_bytes = (command.rstrip("\n") + "\n").encode("utf-8") if command else b""
    if path is None:
        return None, append_bytes
    current = path.read_bytes() if path.is_file() else b""
    return current + append_bytes, append_bytes


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_published_command(path: Path, payload: bytes) -> None:
    if not payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_staged_publication(
    *,
    stage_by_final: Mapping[Path, Path],
    artifacts: Mapping[str, str],
    artifact_sha256: Mapping[str, str],
    inventory: Mapping[str, Any],
    projected_command_path: Path | None,
    projected_command_bytes: bytes | None,
) -> None:
    for name, raw_final in artifacts.items():
        final_path = Path(raw_final).resolve()
        staged_path = stage_by_final.get(final_path)
        if staged_path is None or not staged_path.is_file():
            raise RuntimeError(f"staged artifact is missing: {name}")
        if _sha256(staged_path) != artifact_sha256[name]:
            raise RuntimeError(f"staged artifact hash mismatch: {name}")
    files = inventory.get("files", {})
    if not isinstance(files, dict):
        raise RuntimeError("staged immutable inventory has no file map")
    for raw_path, expected in files.items():
        final_path = Path(raw_path).resolve()
        staged_path = stage_by_final.get(final_path)
        if staged_path is not None:
            actual = _sha256(staged_path)
        elif projected_command_path is not None and final_path == projected_command_path:
            if projected_command_bytes is None:
                raise RuntimeError("projected command log bytes are unavailable")
            actual = _digest_bytes(projected_command_bytes)
        else:
            if not final_path.is_file():
                raise RuntimeError(f"inventoried input disappeared before publish: {final_path}")
            actual = _sha256(final_path)
        if actual != str(expected).lower():
            raise RuntimeError(f"staged inventory hash mismatch: {final_path}")


def write_japan_g5_decision_calendar_audit(
    *,
    paths: JapanDataQualityPaths,
    run_id: str,
    decision_calendar: JapanG5Input,
    listing_windows: JapanG5Input,
    feature_observations: JapanG5Input,
    config: JapanG5AuditConfig,
    command: str | None = None,
) -> JapanG5DecisionCalendarArtifacts:
    """Produce the row-level Japan G5 decision-calendar audit.

    All three inputs must be explicit, hash-pinned artifacts from full-scope,
    scorecard-eligible upstream manifests. Audit failures are materialized as
    ineligible evidence; unreadable or structurally incompatible inputs raise.
    """

    if paths.leakage_qa_dir is None:
        raise ValueError("paths.leakage_qa_dir is required")
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    features = tuple(config.required_features)
    if not features or len(set(features)) != len(features):
        raise ValueError("required_features must be non-empty and unique")
    invalid_features = [feature for feature in features if not _FEATURE_RE.fullmatch(feature)]
    if invalid_features:
        raise ValueError(f"feature names are not safe column identifiers: {invalid_features}")
    if not config.decision_schedule.strip():
        raise ValueError("decision_schedule must be non-empty")

    sources = {
        "decision_calendar": decision_calendar,
        "listing_windows": listing_windows,
        "feature_observations": feature_observations,
    }
    for source in sources.values():
        if not source.path.is_file():
            raise FileNotFoundError(source.path)
        if not source.manifest_path.is_file():
            raise FileNotFoundError(source.manifest_path)

    artifact_dir = paths.leakage_qa_dir.resolve()
    if artifact_dir.exists():
        raise FileExistsError(f"write-once Japan G5 artifact directory already exists: {artifact_dir}")
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    decision_panel_path = artifact_dir / "decision_panel.parquet"
    violations_path = artifact_dir / "leakage_violations.csv"
    survivorship_path = artifact_dir / "survivorship_reconciliation.csv"
    split_path = artifact_dir / "split_ledger.csv"
    feature_spec_path = artifact_dir / "feature_spec.json"
    audit_config_path = artifact_dir / "audit_config.json"
    manifest_path = artifact_dir / "manifest.json"
    immutable_path = (paths.immutable_hashes_path or artifact_dir / "immutable_hashes.json").resolve()
    if immutable_path != artifact_dir / "immutable_hashes.json":
        raise ValueError("immutable_hashes_path must be inside the write-once G5 artifact directory")
    if command and paths.commands_log is None:
        raise ValueError("paths.commands_log is required when command logging is requested")
    command_path = paths.commands_log.resolve() if paths.commands_log is not None else None
    if command_path is not None and command_path.is_relative_to(artifact_dir):
        raise ValueError("commands_log must be external to the atomically published G5 directory")
    projected_command_bytes, command_append_bytes = _projected_command_log(command_path, command)

    provenance = {name: _source_provenance(source) for name, source in sources.items()}
    decisions, decision_violations = _canonical_decisions(_read_table(decision_calendar.path))
    listings = _canonical_listings(_read_table(listing_windows.path))
    observations = _canonical_features(_read_table(feature_observations.path))
    panel, survivorship, listing_violations = _reconcile_listings(decisions, listings)
    panel, feature_violations, selected_known, selected_values = _attach_features(panel, observations, features)
    split_ledger, split_blockers, train_test_overlap_rows = _split_ledger(decisions, config)

    violations = decision_violations + listing_violations + feature_violations
    for number, row in enumerate(violations, start=1):
        row["violation_id"] = f"g5-v-{number:09d}"
    violation_frame = pd.DataFrame(violations, columns=_VIOLATION_COLUMNS)
    denominator = len(decisions) * len(features)
    known_at_coverage = selected_known / denominator if denominator else 0.0
    feature_value_coverage = selected_values / denominator if denominator else 0.0
    survivorship_violations = int(survivorship["status"].ne("pass").sum()) if not survivorship.empty else 0
    future_types = {
        "max_feature_available_time_after_decision",
        "source_asof_after_source_known_at",
    }
    future_data_violations = int(violation_frame["violation_type"].isin(future_types).sum())

    blockers: list[str] = []
    blockers.extend(
        f"source_provenance_failed:{name}:{','.join(item['reasons'])}"
        for name, item in provenance.items()
        if not item["verified"]
    )
    blockers.extend(split_blockers)
    upstream_admission = _upstream_admission(paths)
    blockers.extend(
        f"upstream_admission_not_full_or_hash_verified:{component}"
        for component, item in upstream_admission.items()
        if not item["admitted"] or not item["hash_verified"]
    )
    if decisions.empty:
        blockers.append("decision_calendar_is_empty")
    if decisions["entity"].nunique(dropna=True) == 0:
        blockers.append("decision_calendar_has_no_entities")
    if not violation_frame.empty:
        blockers.append(f"row_level_audit_violations:{len(violation_frame)}")
    if survivorship.empty or survivorship_violations:
        blockers.append(f"survivorship_reconciliation_failed:{survivorship_violations}")
    if known_at_coverage != 1.0:
        blockers.append(f"known_at_coverage_is_not_complete:{known_at_coverage:.12f}")
    if feature_value_coverage != 1.0:
        blockers.append(f"feature_value_coverage_is_not_complete:{feature_value_coverage:.12f}")

    feature_spec = {
        "features": list(features),
        "feature_count": len(features),
        "selection": "latest_source_known_at_not_after_exact_decision_time",
        "required_value_coverage": 1.0,
        "required_known_at_coverage": 1.0,
        "source_artifact": str(feature_observations.path.resolve()),
        "source_sha256": _sha256(feature_observations.path),
    }
    frozen_config = {
        **asdict(config),
        "required_features": list(features),
        "timezone": "UTC",
        "listing_calendar_timezone": "Asia/Tokyo",
        "split_interval_policy": "train_then_purge_gap_then_test_then_embargo_gap_then_holdout",
        "input_provenance": provenance,
        "upstream_g1_g4_admission": upstream_admission,
    }
    generated_without_manifest = {
        decision_panel_path.resolve(),
        violations_path.resolve(),
        survivorship_path.resolve(),
        split_path.resolve(),
        feature_spec_path.resolve(),
        audit_config_path.resolve(),
    }
    expected_generated = generated_without_manifest | {manifest_path.resolve()}
    required_gate_paths = _deduplicated_paths(required_japan_hash_inputs(paths))
    missing_external_hash_inputs = [
        path
        for path in required_gate_paths
        if path.resolve() not in expected_generated
        and not (
            command_path is not None
            and path.resolve() == command_path
            and projected_command_bytes is not None
        )
        and not path.is_file()
    ]
    if missing_external_hash_inputs:
        blockers.append(f"immutable_hash_inputs_missing:{len(missing_external_hash_inputs)}")
    if projected_command_bytes is None or len(projected_command_bytes) == 0:
        blockers.append("non_empty_commands_log_is_missing")

    blockers = list(dict.fromkeys(blockers))
    eligible = not blockers
    artifacts = {
        "decision_panel": str(decision_panel_path),
        "leakage_violations": str(violations_path),
        "survivorship_reconciliation": str(survivorship_path),
        "split_ledger": str(split_path),
        "feature_spec": str(feature_spec_path),
        "audit_config": str(audit_config_path),
    }
    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{artifact_dir.name}.tmp-", dir=artifact_dir.parent)
    ).resolve()
    stage_paths = {
        "decision_panel": stage_dir / decision_panel_path.name,
        "leakage_violations": stage_dir / violations_path.name,
        "survivorship_reconciliation": stage_dir / survivorship_path.name,
        "split_ledger": stage_dir / split_path.name,
        "feature_spec": stage_dir / feature_spec_path.name,
        "audit_config": stage_dir / audit_config_path.name,
    }
    stage_manifest = stage_dir / manifest_path.name
    stage_immutable = stage_dir / immutable_path.name
    stage_by_final: dict[Path, Path] = {
        Path(artifacts[name]).resolve(): stage_path
        for name, stage_path in stage_paths.items()
    }
    stage_by_final[manifest_path.resolve()] = stage_manifest
    published = False
    manifest: dict[str, Any]
    inventory: dict[str, Any]
    try:
        _atomic_parquet(stage_paths["decision_panel"], panel)
        _atomic_csv(stage_paths["leakage_violations"], violation_frame)
        _atomic_csv(stage_paths["survivorship_reconciliation"], survivorship)
        _atomic_csv(stage_paths["split_ledger"], split_ledger)
        _atomic_json(stage_paths["feature_spec"], feature_spec)
        _atomic_json(stage_paths["audit_config"], frozen_config)
        artifact_sha256 = {
            name: _sha256(stage_paths[name])
            for name in artifacts
        }
        manifest = {
            "run_id": run_id,
            "artifact_type": "japan_g5_true_decision_calendar_audit",
            "scope": "japan_g5_true_decision_calendar_audit",
            "status": "pass" if eligible else "fail",
            "evidence_scope": "full" if eligible else "incomplete",
            "scorecard_eligible": eligible,
            "partial": not eligible,
            "synthetic": False,
            "research_only": True,
            "production_ready": False,
            "live_execution_ready": False,
            "ibkr_ready": False,
            "decision_rows": int(len(decisions)),
            "decision_entities": int(decisions["entity"].nunique(dropna=True)),
            "decision_start_utc": str(decisions["decision_time_utc"].min()),
            "decision_end_utc": str(decisions["decision_time_utc"].max()),
            "features": list(features),
            "known_at_coverage": float(known_at_coverage),
            "feature_value_coverage": float(feature_value_coverage),
            "total_leakage_violations": int(len(violation_frame)),
            "future_data_violations": future_data_violations,
            "train_test_overlap_rows": train_test_overlap_rows,
            "survivorship_violations": survivorship_violations,
            "purge_days": config.purge_days,
            "embargo_days": config.embargo_days,
            "holdout_start": config.holdout_start,
            "decision_schedule": config.decision_schedule,
            "source_provenance": provenance,
            "upstream_g1_g4_admission": upstream_admission,
            "critical_blockers": blockers,
            "artifacts": artifacts,
            "artifact_sha256": artifact_sha256,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(stage_manifest, manifest)

        inventory_paths = _deduplicated_paths(
            [
                *required_gate_paths,
                *generated_without_manifest,
                manifest_path,
                *(source.path for source in sources.values()),
                *(source.manifest_path for source in sources.values()),
            ]
        )
        files: dict[str, str] = {}
        missing_inventory: list[str] = []
        for final_path in inventory_paths:
            staged_path = stage_by_final.get(final_path.resolve())
            if staged_path is not None and staged_path.is_file():
                files[str(final_path.resolve())] = _sha256(staged_path)
            elif command_path is not None and final_path.resolve() == command_path:
                if projected_command_bytes is None:
                    missing_inventory.append(str(final_path.resolve()))
                else:
                    files[str(final_path.resolve())] = _digest_bytes(projected_command_bytes)
            elif final_path.is_file():
                files[str(final_path.resolve())] = _sha256(final_path)
            else:
                missing_inventory.append(str(final_path.resolve()))
        coverage_complete = not missing_inventory and len(files) == len(inventory_paths)
        inventory_eligible = bool(eligible and coverage_complete)
        inventory = {
            "run_id": f"{run_id}-IMMUTABLE-INVENTORY",
            "artifact_type": "japan_g5_immutable_hash_inventory",
            "status": "pass" if inventory_eligible else "incomplete",
            "scorecard_eligible": inventory_eligible,
            "evidence_scope": "full" if inventory_eligible else "incomplete",
            "algorithm": "sha256",
            "immutable": True,
            "coverage_complete": coverage_complete,
            "required_files": len(inventory_paths),
            "hashed_files": len(files),
            "missing_files": missing_inventory,
            "files": files,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(stage_immutable, inventory)
        if _read_json(stage_manifest) != manifest:
            raise RuntimeError("staged manifest failed read-after-write verification")
        if _read_json(stage_immutable) != inventory:
            raise RuntimeError("staged immutable inventory failed read-after-write verification")
        expected_stage_files = set(stage_by_final.values()) | {stage_immutable}
        actual_stage_files = {path for path in stage_dir.rglob("*") if path.is_file()}
        if actual_stage_files != expected_stage_files:
            raise RuntimeError("staged G5 directory contains an unexpected or incomplete file set")
        _verify_staged_publication(
            stage_by_final=stage_by_final,
            artifacts=artifacts,
            artifact_sha256=artifact_sha256,
            inventory=inventory,
            projected_command_path=command_path,
            projected_command_bytes=projected_command_bytes,
        )
        _fsync_tree(stage_dir)
        if artifact_dir.exists():
            raise FileExistsError(
                f"write-once Japan G5 artifact directory appeared during publish: {artifact_dir}"
            )
        os.replace(stage_dir, artifact_dir)
        published = True
        _fsync_directory(artifact_dir.parent)
    finally:
        if not published and stage_dir.exists():
            shutil.rmtree(stage_dir)

    if command_path is not None and command_append_bytes:
        expected_prior = projected_command_bytes[: -len(command_append_bytes)]
        try:
            current = command_path.read_bytes() if command_path.is_file() else b""
            if current != expected_prior:
                raise RuntimeError("commands.log changed concurrently during G5 publication")
            _append_published_command(command_path, command_append_bytes)
            expected_digest = inventory["files"].get(str(command_path))
            if expected_digest != _sha256(command_path):
                raise RuntimeError("published commands.log does not match immutable inventory")
        except Exception as exc:
            failure_path = artifact_dir.parent / f".{artifact_dir.name}.command-log-failure.json"
            _atomic_json(
                failure_path,
                {
                    "run_id": run_id,
                    "artifact_dir": str(artifact_dir),
                    "commands_log": str(command_path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            raise RuntimeError(
                f"G5 directory published but command logging failed; see {failure_path}"
            ) from exc

    return JapanG5DecisionCalendarArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        decision_panel_path=decision_panel_path,
        leakage_violations_path=violations_path,
        survivorship_reconciliation_path=survivorship_path,
        split_ledger_path=split_path,
        feature_spec_path=feature_spec_path,
        audit_config_path=audit_config_path,
        immutable_hashes_path=immutable_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
