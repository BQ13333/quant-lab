from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


UTC = timezone.utc
TERMINAL_POSITIVE = {"completed", "completed_partial"}
TERMINAL_STRUCTURAL = {"completed_prelisting", "completed_postretirement"}


@dataclass(frozen=True)
class JapanOhlcvRecoveryMergeArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def merge_japan_ohlcv_recovery(
    *,
    run_id: str,
    plan_row_id: str,
    stage_artifact_dir: str | Path,
    stage_lake_root: str | Path,
    production_artifact_dir: str | Path,
    production_lake_root: str | Path,
    expected_stage_plan_sha256: str,
    expected_production_plan_sha256: str,
    artifact_root: str | Path = "artifacts/japan_ohlcv_recovery_merge",
) -> JapanOhlcvRecoveryMergeArtifacts:
    stage_artifact = Path(stage_artifact_dir).resolve()
    stage_lake = Path(stage_lake_root).resolve()
    production_artifact = Path(production_artifact_dir).resolve()
    production_lake = Path(production_lake_root).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    root.mkdir(parents=True, exist_ok=False)

    stage_plan = stage_artifact / "historical_plan.csv"
    production_plan = production_artifact / "historical_plan.csv"
    _require_hash(stage_plan, expected_stage_plan_sha256)
    _require_hash(production_plan, expected_production_plan_sha256)
    plan_columns = [
        "plan_row_id", "run_id", "country", "exchange", "batch_no", "window_no",
        "start_date", "end_date", "interval", "tickers", "local_symbols",
    ]
    stage_plan_row = _one(_csv(stage_plan), plan_row_id, "stage plan")
    production_plan_row = _one(_csv(production_plan), plan_row_id, "production plan")
    if any(stage_plan_row.get(column, "") != production_plan_row.get(column, "") for column in plan_columns):
        raise ValueError("stage and production plan-row identities differ")

    stage_manifest_path = stage_artifact / "historical_manifest.csv"
    production_manifest_path = production_artifact / "historical_manifest.csv"
    stage_manifest = _csv(stage_manifest_path)
    production_manifest = _csv(production_manifest_path)
    stage_row = _one(stage_manifest, plan_row_id, "stage manifest")
    production_row = _one(production_manifest, plan_row_id, "production manifest")
    expected_rows = int(stage_row.get("long_ohlcv_rows", "0") or 0)
    positive = stage_row.get("status") in TERMINAL_POSITIVE and expected_rows > 0
    structural = bool(
        stage_row.get("status") in TERMINAL_STRUCTURAL
        and expected_rows == 0
        and stage_row.get("structural_reason")
    )
    if not positive and not structural:
        raise ValueError("stage row is not a positive or source-backed structural recovery")
    if production_row.get("status") not in {"failed_empty", "failed_retryable_provider"}:
        raise ValueError("production row is not the quarantined failed row")

    stage_long = Path(stage_row["long_ohlcv_root"]).resolve()
    production_long = Path(production_plan_row["long_ohlcv_root"]).resolve()
    _require_within(stage_long, stage_lake)
    _require_within(production_long, production_lake)
    staged = _batch_table(stage_long, plan_row_id, allow_missing=structural)
    if len(staged) != expected_rows or staged.duplicated(["ticker", "date"]).any():
        raise ValueError("staged long payload count or key uniqueness failed")
    if positive:
        if not staged["ticker"].isin(stage_plan_row["tickers"].split(",")).all():
            raise ValueError("staged payload contains an unplanned ticker")
        dates = pd.to_datetime(staged["date"])
        if dates.min() < pd.Timestamp(stage_plan_row["start_date"]) or dates.max() > pd.Timestamp(stage_plan_row["end_date"]):
            raise ValueError("staged payload falls outside the plan window")
        canonical = _csv(Path(stage_row["canonical_coverage_path"]))
        if int(canonical["rows"].astype(int).sum()) != expected_rows:
            raise ValueError("canonical coverage does not reconcile to staged rows")
    existing_batch = _batch_table(production_long, plan_row_id, allow_missing=True)
    if not existing_batch.empty:
        raise ValueError("production lake already contains the recovery batch")
    overlap = _ticker_table(production_long, sorted(staged["ticker"].unique()))
    if not overlap.empty and staged.merge(overlap, on=["ticker", "date"], how="inner").shape[0]:
        raise ValueError("staged ticker/date keys overlap another production batch")

    copy_rows: list[dict[str, Any]] = []
    fixed_targets = {
        "output_path": Path(production_plan_row["output_path"]),
        "coverage_path": Path(production_plan_row["coverage_path"]),
    }
    if positive:
        fixed_targets.update(
            {
                "ohlcv_path": Path(production_plan_row["ohlcv_path"]),
                "canonical_coverage_path": Path(production_plan_row["coverage_path"]).with_name(
                    "prices.canonical_coverage.csv"
                ),
            }
        )
    target_modes: dict[str, str] = {}
    for field, target in fixed_targets.items():
        source = Path(stage_row[field]).resolve()
        _require_within(source, stage_lake)
        resolved_target = target.resolve()
        if not resolved_target.exists():
            target_modes[field] = "new"
        elif _sha(resolved_target) == _sha(source):
            target_modes[field] = "identical"
        elif _verified_failed_placeholder(field, resolved_target, stage_plan_row["tickers"].split(",")):
            target_modes[field] = "placeholder"
        else:
            raise ValueError(f"pre-existing production payload is not recoverable: {resolved_target}")

    if positive:
        copy_rows.extend(_write_batch_partitions(staged, production_long, plan_row_id))
    for field, target in fixed_targets.items():
        source = Path(stage_row[field]).resolve()
        _require_within(source, stage_lake)
        resolved_target = target.resolve()
        replaced_placeholder = ""
        mode = target_modes[field]
        if mode == "placeholder":
            replaced_placeholder = _sha(resolved_target)
        if mode != "identical":
            _atomic_copy(source, resolved_target, allow_replace=bool(replaced_placeholder))
        record = _copy_record(field, source, resolved_target)
        record["replaced_placeholder_sha256"] = replaced_placeholder
        record["copy_mode"] = mode
        copy_rows.append(record)

    before_manifest_sha256 = _sha(production_manifest_path)
    replacement = dict(stage_row)
    for field, target in fixed_targets.items():
        replacement[field] = str(target)
    replacement["long_ohlcv_root"] = str(production_long)
    columns = list(dict.fromkeys([*production_manifest.columns, *stage_manifest.columns]))
    updated = production_manifest.reindex(columns=columns, fill_value="")
    mask = updated["plan_row_id"].eq(plan_row_id)
    for column in columns:
        updated.loc[mask, column] = replacement.get(column, "")
    _atomic_csv(production_manifest_path, updated)

    quarantine_path = production_artifact / "quarantined_plan_rows.csv"
    quarantine_before = _sha(quarantine_path)
    quarantine = _csv(quarantine_path)
    if quarantine["plan_row_id"].eq(plan_row_id).sum() != 1:
        raise ValueError("quarantine row is not exactly one before promotion")
    _atomic_csv(quarantine_path, quarantine.loc[~quarantine["plan_row_id"].eq(plan_row_id)])

    copied = pd.DataFrame(copy_rows)
    copied_path = root / "copied_files.csv"
    _atomic_csv(copied_path, copied)
    manifest = {
        "schema_version": "1.1-guarded-terminal-recovery-merge",
        "run_id": run_id,
        "status": "merged_and_validated",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "plan_row_id": plan_row_id,
        "stage_status": stage_row["status"],
        "merged_long_ohlcv_rows": expected_rows,
        "merged_tickers": sorted(staged["ticker"].unique()),
        "recovery_kind": "positive_payload" if positive else "structural_zero_payload",
        "copied_files": len(copied),
        "source": {
            "stage_plan_sha256": expected_stage_plan_sha256,
            "stage_manifest_sha256": _sha(stage_manifest_path),
            "production_plan_sha256": expected_production_plan_sha256,
        },
        "production": {
            "manifest_path": str(production_manifest_path),
            "manifest_sha256_before": before_manifest_sha256,
            "manifest_sha256_after": _sha(production_manifest_path),
            "quarantine_path": str(quarantine_path),
            "quarantine_sha256_before": quarantine_before,
            "quarantine_sha256_after": _sha(quarantine_path),
            "production_row_status": replacement["status"],
        },
        "checks": {
            "plan_identity_equal": True,
            "canonical_rows_reconciled": positive,
            "structural_reason_present": structural,
            "stage_keys_unique": True,
            "production_batch_absent_before_merge": True,
            "cross_batch_key_overlap": 0,
            "atomic_new_file_copy": True,
            "quarantine_row_removed": True,
        },
        "outputs": {"copied_files.csv": _descriptor(copied_path)},
        "promotion_eligible": False,
        "alpha_or_trading_promoted": False,
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    previous_marker = _read_json(stage_artifact / "guarded_merge_manifest.json")
    marker = {
        "status": manifest["status"],
        "plan_row_id": plan_row_id,
        "merged_long_ohlcv_rows": expected_rows,
        "merge_manifest_path": str(manifest_path),
        "merge_manifest_sha256": _sha(manifest_path),
    }
    if previous_marker.get("initial_gate_manifest_path"):
        marker["initial_gate_plan_row_id"] = previous_marker["initial_gate_plan_row_id"]
        marker["initial_gate_manifest_path"] = previous_marker["initial_gate_manifest_path"]
        marker["initial_gate_manifest_sha256"] = previous_marker["initial_gate_manifest_sha256"]
    elif previous_marker.get("merge_manifest_path") and int(previous_marker.get("merged_long_ohlcv_rows", 0)) > 0:
        marker["initial_gate_plan_row_id"] = previous_marker["plan_row_id"]
        marker["initial_gate_manifest_path"] = previous_marker["merge_manifest_path"]
        marker["initial_gate_manifest_sha256"] = previous_marker["merge_manifest_sha256"]
    elif positive:
        marker["initial_gate_plan_row_id"] = plan_row_id
        marker["initial_gate_manifest_path"] = str(manifest_path)
        marker["initial_gate_manifest_sha256"] = _sha(manifest_path)
    _atomic_json(stage_artifact / "guarded_merge_manifest.json", marker)
    return JapanOhlcvRecoveryMergeArtifacts(run_id, root, manifest_path, manifest)


def _batch_table(root: Path, batch_id: str, *, allow_missing: bool = False) -> pd.DataFrame:
    if not root.is_dir() or not any(root.rglob("*.parquet")):
        if allow_missing:
            return pd.DataFrame(columns=["ticker", "date", "batch_id"])
        raise ValueError(f"long OHLCV root is missing: {root}")
    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    table = dataset.to_table(filter=ds.field("batch_id") == batch_id)
    return table.to_pandas()


def _ticker_table(root: Path, tickers: list[str]) -> pd.DataFrame:
    if not root.is_dir() or not any(root.rglob("*.parquet")):
        return pd.DataFrame(columns=["ticker", "date"])
    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    return dataset.to_table(
        columns=["ticker", "date"], filter=ds.field("ticker").isin(tickers)
    ).to_pandas()


def _write_batch_partitions(
    staged: pd.DataFrame, production_root: Path, plan_row_id: str
) -> list[dict[str, Any]]:
    partition_columns = [
        "provider_partition",
        "interval_partition",
        "adjustment_partition",
        "country_partition",
        "exchange_partition",
        "year_partition",
        "month_partition",
    ]
    if any(column not in staged.columns for column in partition_columns):
        raise ValueError("staged batch is missing required partition columns")
    data_columns = [column for column in staged.columns if column not in partition_columns]
    safe_plan_row_id = "".join(character if character.isalnum() or character in "-_" else "-" for character in plan_row_id)
    records: list[dict[str, Any]] = []
    for keys, group in staged.groupby(partition_columns, dropna=False, sort=True, observed=True):
        target_dir = production_root
        for column, value in zip(partition_columns, keys, strict=True):
            target_dir = target_dir / f"{column}={value}"
        target = target_dir / f"recovery-{safe_plan_row_id}.parquet"
        payload = group[data_columns].reset_index(drop=True)
        mode = "new"
        if target.exists():
            existing = pd.read_parquet(target)
            if len(existing) != len(payload) or set(existing["batch_id"].astype(str)) != {plan_row_id}:
                raise ValueError(f"pre-existing recovery partition does not reconcile: {target}")
            mode = "identical_checkpoint"
        else:
            _atomic_parquet(target, payload)
        record = {
            "kind": "long_ohlcv_partition",
            "source": "filtered_stage_batch",
            "target": str(target),
            "sha256": _sha(target),
            "bytes": target.stat().st_size,
            "rows": len(payload),
            "copy_mode": mode,
        }
        records.append(record)
    if sum(int(record["rows"]) for record in records) != len(staged):
        raise ValueError("partitioned recovery writes do not reconcile to the staged batch")
    return records


def _one(frame: pd.DataFrame, plan_row_id: str, label: str) -> dict[str, str]:
    rows = frame.loc[frame["plan_row_id"].eq(plan_row_id)]
    if len(rows) != 1:
        raise ValueError(f"{label} row is not exactly one")
    return rows.iloc[0].to_dict()


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _require_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes declared root: {path}") from exc


def _require_hash(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or _sha(path) != expected:
        raise ValueError(f"immutable hash mismatch: {path}")


def _atomic_copy(source: Path, target: Path, *, allow_replace: bool = False) -> None:
    if source.is_symlink() or not source.is_file() or (target.exists() and not allow_replace):
        raise ValueError(f"unsafe or pre-existing recovery copy: {source} -> {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        if _sha(Path(temporary)) != _sha(source):
            raise ValueError("recovery copy hash mismatch")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary_name, index=False)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _verified_failed_placeholder(field: str, path: Path, planned_tickers: list[str]) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        if field == "output_path":
            frame = pd.read_parquet(path)
            return frame.empty and len(frame.columns) == 0
        if field == "coverage_path":
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            return bool(
                len(frame) == len(planned_tickers)
                and set(frame["ticker"]) == set(planned_tickers)
                and frame["rows"].astype(int).eq(0).all()
            )
    except Exception:
        return False
    return False


def _copy_record(kind: str, source: Path, target: Path) -> dict[str, Any]:
    return {
        "kind": kind,
        "source": str(source),
        "target": str(target),
        "sha256": _sha(target),
        "bytes": target.stat().st_size,
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
