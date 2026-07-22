from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quant_lab.data.data_lake import _normalize_long_ohlcv_schema, _slug


DEFAULT_HEALTH_GROUP_COLUMNS: tuple[str, ...] = (
    "provider_partition",
    "interval_partition",
    "adjustment_partition",
    "country_partition",
    "exchange_partition",
    "year_partition",
)
DEFAULT_COMPACT_PARTITION_COLUMNS: tuple[str, ...] = ("provider", "interval", "adjustment", "country", "exchange", "year")


@dataclass(frozen=True)
class PartitionHealthReport:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    file_inventory_path: Path
    partition_health_path: Path
    summary_path: Path
    file_inventory: pd.DataFrame
    partition_health: pd.DataFrame
    manifest: dict[str, Any]


@dataclass(frozen=True)
class CompactionResult:
    run_id: str
    artifact_dir: Path
    output_root: Path
    manifest_path: Path
    compaction_manifest_path: Path
    partition_health_before_path: Path
    partition_health_after_path: Path
    summary_path: Path
    compaction_manifest: pd.DataFrame
    partition_health_before: pd.DataFrame
    partition_health_after: pd.DataFrame
    manifest: dict[str, Any]


def scan_parquet_file_inventory(long_ohlcv_root: str | Path) -> pd.DataFrame:
    """Return file-level Parquet metadata without loading the OHLCV rows."""

    root = Path(long_ohlcv_root)
    if not root.exists():
        raise FileNotFoundError(f"long OHLCV root does not exist: {root}")
    files = sorted(root.glob("**/*.parquet"))
    rows: list[dict[str, Any]] = []
    for file_path in files:
        metadata = pq.ParquetFile(file_path).metadata
        partitions = _partition_values(root, file_path)
        rows.append(
            {
                "file_path": str(file_path),
                "relative_path": str(file_path.relative_to(root)),
                "file_size_bytes": int(file_path.stat().st_size),
                "rows": int(metadata.num_rows),
                "row_groups": int(metadata.num_row_groups),
                **partitions,
            }
        )
    return pd.DataFrame(rows)


def build_partition_health(
    file_inventory: pd.DataFrame,
    group_columns: tuple[str, ...] = DEFAULT_HEALTH_GROUP_COLUMNS,
    min_file_size_bytes: int = 1_000_000,
    max_files_per_partition: int = 1,
) -> pd.DataFrame:
    """Summarize tiny-file and duplicate-file risk by logical partition."""

    if file_inventory.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "parquet_files",
                "rows",
                "total_size_bytes",
                "avg_file_size_bytes",
                "max_file_size_bytes",
                "row_groups",
                "small_file_count",
                "duplicate_file_count",
                "status",
                "blocker",
            ]
        )
    present_group_cols = [col for col in group_columns if col in file_inventory.columns]
    if not present_group_cols:
        present_group_cols = ["_root"]
        file_inventory = file_inventory.copy()
        file_inventory["_root"] = "root"
    grouped = (
        file_inventory.groupby(present_group_cols, dropna=False)
        .agg(
            parquet_files=("file_path", "size"),
            rows=("rows", "sum"),
            total_size_bytes=("file_size_bytes", "sum"),
            avg_file_size_bytes=("file_size_bytes", "mean"),
            max_file_size_bytes=("file_size_bytes", "max"),
            row_groups=("row_groups", "sum"),
            small_file_count=("file_size_bytes", lambda values: int((values < min_file_size_bytes).sum())),
        )
        .reset_index()
    )
    grouped["duplicate_file_count"] = grouped["parquet_files"].sub(max_files_per_partition).clip(lower=0).astype(int)
    grouped["status"] = "pass"
    grouped["blocker"] = ""
    too_many = grouped["parquet_files"].gt(max_files_per_partition)
    tiny = grouped["small_file_count"].gt(0)
    grouped.loc[too_many | tiny, "status"] = "warn"
    grouped.loc[too_many, "blocker"] = "too_many_files_for_partition"
    grouped.loc[tiny & ~too_many, "blocker"] = "small_files_below_threshold"
    numeric = ["avg_file_size_bytes", "max_file_size_bytes"]
    for column in numeric:
        grouped[column] = grouped[column].round(2)
    return grouped.sort_values(["status", "parquet_files", "rows"], ascending=[True, False, False]).reset_index(drop=True)


def write_partition_health_report(
    long_ohlcv_root: str | Path,
    artifact_root: str | Path = "artifacts/data_lake_ops",
    run_id: str | None = None,
    min_file_size_bytes: int = 1_000_000,
    max_files_per_partition: int = 1,
) -> PartitionHealthReport:
    run_id = run_id or datetime.now(timezone.utc).strftime("PARTITION-HEALTH-%Y%m%d-%H%M%S")
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    inventory = scan_parquet_file_inventory(long_ohlcv_root)
    health = build_partition_health(
        inventory,
        min_file_size_bytes=min_file_size_bytes,
        max_files_per_partition=max_files_per_partition,
    )

    file_inventory_path = artifact_dir / "file_inventory.csv"
    partition_health_path = artifact_dir / "partition_health.csv"
    summary_path = artifact_dir / "partition_health_summary.md"
    manifest_path = artifact_dir / "manifest.json"
    inventory.to_csv(file_inventory_path, index=False)
    health.to_csv(partition_health_path, index=False)

    manifest = _health_manifest(run_id, artifact_dir, long_ohlcv_root, inventory, health)
    manifest["artifacts"] = {
        "file_inventory": str(file_inventory_path.resolve()),
        "partition_health": str(partition_health_path.resolve()),
        "summary": str(summary_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(_health_summary_md(manifest, health), encoding="utf-8")
    return PartitionHealthReport(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        file_inventory_path=file_inventory_path,
        partition_health_path=partition_health_path,
        summary_path=summary_path,
        file_inventory=inventory,
        partition_health=health,
        manifest=manifest,
    )


def compact_long_ohlcv_partitions(
    long_ohlcv_root: str | Path,
    output_root: str | Path | None = None,
    artifact_root: str | Path = "artifacts/data_lake_ops",
    run_id: str | None = None,
    partition_cols: tuple[str, ...] = DEFAULT_COMPACT_PARTITION_COLUMNS,
    overwrite: bool = False,
    min_file_size_bytes: int = 1_000_000,
    max_files_per_partition: int = 1,
) -> CompactionResult:
    """Compact daily long-OHLCV parquet into year-level partitions one group at a time."""

    run_id = run_id or datetime.now(timezone.utc).strftime("LONG-OHLCV-COMPACTION-%Y%m%d-%H%M%S")
    source_root = Path(long_ohlcv_root)
    if not source_root.exists():
        raise FileNotFoundError(f"long OHLCV root does not exist: {source_root}")
    output_path = Path(output_root) if output_root else source_root.parent / "ohlcv_compacted"
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"compaction output already exists; pass overwrite=True: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    inventory_before = scan_parquet_file_inventory(source_root)
    health_before = build_partition_health(
        inventory_before,
        min_file_size_bytes=min_file_size_bytes,
        max_files_per_partition=max_files_per_partition,
    )
    groups = _inventory_groups(inventory_before)
    manifest_rows: list[dict[str, Any]] = []
    for group_key, files in groups:
        file_paths = files["file_path"].astype(str).tolist()
        try:
            data = pd.read_parquet(file_paths)
            data = _normalize_long_ohlcv_schema(data)
            if "year" not in data.columns or data["year"].isna().all():
                data["date"] = pd.to_datetime(data["date"], utc=True, errors="coerce").dt.tz_localize(None)
                data["year"] = data["date"].dt.year.astype("Int64")
            data = data.dropna(subset=["date", "ticker"]).drop_duplicates(["date", "ticker", "run_id", "batch_id"], keep="last")
            for col in partition_cols:
                if col not in data.columns:
                    data[col] = pd.NA
            physical_cols = []
            for col in partition_cols:
                physical = f"{col}_partition"
                if col == "year":
                    data[physical] = data[col].astype("string").fillna("unknown")
                else:
                    data[physical] = data[col].map(_slug)
                physical_cols.append(physical)
            rows_in = int(files["rows"].sum())
            source_bytes = int(files["file_size_bytes"].sum())
            before_files = _files_under(output_path)
            data.sort_values([col for col in ("date", "ticker") if col in data.columns]).to_parquet(
                output_path,
                index=False,
                partition_cols=physical_cols,
            )
            after_files = _files_under(output_path)
            new_files = sorted(str(path) for path in set(after_files) - set(before_files))
            manifest_rows.append(
                {
                    **group_key,
                    "status": "compacted",
                    "source_files": len(file_paths),
                    "source_rows": rows_in,
                    "source_size_bytes": source_bytes,
                    "output_files": len(new_files),
                    "output_rows": int(len(data)),
                    "output_root": str(output_path),
                    "new_output_files": "|".join(new_files),
                    "error": "",
                }
            )
        except Exception as exc:
            manifest_rows.append(
                {
                    **group_key,
                    "status": "failed",
                    "source_files": len(file_paths),
                    "source_rows": int(files["rows"].sum()),
                    "source_size_bytes": int(files["file_size_bytes"].sum()),
                    "output_files": 0,
                    "output_rows": 0,
                    "output_root": str(output_path),
                    "new_output_files": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    compaction_manifest = pd.DataFrame(manifest_rows)
    inventory_after = scan_parquet_file_inventory(output_path)
    health_after = build_partition_health(
        inventory_after,
        min_file_size_bytes=min_file_size_bytes,
        max_files_per_partition=max_files_per_partition,
    )

    compaction_manifest_path = artifact_dir / "compaction_manifest.csv"
    health_before_path = artifact_dir / "partition_health_before.csv"
    health_after_path = artifact_dir / "partition_health_after.csv"
    summary_path = artifact_dir / "compaction_summary.md"
    manifest_path = artifact_dir / "manifest.json"
    compaction_manifest.to_csv(compaction_manifest_path, index=False)
    health_before.to_csv(health_before_path, index=False)
    health_after.to_csv(health_after_path, index=False)

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "source_root": str(source_root.resolve()),
        "output_root": str(output_path.resolve()),
        "status": "complete" if not compaction_manifest.empty and compaction_manifest["status"].eq("compacted").all() else "partial",
        "source_files": int(len(inventory_before)),
        "output_files": int(len(inventory_after)),
        "source_rows": int(inventory_before["rows"].sum()) if not inventory_before.empty else 0,
        "output_rows": int(inventory_after["rows"].sum()) if not inventory_after.empty else 0,
        "failed_groups": int(compaction_manifest["status"].eq("failed").sum()) if not compaction_manifest.empty else 0,
        "generated_at_utc": _now_utc(),
        "artifacts": {
            "compaction_manifest": str(compaction_manifest_path.resolve()),
            "partition_health_before": str(health_before_path.resolve()),
            "partition_health_after": str(health_after_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(_compaction_summary_md(manifest, compaction_manifest, health_before, health_after), encoding="utf-8")
    return CompactionResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        output_root=output_path,
        manifest_path=manifest_path,
        compaction_manifest_path=compaction_manifest_path,
        partition_health_before_path=health_before_path,
        partition_health_after_path=health_after_path,
        summary_path=summary_path,
        compaction_manifest=compaction_manifest,
        partition_health_before=health_before,
        partition_health_after=health_after,
        manifest=manifest,
    )


def _partition_values(root: Path, file_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in file_path.relative_to(root).parts[:-1]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    return values


def _inventory_groups(inventory: pd.DataFrame) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    if inventory.empty:
        return []
    group_cols = [col for col in DEFAULT_HEALTH_GROUP_COLUMNS if col in inventory.columns]
    if not group_cols:
        return [({}, inventory)]
    grouped = []
    for key, group in inventory.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        grouped.append((dict(zip(group_cols, key, strict=False)), group.copy()))
    return grouped


def _files_under(path: Path) -> set[Path]:
    return set(path.glob("**/*.parquet")) if path.exists() else set()


def _health_manifest(
    run_id: str,
    artifact_dir: Path,
    long_ohlcv_root: str | Path,
    inventory: pd.DataFrame,
    health: pd.DataFrame,
) -> dict[str, Any]:
    warn_partitions = int(health["status"].eq("warn").sum()) if not health.empty and "status" in health.columns else 0
    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "long_ohlcv_root": str(Path(long_ohlcv_root).resolve()),
        "status": "pass" if warn_partitions == 0 and not inventory.empty else "warn" if not inventory.empty else "empty",
        "parquet_files": int(len(inventory)),
        "partitions": int(len(health)),
        "warn_partitions": warn_partitions,
        "rows": int(inventory["rows"].sum()) if not inventory.empty else 0,
        "total_size_bytes": int(inventory["file_size_bytes"].sum()) if not inventory.empty else 0,
        "generated_at_utc": _now_utc(),
    }


def _health_summary_md(manifest: dict[str, Any], health: pd.DataFrame) -> str:
    worst = health.sort_values(["status", "parquet_files"], ascending=[False, False]).head(20) if not health.empty else health
    return "\n".join(
        [
            f"# Partition Health - {manifest['run_id']}",
            "",
            f"- Status: `{manifest['status']}`",
            f"- Rows: {manifest['rows']:,}",
            f"- Parquet files: {manifest['parquet_files']:,}",
            f"- Partitions: {manifest['partitions']:,}",
            f"- Warning partitions: {manifest['warn_partitions']:,}",
            "",
            "## Worst Partitions",
            "",
            _markdown_table(worst),
            "",
        ]
    )


def _compaction_summary_md(
    manifest: dict[str, Any],
    compaction_manifest: pd.DataFrame,
    health_before: pd.DataFrame,
    health_after: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            f"# Long OHLCV Compaction - {manifest['run_id']}",
            "",
            f"- Status: `{manifest['status']}`",
            f"- Source files: {manifest['source_files']:,}",
            f"- Output files: {manifest['output_files']:,}",
            f"- Source rows: {manifest['source_rows']:,}",
            f"- Output rows: {manifest['output_rows']:,}",
            f"- Failed groups: {manifest['failed_groups']:,}",
            "",
            "## Before Health",
            "",
            _markdown_table(health_before.head(20)),
            "",
            "## After Health",
            "",
            _markdown_table(health_after.head(20)),
            "",
            "## Compaction Manifest",
            "",
            _markdown_table(compaction_manifest.head(20)),
            "",
        ]
    )


def _markdown_table(data: pd.DataFrame) -> str:
    if data.empty:
        return "_No rows._"
    display = data.copy()
    columns = [str(col) for col in display.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in display.astype("object").where(pd.notna(display), "").to_dict("records"):
        rows.append("| " + " | ".join(str(record[col]).replace("|", "\\|").replace("\n", " ") for col in display.columns) + " |")
    return "\n".join([header, separator, *rows])


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CompactionResult",
    "PartitionHealthReport",
    "build_partition_health",
    "compact_long_ohlcv_partitions",
    "scan_parquet_file_inventory",
    "write_partition_health_report",
]
