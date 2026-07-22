from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


_COMPLETE_STATUSES = {"complete", "completed", "pass", "passed", "success", "succeeded"}
_DEFAULT_KEY_CANDIDATES = (
    ("ticker", "date"),
    ("ric", "date"),
    ("security_id", "date"),
    ("symbol", "date"),
)
_PROMOTION_POLICY = {
    "g2_scorecard_credit": False,
    "alpha": False,
    "historical_backtest": False,
    "production": False,
    "ibkr": False,
    "live_trading": False,
}


@dataclass(frozen=True)
class JapanG2PriceUnionResult:
    run_id: str
    artifact_dir: Path
    dataset_root: Path
    manifest_path: Path
    checksums_path: Path
    source_files_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _Lane:
    role: str
    root: Path
    compaction_path: Path
    row_equality_path: Path
    compaction: dict[str, Any]
    row_equality: dict[str, Any]
    expected_rows: int
    files: tuple[Path, ...]


def build_japan_g2_price_union(
    *,
    active_compacted_root: str | Path,
    active_compaction_manifest: str | Path,
    active_row_equality_manifest: str | Path,
    reference_compacted_root: str | Path,
    reference_compaction_manifest: str | Path,
    reference_row_equality_manifest: str | Path,
    artifact_root: str | Path = "artifacts/japan_g2_price_union",
    run_id: str,
    key_columns: Sequence[str] | None = None,
    batch_size: int = 8_192,
    copy_chunk_size: int = 1024 * 1024,
) -> JapanG2PriceUnionResult:
    """Publish a row-unique active/reference Parquet union with bounded memory use.

    Parquet rows are scanned in record batches and uniqueness is maintained in
    on-disk SQLite indexes. Source files are copied byte-for-byte in chunks so a
    published artifact cannot later change with a hardlinked source.
    """

    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path component")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if copy_chunk_size <= 0:
        raise ValueError("copy_chunk_size must be positive")

    artifact_root = Path(artifact_root)
    artifact_dir = artifact_root / run_id
    if artifact_dir.exists():
        raise FileExistsError(f"immutable Japan G2 price-union artifact already exists: {artifact_dir}")

    lanes = (
        _validate_lane(
            role="active",
            compacted_root=active_compacted_root,
            compaction_manifest=active_compaction_manifest,
            row_equality_manifest=active_row_equality_manifest,
        ),
        _validate_lane(
            role="reference",
            compacted_root=reference_compacted_root,
            compaction_manifest=reference_compaction_manifest,
            row_equality_manifest=reference_row_equality_manifest,
        ),
    )
    common_schema = _validate_schemas(lanes)
    resolved_key_columns = _resolve_key_columns(common_schema, key_columns)

    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=artifact_root))
    dataset_dir = temporary / "price_union"
    sqlite_path = temporary / ".uniqueness.sqlite3"
    connection = _open_uniqueness_index(sqlite_path)
    inventory: list[dict[str, Any]] = []
    role_rows = {lane.role: 0 for lane in lanes}
    duplicate_rows = 0
    duplicate_keys = 0
    null_key_rows = 0
    manifest_hashes = _input_manifest_hashes(lanes)

    try:
        for lane in lanes:
            for source_path in lane.files:
                relative_path = source_path.relative_to(lane.root)
                output_path = dataset_dir / lane.role / relative_path
                source_sha256_before = _sha256(source_path)
                rows, row_duplicates, key_duplicates, null_keys = _index_parquet_file(
                    source_path,
                    connection,
                    common_schema.names,
                    resolved_key_columns,
                    batch_size,
                )
                output_sha256 = _copy_with_sha256(source_path, output_path, copy_chunk_size)
                source_sha256_after = _sha256(source_path)
                if source_sha256_before != source_sha256_after:
                    raise RuntimeError(f"source Parquet changed during build: {source_path}")
                if output_sha256 != source_sha256_before:
                    raise RuntimeError(f"copied Parquet hash mismatch: {source_path}")

                metadata_rows = int(pq.ParquetFile(source_path).metadata.num_rows)
                if rows != metadata_rows:
                    raise RuntimeError(
                        f"Parquet scan count mismatch for {source_path}: metadata={metadata_rows}, scanned={rows}"
                    )
                role_rows[lane.role] += rows
                duplicate_rows += row_duplicates
                duplicate_keys += key_duplicates
                null_key_rows += null_keys
                inventory.append(
                    {
                        "role": lane.role,
                        "source_root": str(lane.root),
                        "source_path": str(source_path),
                        "source_relative_path": relative_path.as_posix(),
                        "output_relative_path": output_path.relative_to(temporary).as_posix(),
                        "rows": rows,
                        "size_bytes": int(source_path.stat().st_size),
                        "sha256": source_sha256_before,
                        "source_hash_verified_after_copy": True,
                        "output_hash_verified": True,
                    }
                )
            if role_rows[lane.role] != lane.expected_rows:
                raise ValueError(
                    f"{lane.role} Parquet rows do not reconcile to manifests: "
                    f"expected={lane.expected_rows}, actual={role_rows[lane.role]}"
                )

        if null_key_rows:
            raise ValueError(f"price union contains {null_key_rows} rows with null key values")
        if duplicate_keys:
            raise ValueError(f"price union contains {duplicate_keys} duplicate key rows")
        if duplicate_rows:
            raise ValueError(f"price union contains {duplicate_rows} exact duplicate rows")

        _verify_sources_unchanged(lanes, inventory)
        _verify_input_manifests_unchanged(manifest_hashes)
        connection.close()
        connection = None
        sqlite_path.unlink(missing_ok=True)

        inventory.sort(key=lambda row: (str(row["role"]), str(row["source_relative_path"])))
        source_files_path = temporary / "source_files.json"
        source_files_path.write_text(
            json.dumps({"algorithm": "sha256", "files": inventory}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        total_rows = sum(role_rows.values())
        manifest = {
            "run_id": run_id,
            "artifact_type": "japan_g2_price_union",
            "status": "complete",
            "evidence_scope": "active_reference_compacted_price_union_only",
            "dataset_root": str((artifact_dir / "price_union").resolve()),
            "roles": ["active", "reference"],
            "key_columns": list(resolved_key_columns),
            "schema": [{"name": field.name, "type": str(field.type)} for field in common_schema],
            "source_files": len(inventory),
            "source_rows": total_rows,
            "output_files": len(inventory),
            "output_rows": total_rows,
            "role_rows": role_rows,
            "exact_duplicate_rows": 0,
            "duplicate_key_rows": 0,
            "null_key_rows": 0,
            "exact_rows_unique": True,
            "keys_unique": True,
            "source_file_hashes_verified": True,
            "input_manifest_hashes_verified": True,
            "memory_bounded": True,
            "batch_size": batch_size,
            "copy_chunk_size": copy_chunk_size,
            "source_manifests": _manifest_inventory(lanes, manifest_hashes),
            "checksums_file": str((artifact_dir / "checksums.sha256").resolve()),
            "promotion": False,
            "promotion_allowed": False,
            "scorecard_eligible": False,
            "promotion_policy": dict(_PROMOTION_POLICY),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        checksums_path = temporary / "checksums.sha256"
        checksum_targets = [*sorted(dataset_dir.rglob("*.parquet")), source_files_path, manifest_path]
        checksums_path.write_text(
            "".join(
                f"{_sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in checksum_targets
            ),
            encoding="utf-8",
        )
        _verify_checksums(temporary, checksums_path)
        os.replace(temporary, artifact_dir)
    except Exception:
        if connection is not None:
            connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return JapanG2PriceUnionResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        dataset_root=artifact_dir / "price_union",
        manifest_path=artifact_dir / "manifest.json",
        checksums_path=artifact_dir / "checksums.sha256",
        source_files_path=artifact_dir / "source_files.json",
        manifest=manifest,
    )


def write_japan_g2_price_union(**kwargs: Any) -> JapanG2PriceUnionResult:
    """Compatibility name matching the repository's immutable artifact writers."""

    return build_japan_g2_price_union(**kwargs)


def _validate_lane(
    *,
    role: str,
    compacted_root: str | Path,
    compaction_manifest: str | Path,
    row_equality_manifest: str | Path,
) -> _Lane:
    root = Path(compacted_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{role} compacted Parquet root is required: {root}")
    compaction_path = _manifest_path(compaction_manifest)
    equality_path = _manifest_path(row_equality_manifest)
    compaction = _read_json(compaction_path)
    equality = _read_json(equality_path)

    compaction_status = str(compaction.get("status", "")).strip().lower()
    source_rows = _required_int(compaction, "source_rows", role, "compaction")
    output_rows = _required_int(compaction, "output_rows", role, "compaction")
    failed_groups = _required_int(compaction, "failed_groups", role, "compaction")
    declared_output = _declared_root(compaction_path, compaction, ("output_root", "compacted_root"))
    if (
        compaction_status not in _COMPLETE_STATUSES
        or source_rows <= 0
        or source_rows != output_rows
        or failed_groups != 0
        or declared_output != root
    ):
        raise ValueError(f"{role} compaction manifest is incomplete or does not describe the compacted root")

    equality_status = str(equality.get("status", "")).strip().lower()
    equality_source_rows = _required_int(equality, "source_rows", role, "row-equality")
    equality_rows = _required_int(equality, "compacted_rows", role, "row-equality")
    declared_compacted = _declared_root(equality_path, equality, ("compacted_root", "output_root"))
    if (
        equality_status not in _COMPLETE_STATUSES
        or equality.get("row_equality_passed") is not True
        or equality_source_rows <= 0
        or equality_source_rows != equality_rows
        or equality.get("failed_years") != []
        or declared_compacted != root
        or equality_rows != output_rows
    ):
        raise ValueError(f"{role} row-equality manifest is incomplete, non-equal, or inconsistent")

    files = tuple(sorted(root.rglob("*.parquet"), key=lambda path: path.relative_to(root).as_posix()))
    if not files:
        raise ValueError(f"{role} compacted root contains no Parquet files")
    for path in files:
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(root):
            raise ValueError(f"{role} compacted root contains an unsafe Parquet path: {path}")
    return _Lane(
        role=role,
        root=root,
        compaction_path=compaction_path,
        row_equality_path=equality_path,
        compaction=compaction,
        row_equality=equality,
        expected_rows=output_rows,
        files=files,
    )


def _manifest_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"required manifest does not exist: {path}")
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return value


def _required_int(manifest: dict[str, Any], key: str, role: str, kind: str) -> int:
    value = manifest.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{role} {kind} manifest has invalid {key}")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{role} {kind} manifest is missing integer {key}") from exc
    if value != integer:
        raise ValueError(f"{role} {kind} manifest has non-integral {key}")
    return integer


def _declared_root(path: Path, manifest: dict[str, Any], keys: Iterable[str]) -> Path | None:
    for key in keys:
        value = manifest.get(key)
        if value:
            declared = Path(str(value)).expanduser()
            if not declared.is_absolute():
                declared = path.parent / declared
            return declared.resolve()
    return None


def _validate_schemas(lanes: Sequence[_Lane]) -> pa.Schema:
    expected: pa.Schema | None = None
    signature: tuple[tuple[str, str, bool], ...] | None = None
    for lane in lanes:
        for path in lane.files:
            schema = pq.ParquetFile(path).schema_arrow.remove_metadata()
            current = tuple(sorted((field.name, str(field.type), field.nullable) for field in schema))
            if signature is None:
                signature = current
                expected = pa.schema(sorted(schema, key=lambda field: field.name))
            elif current != signature:
                raise ValueError(f"incompatible Parquet schema in price union: {path}")
    if expected is None:
        raise ValueError("price union has no Parquet schema")
    return expected


def _resolve_key_columns(schema: pa.Schema, requested: Sequence[str] | None) -> tuple[str, ...]:
    names = set(schema.names)
    if requested is not None:
        columns = tuple(str(column).strip() for column in requested)
        if not columns or any(not column for column in columns) or len(set(columns)) != len(columns):
            raise ValueError("key_columns must contain distinct, non-empty column names")
        missing = [column for column in columns if column not in names]
        if missing:
            raise ValueError(f"key columns are absent from Parquet schema: {missing}")
        return columns
    for candidate in _DEFAULT_KEY_CANDIDATES:
        if set(candidate).issubset(names):
            return candidate
    raise ValueError("could not infer a price key; pass key_columns explicitly")


def _open_uniqueness_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-8192")
    connection.execute("CREATE TABLE seen_rows (value BLOB PRIMARY KEY) WITHOUT ROWID")
    connection.execute("CREATE TABLE seen_keys (value BLOB PRIMARY KEY) WITHOUT ROWID")
    return connection


def _index_parquet_file(
    path: Path,
    connection: sqlite3.Connection,
    row_columns: Sequence[str],
    key_columns: Sequence[str],
    batch_size: int,
) -> tuple[int, int, int, int]:
    parquet = pq.ParquetFile(path)
    rows = 0
    duplicate_rows = 0
    duplicate_keys = 0
    null_keys = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=list(row_columns), use_threads=False):
        row_values: list[tuple[bytes]] = []
        key_values: list[tuple[bytes]] = []
        column_positions = {name: batch.schema.get_field_index(name) for name in batch.schema.names}
        for row_index in range(batch.num_rows):
            values = tuple(batch.column(column_positions[name])[row_index].as_py() for name in row_columns)
            key = tuple(batch.column(column_positions[name])[row_index].as_py() for name in key_columns)
            if any(_is_null_key(value) for value in key):
                null_keys += 1
            row_values.append((_encode_tuple(values),))
            key_values.append((_encode_tuple(key),))
        before = connection.total_changes
        connection.executemany("INSERT OR IGNORE INTO seen_rows(value) VALUES (?)", row_values)
        duplicate_rows += len(row_values) - (connection.total_changes - before)
        before = connection.total_changes
        connection.executemany("INSERT OR IGNORE INTO seen_keys(value) VALUES (?)", key_values)
        duplicate_keys += len(key_values) - (connection.total_changes - before)
        rows += batch.num_rows
        connection.commit()
    return rows, duplicate_rows, duplicate_keys, null_keys


def _is_null_key(value: Any) -> bool:
    return value is None or isinstance(value, float) and math.isnan(value)


def _encode_tuple(values: Sequence[Any]) -> bytes:
    return json.dumps(
        [_canonical_value(value) for value in values],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_value(value: Any) -> Any:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if math.isnan(value):
            return ["float", "nan"]
        if math.isinf(value):
            return ["float", "+inf" if value > 0 else "-inf"]
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, date):
        return ["date", value.isoformat()]
    if isinstance(value, time):
        return ["time", value.isoformat()]
    if isinstance(value, timedelta):
        return ["timedelta", value.days, value.seconds, value.microseconds]
    if isinstance(value, (list, tuple)):
        return ["list", [_canonical_value(item) for item in value]]
    if isinstance(value, dict):
        items = [(_encode_tuple((key,)).decode("utf-8"), _canonical_value(item)) for key, item in value.items()]
        return ["dict", sorted(items)]
    return [f"{type(value).__module__}.{type(value).__qualname__}", repr(value)]


def _copy_with_sha256(source: Path, output: Path, chunk_size: int) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as reader, output.open("xb") as writer:
        while chunk := reader.read(chunk_size):
            writer.write(chunk)
            digest.update(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    shutil.copystat(source, output, follow_symlinks=False)
    return digest.hexdigest()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _input_manifest_hashes(lanes: Sequence[_Lane]) -> dict[Path, str]:
    paths = {path for lane in lanes for path in (lane.compaction_path, lane.row_equality_path)}
    return {path: _sha256(path) for path in sorted(paths)}


def _verify_input_manifests_unchanged(hashes: dict[Path, str]) -> None:
    for path, expected in hashes.items():
        if _sha256(path) != expected:
            raise RuntimeError(f"input manifest changed during build: {path}")


def _verify_sources_unchanged(lanes: Sequence[_Lane], inventory: Sequence[dict[str, Any]]) -> None:
    expected_files = {path for lane in lanes for path in lane.files}
    current_files = {
        path
        for lane in lanes
        for path in lane.root.rglob("*.parquet")
        if path.is_file()
    }
    if current_files != expected_files:
        raise RuntimeError("source Parquet file set changed during build")
    expected_hashes = {Path(str(row["source_path"])): str(row["sha256"]) for row in inventory}
    for path in sorted(expected_files):
        if _sha256(path) != expected_hashes[path]:
            raise RuntimeError(f"source Parquet changed during build: {path}")


def _manifest_inventory(lanes: Sequence[_Lane], hashes: dict[Path, str]) -> dict[str, Any]:
    return {
        lane.role: {
            "compaction": {"path": str(lane.compaction_path), "sha256": hashes[lane.compaction_path]},
            "row_equality": {
                "path": str(lane.row_equality_path),
                "sha256": hashes[lane.row_equality_path],
            },
        }
        for lane in lanes
    }


def _verify_checksums(root: Path, checksums_path: Path) -> None:
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if _sha256(root / relative) != expected:
            raise RuntimeError(f"published checksum verification failed: {relative}")
