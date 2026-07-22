"""Local, admitted LSEG point-in-time partition adapter.

The adapter never calls Yahoo or a remote model.  It opens only partitions that
belong to a previously verified ``VerifiedDataAdmission`` bundle, rechecks their
immutable references, and exposes exact market/time/schema lineage to the data
steward and feature pipeline.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from quant_lab.research.data_admission import (
    ArtifactRef,
    DataAdmissionError,
    VerifiedDataAdmission,
    canonical_arrow_schema_sha256,
    open_immutable_ref,
    read_immutable_ref,
)


@dataclass(frozen=True)
class AdmittedLSEGPartition:
    relative_path: str
    absolute_path: Path
    byte_size: int
    sha256: str
    row_count: int
    market: str
    min_effective_at: str
    max_effective_at: str
    min_published_at: str
    max_published_at: str
    min_ingested_at: str
    max_ingested_at: str
    schema_sha256: str
    file_identity: tuple[int, int, int, int, int, int, int, int] | None = None


def _file_identity(details: os.stat_result) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
        stat.S_IMODE(details.st_mode),
        details.st_uid,
        details.st_nlink,
    )


@contextmanager
def _open_identity_checked_partition(
    admission_root: Path, partition: AdmittedLSEGPartition
) -> Iterator[BinaryIO]:
    """Open a previously hash-verified partition for one-pass Arrow scanning."""
    if partition.file_identity is None:
        raise DataAdmissionError("LSEG partition has no post-verification file identity")
    root = admission_root.resolve()
    path = partition.absolute_path
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DataAdmissionError("LSEG partition escaped the admission root") from exc
    cursor = path.parent
    while True:
        if cursor.is_symlink() or not cursor.is_dir():
            raise DataAdmissionError("LSEG partition parent is not a regular directory")
        details = cursor.stat()
        if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o022:
            raise DataAdmissionError("LSEG partition directories lost immutable custody")
        if cursor.resolve() == root:
            break
        if cursor == cursor.parent:
            raise DataAdmissionError("LSEG partition parent escaped its admission root")
        cursor = cursor.parent
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DataAdmissionError(f"cannot open admitted LSEG partition: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) & 0o222
            or before.st_nlink != 1
            or _file_identity(before) != partition.file_identity
        ):
            raise DataAdmissionError("LSEG partition identity changed after hash verification")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
        if _file_identity(os.fstat(descriptor)) != partition.file_identity:
            raise DataAdmissionError("LSEG partition changed during Arrow scan")
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class AdmittedLSEGDataset:
    """Sealed partition descriptor with bounded, hash-rechecked batch scans."""

    admission_root: Path
    admission_subject_sha256: str
    data_manifest_id: str
    partitions: tuple[AdmittedLSEGPartition, ...]
    schema: pa.Schema
    row_count: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "admission_subject_sha256": self.admission_subject_sha256,
            "data_manifest_id": self.data_manifest_id,
            "schema_sha256": canonical_arrow_schema_sha256(self.schema),
            "partition_count": len(self.partitions),
            "row_count": self.row_count,
            "markets": sorted({partition.market for partition in self.partitions}),
            "scanner": "AdmittedLSEGDataset.iter_batches",
            "default_batch_size": 65_536,
        }

    def iter_batches(
        self,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        if batch_size < 1 or batch_size > 65_536:
            raise ValueError("batch_size must be between 1 and 65,536")
        selected = list(columns) if columns is not None else None
        if selected is not None and (not selected or not set(selected).issubset(self.schema.names)):
            raise ValueError("scanner columns must be a nonempty subset of the admitted schema")
        for partition in self.partitions:
            with _open_identity_checked_partition(self.admission_root, partition) as handle:
                parquet_file = pq.ParquetFile(handle)
                for batch in parquet_file.iter_batches(
                    batch_size=batch_size, columns=selected
                ):
                    yield batch


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def admitted_lseg_partitions(
    *,
    admission_root: Path,
    admission_bundle_path: Path,
    verified: VerifiedDataAdmission,
    verify_partition_bytes: bool = True,
) -> tuple[AdmittedLSEGPartition, ...]:
    """Return only hash-verified local partitions from the sealed admission."""
    root = admission_root.expanduser().resolve()
    if admission_bundle_path.is_symlink() or not admission_bundle_path.is_file():
        raise DataAdmissionError("LSEG adapter bundle must be a regular non-symlink file")
    try:
        bundle_relative = admission_bundle_path.resolve().relative_to(root)
    except ValueError as exc:
        raise DataAdmissionError("LSEG adapter bundle escaped the admission root") from exc
    bundle_details = admission_bundle_path.stat()
    bundle_reference = ArtifactRef(
        bundle_relative.as_posix(), bundle_details.st_size, verified.admission_bundle_sha256
    )
    bundle_bytes = read_immutable_ref(root, bundle_reference, "admission bundle")
    if _sha(bundle_bytes) != verified.admission_bundle_sha256:
        raise DataAdmissionError("LSEG adapter bundle differs from the verified admission")
    bundle = json.loads(bundle_bytes)
    if not isinstance(bundle, Mapping):
        raise DataAdmissionError("LSEG adapter bundle must contain an object")
    manifest_ref = ArtifactRef.from_dict(bundle["data_manifest_ref"], "data_manifest_ref")
    if manifest_ref.sha256 != verified.data_manifest_sha256:
        raise DataAdmissionError("LSEG adapter data manifest is not the verified manifest")
    manifest = json.loads(read_immutable_ref(root, manifest_ref, "data_manifest_ref"))
    if manifest["manifest_id"] != verified.data_manifest_id:
        raise DataAdmissionError("LSEG adapter manifest ID differs from the verified admission")
    inventory_ref = ArtifactRef.from_dict(manifest["inventory_ref"], "inventory_ref")
    inventory = json.loads(read_immutable_ref(root, inventory_ref, "inventory_ref"))
    if not isinstance(inventory, list) or len(inventory) != manifest["partition_count"]:
        raise DataAdmissionError("LSEG adapter inventory count differs from the admitted manifest")
    partition_root = Path(manifest["partition_root"])
    rows: list[AdmittedLSEGPartition] = []
    for item in inventory:
        if not isinstance(item, Mapping):
            raise DataAdmissionError("LSEG adapter inventory row must be an object")
        reference = ArtifactRef(
            path=(partition_root / item["relative_path"]).as_posix(),
            byte_size=item["byte_size"],
            sha256=item["sha256"],
        )
        identity = None
        if verify_partition_bytes:
            with open_immutable_ref(
                root, reference, f"LSEG partition {item['relative_path']}"
            ) as handle:
                identity = _file_identity(os.fstat(handle.fileno()))
        rows.append(
            AdmittedLSEGPartition(
                relative_path=item["relative_path"],
                absolute_path=root / reference.path,
                byte_size=item["byte_size"],
                sha256=item["sha256"],
                row_count=item["row_count"],
                market=item["market"],
                min_effective_at=item["min_effective_at"],
                max_effective_at=item["max_effective_at"],
                min_published_at=item["min_published_at"],
                max_published_at=item["max_published_at"],
                min_ingested_at=item["min_ingested_at"],
                max_ingested_at=item["max_ingested_at"],
                schema_sha256=item["schema_sha256"],
                file_identity=identity,
            )
        )
    if tuple(sorted(row.market for row in rows)) != tuple(
        sorted(item["market"] for item in inventory)
    ):
        raise DataAdmissionError("LSEG adapter market lineage changed during materialization")
    return tuple(rows)


def open_admitted_lseg_dataset(
    *,
    admission_root: Path,
    admission_bundle_path: Path,
    verified: VerifiedDataAdmission,
) -> AdmittedLSEGDataset:
    """Open a bounded scanner descriptor without materializing the full lake."""
    root = admission_root.expanduser().resolve()
    partitions = admitted_lseg_partitions(
        admission_root=root,
        admission_bundle_path=admission_bundle_path,
        verified=verified,
        verify_partition_bytes=False,
    )
    if not partitions:
        raise DataAdmissionError("verified LSEG admission contains no partitions")
    schema: pa.Schema | None = None
    expected_schema_hash: str | None = None
    row_count = 0
    verified_partitions: list[AdmittedLSEGPartition] = []
    for partition in partitions:
        reference = ArtifactRef(
            path=partition.absolute_path.relative_to(root).as_posix(),
            byte_size=partition.byte_size,
            sha256=partition.sha256,
        )
        with open_immutable_ref(root, reference, f"LSEG partition {partition.relative_path}") as handle:
            identity = _file_identity(os.fstat(handle.fileno()))
            parquet_file = pq.ParquetFile(handle)
            partition_schema = parquet_file.schema_arrow.remove_metadata()
            physical_rows = parquet_file.metadata.num_rows
        schema_hash = canonical_arrow_schema_sha256(partition_schema)
        if schema_hash != partition.schema_sha256:
            raise DataAdmissionError("typed LSEG table schema differs from its signed inventory")
        if expected_schema_hash is None:
            expected_schema_hash = schema_hash
        elif schema_hash != expected_schema_hash:
            raise DataAdmissionError("admitted LSEG partitions do not share one typed schema")
        if physical_rows != partition.row_count:
            raise DataAdmissionError("typed LSEG table row count differs from its signed inventory")
        schema = partition_schema
        row_count += physical_rows
        verified_partitions.append(replace(partition, file_identity=identity))
    if schema is None:
        raise DataAdmissionError("verified LSEG admission has no physical schema")
    return AdmittedLSEGDataset(
        admission_root=root,
        admission_subject_sha256=verified.admission_subject_sha256,
        data_manifest_id=verified.data_manifest_id,
        partitions=tuple(verified_partitions),
        schema=schema,
        row_count=row_count,
    )


__all__ = [
    "AdmittedLSEGPartition",
    "AdmittedLSEGDataset",
    "admitted_lseg_partitions",
    "open_admitted_lseg_dataset",
]
