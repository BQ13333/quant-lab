from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from quant_lab.data.japan_g4_halt_report_normalization_repair import (
    AGGREGATE_DISPOSITION,
    APPENDIX_DISPOSITION,
    NORMALIZED_DISPOSITION,
    verify_halt_report_normalization_repair,
)
from quant_lab.data.japan_g4_halt_report_normalization_repair_validator import (
    verify_halt_report_normalization_repair_validation,
)


PRODUCER_RUN_ID = "JPN-G4-HALT-REPORT-NORMALIZATION-REPAIR-20260712-V1"
VALIDATION_RUN_ID = (
    "JPN-G4-HALT-REPORT-NORMALIZATION-REPAIR-"
    "INDEPENDENT-VALIDATION-20260712-V1"
)

EXPECTED_SOURCE_HASHES = {
    "2012-02": "c86c868a01e18e2afc17b5bf15f408e8c838a3ff2397cf18a81f9cb9581f9720",
    "2013-09": "e7c63b7e0ff65dc6a4ea5ad8a6650c8b6b1306f79454092549c4c42589ba8958",
    "2020-10": "2ff5a87bcc558b574ad38b4ea7c1745f3692b038473812942fb130784bdd4725",
}

_EXPECTED_DISPOSITION_COUNTS = {
    NORMALIZED_DISPOSITION: 1,
    APPENDIX_DISPOSITION: 3,
    AGGREGATE_DISPOSITION: 6,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRODUCER_REQUIRED_FILES = {
    "normalization_repairs.csv",
    "source_hash_validation.csv",
    "unchanged_date_granular_contract.csv",
    *(f"raw/{month}.pdf" for month in EXPECTED_SOURCE_HASHES),
}
_VALIDATION_REQUIRED_FILES = {
    "independent_semantic_validation.csv",
    "independent_source_hash_validation.csv",
    "repair_reconciliation.csv",
    "unchanged_contract_validation.csv",
    "validation_issues.csv",
}


@dataclass(frozen=True)
class ValidatedHaltNormalizationSidecar:
    normalized_repairs: pd.DataFrame
    evidence: pd.DataFrame
    copied_paths: tuple[Path, ...]
    lineage: dict[str, Any]

    @property
    def repairs(self) -> pd.DataFrame:
        return self.normalized_repairs

    @property
    def all_rows(self) -> pd.DataFrame:
        """All ten admitted rows, grouped as repair then evidence."""
        return pd.concat([self.normalized_repairs, self.evidence], ignore_index=True)


def copy_validated_halt_normalization_sidecar(
    producer_dir: str | Path,
    validation_dir: str | Path,
    destination: str | Path,
    *,
    atomic_write: Callable[[Path, bytes], None] | None = None,
) -> ValidatedHaltNormalizationSidecar:
    """Admit and copy the sealed V1 repair sidecar after fail-closed validation."""

    producer = _input_root(producer_dir, "producer")
    validation = _input_root(validation_dir, "validation")
    target = _destination_root(destination, producer, validation)

    producer_manifest, producer_inventory = _audit_packet(
        producer, _PRODUCER_REQUIRED_FILES, "producer"
    )
    validation_manifest, validation_inventory = _audit_packet(
        validation, _VALIDATION_REQUIRED_FILES, "validation"
    )
    producer_verifier_issues = verify_halt_report_normalization_repair(producer)
    validation_verifier_issues = verify_halt_report_normalization_repair_validation(
        validation
    )
    if producer_verifier_issues or validation_verifier_issues:
        raise ValueError(
            "normalization sidecar verifier rejected packet: "
            + "|".join(producer_verifier_issues + validation_verifier_issues)
        )

    producer_manifest_sha256 = _sha256(producer / "manifest.json")
    validation_manifest_sha256 = _sha256(validation / "manifest.json")
    _validate_manifest_contract(
        producer_manifest,
        validation_manifest,
        producer_manifest_sha256=producer_manifest_sha256,
    )

    repairs = _read_csv(producer / "normalization_repairs.csv")
    reconciliation = _read_csv(validation / "repair_reconciliation.csv")
    semantics = _read_csv(validation / "independent_semantic_validation.csv")
    validation_issues = _read_csv(validation / "validation_issues.csv")
    if list(validation_issues.columns) != ["issue_code", "record_id", "detail"] or len(
        validation_issues
    ):
        raise ValueError("normalization sidecar independent validator has issues")
    _validate_exact_reconciliation(repairs, reconciliation, semantics)
    _validate_raw_hashes(producer, validation, repairs)
    _validate_no_invention(repairs)

    writer = atomic_write or _atomic_write
    copied: list[Path] = []
    copied.extend(
        _copy_packet(
            producer,
            target / "producer",
            producer_inventory,
            writer=writer,
        )
    )
    copied.extend(
        _copy_packet(
            validation,
            target / "validation",
            validation_inventory,
            writer=writer,
        )
    )

    copied_producer_manifest, _ = _audit_packet(
        target / "producer", _PRODUCER_REQUIRED_FILES, "copied producer"
    )
    copied_validation_manifest, _ = _audit_packet(
        target / "validation", _VALIDATION_REQUIRED_FILES, "copied validation"
    )
    if (
        copied_producer_manifest != producer_manifest
        or copied_validation_manifest != validation_manifest
        or _sha256(target / "producer" / "manifest.json")
        != producer_manifest_sha256
        or _sha256(target / "validation" / "manifest.json")
        != validation_manifest_sha256
    ):
        raise ValueError("normalization sidecar copy verification failed")

    copied_repairs = _read_csv(target / "producer" / "normalization_repairs.csv")
    normalized = copied_repairs.loc[
        copied_repairs["normalization_disposition"].eq(NORMALIZED_DISPOSITION)
    ].reset_index(drop=True)
    evidence = copied_repairs.loc[
        copied_repairs["normalization_disposition"].isin(
            {APPENDIX_DISPOSITION, AGGREGATE_DISPOSITION}
        )
    ].reset_index(drop=True)
    lineage = {
        "producer_run_id": PRODUCER_RUN_ID,
        "producer_manifest_sha256": producer_manifest_sha256,
        "validation_run_id": VALIDATION_RUN_ID,
        "validation_manifest_sha256": validation_manifest_sha256,
        "repair_rows_reconciled": 10,
        "instrument_event_rows": 1,
        "appendix_reference_rows": 3,
        "market_aggregate_rows": 6,
        "invented_local_codes": 0,
        "source_hashes_verified": True,
        "independent_semantics_verified": True,
        "refinitiv_used": False,
    }
    lineage_path = target / "lineage.json"
    _write_checked(lineage_path, _json_bytes(lineage), writer)
    copied.append(lineage_path)
    return ValidatedHaltNormalizationSidecar(
        normalized_repairs=normalized,
        evidence=evidence,
        copied_paths=tuple(copied),
        lineage=lineage,
    )


def admit_halt_normalization_sidecar(
    producer_dir: str | Path,
    validation_dir: str | Path,
    destination: str | Path,
    *,
    atomic_write: Callable[[Path, bytes], None] | None = None,
) -> ValidatedHaltNormalizationSidecar:
    return copy_validated_halt_normalization_sidecar(
        producer_dir,
        validation_dir,
        destination,
        atomic_write=atomic_write,
    )


def _validate_manifest_contract(
    producer: dict[str, Any],
    validation: dict[str, Any],
    *,
    producer_manifest_sha256: str,
) -> None:
    producer_contract = {
        "schema_version": "1.0-bounded-exact-record-repair",
        "run_id": PRODUCER_RUN_ID,
        "status": "bounded_normalization_repair_candidate_pending_independent_validation",
        "repair_rows": 10,
        "wrapped_time_range_rows": 1,
        "appendix_reference_rows": 3,
        "market_aggregate_rows": 6,
        "instrument_events_created": 1,
        "evidence_only_rows": 9,
        "invented_local_codes": 0,
        "source_hashes_exact": True,
        "independent_validation_required": True,
        "refinitiv_used": False,
        "primary_halt_v8_modified": False,
        "scorecard_eligible": False,
        "write_once": True,
    }
    validation_contract = {
        "schema_version": "1.0-independent-pdfplumber-validation",
        "run_id": VALIDATION_RUN_ID,
        "status": "pass_bounded_normalization_repair",
        "producer_run_id": PRODUCER_RUN_ID,
        "validation_issue_count": 0,
        "repair_rows_expected": 10,
        "repair_rows_reconciled": 10,
        "wrapped_time_range_rows": 1,
        "appendix_reference_rows": 3,
        "market_aggregate_rows": 6,
        "instrument_events_created": 1,
        "evidence_only_rows": 9,
        "invented_local_codes": 0,
        "source_hashes_verified": True,
        "source_semantics_verified": True,
        "source_record_hashes_recomputed": True,
        "event_ids_recomputed": True,
        "independent_parser": "pdfplumber sequential page text",
        "refinitiv_used": False,
        "primary_halt_v8_modified": False,
        "scorecard_eligible": False,
        "write_once": True,
    }
    if any(producer.get(key) != value for key, value in producer_contract.items()):
        raise ValueError("normalization sidecar producer contract failed")
    if any(validation.get(key) != value for key, value in validation_contract.items()):
        raise ValueError("normalization sidecar validation contract failed")
    if validation.get("producer_manifest_sha256") != producer_manifest_sha256:
        raise ValueError("normalization sidecar producer/validator lineage mismatch")
    if validation.get("validation_issue_codes") != []:
        raise ValueError("normalization sidecar validator declared issues")


def _validate_exact_reconciliation(
    repairs: pd.DataFrame,
    reconciliation: pd.DataFrame,
    semantics: pd.DataFrame,
) -> None:
    if any(
        "normalization_disposition" not in frame
        for frame in (repairs, reconciliation, semantics)
    ):
        raise ValueError("normalization sidecar reconciliation schema incomplete")
    disposition_counts = repairs.get(
        "normalization_disposition", pd.Series(dtype=str)
    ).value_counts().to_dict()
    repair_ids = _exact_record_ids(repairs, "repairs")
    reconciled_ids = _exact_record_ids(reconciliation, "reconciliation")
    semantic_ids = _exact_record_ids(semantics, "semantic validation")
    truth_columns = ("exact_match", "source_record_hash_match", "event_id_match")
    if (
        len(repairs) != 10
        or disposition_counts != _EXPECTED_DISPOSITION_COUNTS
        or repair_ids != reconciled_ids
        or repair_ids != semantic_ids
        or any(column not in reconciliation for column in truth_columns)
        or not all(_truths(reconciliation[column]).all() for column in truth_columns)
        or "semantic_match" not in semantics
        or not _truths(semantics["semantic_match"]).all()
    ):
        raise ValueError("normalization sidecar exact 10/10 reconciliation failed")
    repair_dispositions = repairs.set_index("source_record_sha256")[
        "normalization_disposition"
    ].to_dict()
    for frame in (reconciliation, semantics):
        observed = frame.set_index("source_record_sha256")[
            "normalization_disposition"
        ].to_dict()
        if observed != repair_dispositions:
            raise ValueError("normalization sidecar disposition reconciliation failed")


def _validate_raw_hashes(
    producer: Path, validation: Path, repairs: pd.DataFrame
) -> None:
    declared = _read_csv(producer / "source_hash_validation.csv")
    independent = _read_csv(
        validation / "independent_source_hash_validation.csv"
    )
    producer_columns = {
        "report_month",
        "retained_path",
        "expected_sha256",
        "observed_sha256",
        "hash_match",
        "semantic_rows_verified",
    }
    validator_columns = {
        "report_month",
        "expected_sha256",
        "observed_sha256",
        "hash_match",
    }
    if (
        not producer_columns.issubset(declared.columns)
        or not validator_columns.issubset(independent.columns)
        or not {"report_month", "source_sha256"}.issubset(repairs.columns)
    ):
        raise ValueError("normalization sidecar raw hash schema incomplete")
    if (
        len(declared) != 3
        or len(independent) != 3
        or declared["report_month"].duplicated().any()
        or independent["report_month"].duplicated().any()
    ):
        raise ValueError("normalization sidecar raw source denominator mismatch")
    declared_by_month = declared.set_index("report_month").to_dict("index")
    independent_by_month = independent.set_index("report_month").to_dict("index")
    if set(declared_by_month) != set(EXPECTED_SOURCE_HASHES) or set(
        independent_by_month
    ) != set(EXPECTED_SOURCE_HASHES):
        raise ValueError("normalization sidecar raw source month set mismatch")
    for month, expected in EXPECTED_SOURCE_HASHES.items():
        raw_path = producer / "raw" / f"{month}.pdf"
        observed = _sha256(raw_path)
        producer_row = declared_by_month[month]
        validator_row = independent_by_month[month]
        expected_semantic_rows = {"2012-02": "3", "2013-09": "1", "2020-10": "6"}[
            month
        ]
        if (
            observed != expected
            or producer_row.get("retained_path") != f"raw/{month}.pdf"
            or producer_row.get("expected_sha256") != expected
            or producer_row.get("observed_sha256") != expected
            or not _truth(producer_row.get("hash_match"))
            or producer_row.get("semantic_rows_verified")
            != expected_semantic_rows
            or validator_row.get("expected_sha256") != expected
            or validator_row.get("observed_sha256") != expected
            or not _truth(validator_row.get("hash_match"))
        ):
            raise ValueError(f"normalization sidecar raw hash mismatch: {month}")
    row_sources = set(
        zip(repairs["report_month"], repairs["source_sha256"], strict=False)
    )
    if row_sources != set(EXPECTED_SOURCE_HASHES.items()):
        raise ValueError("normalization sidecar repair/raw hash lineage mismatch")


def _validate_no_invention(repairs: pd.DataFrame) -> None:
    required = {
        "raw_code",
        "normalized_local_code",
        "event_date",
        "halt_start_printed",
        "halt_end_printed",
        "repaired_halt_start_printed",
        "repaired_halt_end_printed",
        "repaired_halt_start_jst",
        "repaired_halt_end_jst",
        "normalization_disposition",
        "in_research_scope_after",
        "local_code_invented",
    }
    if not required.issubset(repairs.columns):
        raise ValueError("normalization sidecar repair schema incomplete")
    if (
        repairs["local_code_invented"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .ne("false")
        .any()
    ):
        raise ValueError("normalization sidecar invented a local code")
    normalized = repairs["normalization_disposition"].eq(NORMALIZED_DISPOSITION)
    evidence = ~normalized
    normalized_rows = repairs.loc[normalized]
    evidence_rows = repairs.loc[evidence]
    repaired = normalized_rows.iloc[0] if len(normalized_rows) == 1 else None
    if (
        repaired is None
        or repaired["raw_code"] != "4585"
        or repaired["normalized_local_code"] != "4585"
        or repaired["event_date"] != "2013-09-30"
        or repaired["halt_start_printed"] != ""
        or repaired["halt_end_printed"] != ""
        or repaired["repaired_halt_start_printed"] != "17:10"
        or repaired["repaired_halt_end_printed"] != "17:25"
        or repaired["repaired_halt_start_jst"]
        != "2013-09-30T17:10:00+09:00"
        or repaired["repaired_halt_end_jst"] != "2013-09-30T17:25:00+09:00"
        or not _truth(repaired["in_research_scope_after"])
        or evidence_rows["normalized_local_code"].ne("").any()
        or evidence_rows["in_research_scope_after"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .ne("false")
        .any()
        or evidence_rows[
            [
                "repaired_halt_start_printed",
                "repaired_halt_end_printed",
                "repaired_halt_start_jst",
                "repaired_halt_end_jst",
            ]
        ]
        .ne("")
        .any()
        .any()
    ):
        raise ValueError("normalization sidecar code/time non-invention contract failed")


def _audit_packet(
    root: Path, required_files: set[str], label: str
) -> tuple[dict[str, Any], dict[str, str]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} packet root is unsafe")
    _assert_regular_file(root / "manifest.json", root, label)
    _assert_regular_file(root / "artifact_hashes.json", root, label)
    manifest = _read_json(root / "manifest.json")
    inventory_path = root / "artifact_hashes.json"
    inventory = _read_json(inventory_path)
    if not isinstance(inventory, dict):
        raise ValueError(f"{label} inventory must be an object")
    if _sha256(inventory_path) != manifest.get("artifact_hashes_sha256"):
        raise ValueError(f"{label} inventory digest mismatch")
    for relative_name, expected in inventory.items():
        relative = _safe_relative_path(relative_name, label)
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            raise ValueError(f"{label} declared hash is invalid: {relative_name}")
        path = root / relative
        _assert_regular_file(path, root, label)
        if _sha256(path) != expected:
            raise ValueError(f"{label} declared file hash mismatch: {relative_name}")
    if set(inventory) != required_files:
        raise ValueError(f"{label} declared file denominator mismatch")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} packet contains a symlink")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != required_files | {"manifest.json", "artifact_hashes.json"}:
        raise ValueError(f"{label} actual file denominator mismatch")
    return manifest, inventory


def _copy_packet(
    source: Path,
    destination: Path,
    inventory: dict[str, str],
    *,
    writer: Callable[[Path, bytes], None],
) -> list[Path]:
    copied: list[Path] = []
    for relative_name in sorted(inventory):
        relative = _safe_relative_path(relative_name, "copy")
        target = destination / relative
        _write_checked(target, (source / relative).read_bytes(), writer)
        copied.append(target)
    for name in ("manifest.json", "artifact_hashes.json"):
        target = destination / name
        _write_checked(target, (source / name).read_bytes(), writer)
        copied.append(target)
    return copied


def _input_root(value: str | Path, label: str) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_symlink():
        raise ValueError(f"{label} root must not be a symlink")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} root is unavailable") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} root is not a directory")
    return resolved


def _destination_root(destination: str | Path, *sources: Path) -> Path:
    supplied = Path(destination).expanduser()
    if supplied.is_symlink():
        raise ValueError("normalization sidecar destination must not be a symlink")
    target = supplied.resolve(strict=False)
    for source in sources:
        if target == source or target.is_relative_to(source):
            raise ValueError("normalization sidecar destination overlaps source packet")
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise ValueError("normalization sidecar destination must be absent or empty")
    return target


def _assert_regular_file(path: Path, root: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes packet") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} path contains a symlink: {relative.as_posix()}")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"{label} file is unavailable: {relative.as_posix()}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} path is not a regular file: {relative.as_posix()}")


def _safe_relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} inventory path is invalid")
    path = Path(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{label} inventory path is unsafe: {value}")
    return path


def _exact_record_ids(frame: pd.DataFrame, label: str) -> set[str]:
    if "source_record_sha256" not in frame or len(frame) != 10:
        raise ValueError(f"normalization sidecar {label} row count is not 10")
    identifiers = frame["source_record_sha256"]
    if identifiers.duplicated().any() or not identifiers.map(
        lambda value: bool(_SHA256_RE.fullmatch(str(value)))
    ).all():
        raise ValueError(f"normalization sidecar {label} record IDs are invalid")
    return set(identifiers)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"normalization sidecar CSV is unreadable: {path.name}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=lambda pairs: _strict_object(pairs, path.name),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"normalization sidecar JSON is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"normalization sidecar JSON must be an object: {path.name}")
    return value


def _strict_object(pairs: list[tuple[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"normalization sidecar JSON has duplicate key: {label}:{key}"
            )
        result[key] = value
    return result


def _write_checked(
    path: Path, body: bytes, writer: Callable[[Path, bytes], None]
) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"normalization sidecar copy target already exists: {path}")
    writer(path, body)
    if path.is_symlink():
        raise ValueError(f"normalization sidecar writer created a symlink: {path}")
    try:
        regular = stat.S_ISREG(path.lstat().st_mode)
    except OSError as exc:
        raise ValueError(f"normalization sidecar writer omitted file: {path}") from exc
    if not regular or _sha256(path) != hashlib.sha256(body).hexdigest():
        raise ValueError(f"normalization sidecar copied file mismatch: {path}")


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() == "true"


def _truths(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().eq("true")


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
