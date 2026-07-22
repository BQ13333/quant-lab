from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from quant_lab.data.japan_g4_halt_normalization_sidecar_admission import (
    PRODUCER_RUN_ID,
    VALIDATION_RUN_ID,
    copy_validated_halt_normalization_sidecar,
)
from quant_lab.data.japan_g4_halt_report_normalization_repair import (
    AGGREGATE_DISPOSITION,
    APPENDIX_DISPOSITION,
    NORMALIZED_DISPOSITION,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEALED_PRODUCER = (
    PROJECT_ROOT
    / "artifacts/japan_g4_halt_report_normalization_repair"
    / PRODUCER_RUN_ID
)
SEALED_VALIDATION = (
    PROJECT_ROOT
    / "artifacts/japan_g4_halt_report_normalization_repair_validation"
    / VALIDATION_RUN_ID
)

pytestmark = pytest.mark.skipif(
    not (SEALED_PRODUCER.is_dir() and SEALED_VALIDATION.is_dir()),
    reason="private JPX artifact integration tests",
)


def _packets(tmp_path: Path) -> tuple[Path, Path]:
    producer = tmp_path / "producer"
    validation = tmp_path / "validation"
    shutil.copytree(SEALED_PRODUCER, producer)
    shutil.copytree(SEALED_VALIDATION, validation)
    return producer, validation


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_admits_sealed_sidecar_and_keeps_instrument_separate_from_evidence(
    tmp_path: Path,
) -> None:
    result = copy_validated_halt_normalization_sidecar(
        SEALED_PRODUCER,
        SEALED_VALIDATION,
        tmp_path / "admitted",
    )

    assert len(result.normalized_repairs) == 1
    assert len(result.evidence) == 9
    repair = result.normalized_repairs.iloc[0]
    assert repair["normalization_disposition"] == NORMALIZED_DISPOSITION
    assert repair["raw_code"] == repair["normalized_local_code"] == "4585"
    assert repair["repaired_halt_start_jst"] == "2013-09-30T17:10:00+09:00"
    assert repair["repaired_halt_end_jst"] == "2013-09-30T17:25:00+09:00"
    assert result.evidence["normalization_disposition"].value_counts().to_dict() == {
        AGGREGATE_DISPOSITION: 6,
        APPENDIX_DISPOSITION: 3,
    }
    assert result.evidence["normalized_local_code"].eq("").all()
    assert result.evidence[
        ["repaired_halt_start_jst", "repaired_halt_end_jst"]
    ].eq("").all().all()
    assert result.lineage["repair_rows_reconciled"] == 10
    assert result.lineage["refinitiv_used"] is False
    assert (tmp_path / "admitted/producer/normalization_repairs.csv").is_file()
    assert (
        tmp_path / "admitted/validation/independent_semantic_validation.csv"
    ).is_file()
    assert (tmp_path / "admitted/lineage.json").is_file()


def test_rejects_declared_file_byte_tampering(tmp_path: Path) -> None:
    producer, validation = _packets(tmp_path)
    repairs = producer / "normalization_repairs.csv"
    repairs.write_bytes(repairs.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="declared file hash mismatch"):
        copy_validated_halt_normalization_sidecar(
            producer, validation, tmp_path / "admitted"
        )


def test_rejects_mixed_producer_validator_lineage(tmp_path: Path) -> None:
    producer, validation = _packets(tmp_path)
    manifest_path = validation / "manifest.json"
    manifest = _json(manifest_path)
    manifest["producer_manifest_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="lineage mismatch"):
        copy_validated_halt_normalization_sidecar(
            producer, validation, tmp_path / "admitted"
        )


def test_rejects_resealed_repair_count_drift(tmp_path: Path) -> None:
    producer, validation = _packets(tmp_path)
    repairs_path = producer / "normalization_repairs.csv"
    lines = repairs_path.read_text(encoding="utf-8").splitlines()
    repairs_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    inventory_path = producer / "artifact_hashes.json"
    inventory = _json(inventory_path)
    inventory["normalization_repairs.csv"] = _sha256(repairs_path)
    _write_json(inventory_path, inventory)
    producer_manifest_path = producer / "manifest.json"
    producer_manifest = _json(producer_manifest_path)
    producer_manifest["artifact_hashes_sha256"] = _sha256(inventory_path)
    _write_json(producer_manifest_path, producer_manifest)
    validation_manifest_path = validation / "manifest.json"
    validation_manifest = _json(validation_manifest_path)
    validation_manifest["producer_manifest_sha256"] = _sha256(
        producer_manifest_path
    )
    _write_json(validation_manifest_path, validation_manifest)

    with pytest.raises(ValueError, match="verifier rejected|row count"):
        copy_validated_halt_normalization_sidecar(
            producer, validation, tmp_path / "admitted"
        )


def test_rejects_unsafe_inventory_path(tmp_path: Path) -> None:
    producer, validation = _packets(tmp_path)
    inventory_path = producer / "artifact_hashes.json"
    inventory = _json(inventory_path)
    digest = inventory.pop("normalization_repairs.csv")
    inventory["../normalization_repairs.csv"] = digest
    _write_json(inventory_path, inventory)
    manifest_path = producer / "manifest.json"
    manifest = _json(manifest_path)
    manifest["artifact_hashes_sha256"] = _sha256(inventory_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="inventory path is unsafe"):
        copy_validated_halt_normalization_sidecar(
            producer, validation, tmp_path / "admitted"
        )


def test_rejects_symlinked_declared_file(tmp_path: Path) -> None:
    producer, validation = _packets(tmp_path)
    raw_path = producer / "raw/2013-09.pdf"
    preserved = tmp_path / "preserved.pdf"
    shutil.copyfile(raw_path, preserved)
    raw_path.unlink()
    raw_path.symlink_to(preserved)

    with pytest.raises(ValueError, match="symlink"):
        copy_validated_halt_normalization_sidecar(
            producer, validation, tmp_path / "admitted"
        )
