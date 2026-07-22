from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_halt_history_validator import (
    _validate_lifecycle_packet_copy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COPIED_PACKETS = (
    (
        PROJECT_ROOT
        / "artifacts/japan_g4_tse_wayback_lifecycle"
        / "JPN-G4-TSE-WAYBACK-LIFECYCLE-2014-09-12-COPIED-20260712-V1",
        PROJECT_ROOT
        / "artifacts/japan_g4_tse_wayback_lifecycle_validation"
        / (
            "JPN-G4-TSE-WAYBACK-LIFECYCLE-2014-09-12-COPIED-"
            "INDEPENDENT-VALIDATION-20260712-V1"
        ),
        4,
        8,
        76,
    ),
    (
        PROJECT_ROOT
        / "artifacts/japan_g4_tse_wayback_lifecycle"
        / "JPN-G4-TSE-WAYBACK-LIFECYCLE-2015-01-12-COPIED-20260712-V1",
        PROJECT_ROOT
        / "artifacts/japan_g4_tse_wayback_lifecycle_validation"
        / (
            "JPN-G4-TSE-WAYBACK-LIFECYCLE-2015-01-12-COPIED-"
            "INDEPENDENT-VALIDATION-20260712-V1"
        ),
        12,
        24,
        256,
    ),
)

pytestmark = pytest.mark.skipif(
    not all(
        producer.is_dir() and validation.is_dir()
        for producer, validation, *_ in COPIED_PACKETS
    ),
    reason="private JPX artifact integration tests",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_packet(root: Path) -> None:
    manifest_path = root / "manifest.json"
    inventory_path = root / "artifact_hashes.json"
    inventory = {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "artifact_hashes.json"}
    }
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes_sha256"] = _sha(inventory_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _copied_packet(
    tmp_path: Path,
    producer: Path,
    validation: Path,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "packet-0001"
    shutil.copytree(producer, root / "producer")
    shutil.copytree(validation, root / "validation")
    lineage = _write_lineage(root)
    return root, lineage


def _write_lineage(root: Path) -> dict[str, object]:
    producer = root / "producer"
    validation = root / "validation"
    producer_manifest = json.loads(
        (producer / "manifest.json").read_text(encoding="utf-8")
    )
    validation_manifest = json.loads(
        (validation / "manifest.json").read_text(encoding="utf-8")
    )
    events = pd.read_csv(producer / "exact_events.csv", dtype=str).fillna("")
    packet_lineage: dict[str, object] = {
        "producer_run_id": producer_manifest["run_id"],
        "producer_manifest_sha256": _sha(producer / "manifest.json"),
        "validation_run_id": validation_manifest["run_id"],
        "validation_manifest_sha256": _sha(validation / "manifest.json"),
        "selected_months": producer_manifest["selected_months"],
        "selected_month_count": producer_manifest["selected_month_count"],
        "covered_claims": producer_manifest["exact_nonempty_claim_count"],
        "source_rows": producer_manifest["exact_event_count"],
        "common_equity_rows": int(events["instrument_class"].eq("equity").sum()),
        "preserved_non_equity_rows": int(
            events["instrument_class"].ne("equity").sum()
        ),
        "absence_is_zero": False,
        "official_source_only": True,
        "wayback_transport_only": False,
        "official_copied_file_transport": True,
    }
    (root / "lineage.json").write_text(
        json.dumps(packet_lineage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "packet_id": root.name,
        "packet_path": f"inputs/tse_wayback_lifecycle/packets/{root.name}",
        **packet_lineage,
    }


def _reseal_pair(root: Path) -> dict[str, object]:
    producer = root / "producer"
    validation = root / "validation"
    _seal_packet(producer)
    validation_manifest_path = validation / "manifest.json"
    validation_manifest = json.loads(
        validation_manifest_path.read_text(encoding="utf-8")
    )
    validation_manifest["producer_manifest_sha256"] = _sha(
        producer / "manifest.json"
    )
    validation_manifest_path.write_text(
        json.dumps(validation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _seal_packet(validation)
    return _write_lineage(root)


@pytest.mark.parametrize(
    ("producer", "validation", "month_count", "claim_count", "event_count"),
    COPIED_PACKETS,
)
def test_sealed_official_jpx_copied_packets_reparse_exactly(
    tmp_path: Path,
    producer: Path,
    validation: Path,
    month_count: int,
    claim_count: int,
    event_count: int,
) -> None:
    root, lineage = _copied_packet(tmp_path, producer, validation)
    issues: list[str] = []

    claims, months, events = _validate_lifecycle_packet_copy(
        root, lineage, root.name, issues
    )

    assert issues == []
    assert len(months) == month_count
    assert len(claims) == claim_count
    assert len(events) == event_count


def test_copied_packet_rejects_resealed_unapproved_pdf_route(tmp_path: Path) -> None:
    producer, validation, *_ = COPIED_PACKETS[1]
    root, _ = _copied_packet(tmp_path, producer, validation)
    inventory_path = root / "producer/source_inventory.csv"
    inventory = pd.read_csv(inventory_path, dtype=str).fillna("")
    inventory.loc[0, "official_pdf_url"] = "https://example.com/17_idou1501.pdf"
    inventory.to_csv(inventory_path, index=False)
    lineage = _reseal_pair(root)
    issues: list[str] = []

    _validate_lifecycle_packet_copy(root, lineage, root.name, issues)

    assert any("source_lineage_invalid" in issue for issue in issues)
    assert any("source_claim_mismatch" in issue for issue in issues)


def test_copied_packet_rejects_resealed_inferred_zero_claim(tmp_path: Path) -> None:
    producer, validation, *_ = COPIED_PACKETS[0]
    root, _ = _copied_packet(tmp_path, producer, validation)
    claims_path = root / "producer/exact_nonempty_claims.csv"
    claims = pd.read_csv(claims_path, dtype=str).fillna("")
    claims.loc[0, "zero_event_asserted"] = "True"
    claims.to_csv(claims_path, index=False)
    lineage = _reseal_pair(root)
    issues: list[str] = []

    _validate_lifecycle_packet_copy(root, lineage, root.name, issues)

    assert any("exact_denominator_invalid" in issue for issue in issues)
    assert any("source_claim_mismatch" in issue for issue in issues)


def test_copied_packet_rejects_resealed_event_set_not_in_pdf(tmp_path: Path) -> None:
    producer, validation, *_ = COPIED_PACKETS[0]
    root, _ = _copied_packet(tmp_path, producer, validation)
    producer_csv = root / "producer/exact_events.csv"
    producer_parquet = root / "producer/exact_events.parquet"
    independent_csv = root / "validation/independent_exact_events.csv"
    events = pd.read_csv(producer_csv, dtype=str).fillna("")
    events.loc[0, "issue_name"] = "Forged issue name"
    events.to_csv(producer_csv, index=False)
    events.to_parquet(producer_parquet, index=False)
    independent = pd.read_csv(independent_csv, dtype=str).fillna("")
    independent.loc[0, "issue_name"] = "Forged issue name"
    independent.to_csv(independent_csv, index=False)
    lineage = _reseal_pair(root)
    issues: list[str] = []

    _validate_lifecycle_packet_copy(root, lineage, root.name, issues)

    assert any("independent_event_set_mismatch" in issue for issue in issues)
