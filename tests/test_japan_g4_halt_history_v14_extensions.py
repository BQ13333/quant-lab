from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_halt_history import (
    _apply_lifecycle_taxonomy_repairs,
    _copy_validated_lifecycle_taxonomy_repair,
    _copy_validated_official_timestamp_v14_extension,
    _validate_official_timestamp_v14_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PRODUCER = PROJECT_ROOT / (
    "artifacts/japan_g4_lifecycle_taxonomy_repair/"
    "JPN-G4-LIFECYCLE-TAXONOMY-REPAIR-V14-20260712-V1"
)
TAXONOMY_VALIDATION = PROJECT_ROOT / (
    "artifacts/japan_g4_lifecycle_taxonomy_repair_validation/"
    "JPN-G4-LIFECYCLE-TAXONOMY-REPAIR-INDEPENDENT-VALIDATION-V14-20260712-V1"
)
TIMESTAMP_PRODUCER = PROJECT_ROOT / (
    "artifacts/japan_g4_halt_timestamp_recovery/"
    "JPN-G4-HALT-OFFICIAL-PUBLICATION-TIMESTAMPS-"
    "V14-SIX-DELISTINGS-20260712-V1"
)
TIMESTAMP_VALIDATION = PROJECT_ROOT / (
    "artifacts/japan_g4_halt_timestamp_recovery_validation/"
    "JPN-G4-HALT-OFFICIAL-PUBLICATION-TIMESTAMPS-"
    "V14-SIX-DELISTINGS-INDEPENDENT-VALIDATION-20260712-V1"
)
LIFECYCLE_PACKETS = PROJECT_ROOT / (
    "artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V14/"
    "inputs/tse_wayback_lifecycle/packets"
)

pytestmark = pytest.mark.skipif(
    not all(
        path.is_dir()
        for path in (
            TAXONOMY_PRODUCER,
            TAXONOMY_VALIDATION,
            TIMESTAMP_PRODUCER,
            TIMESTAMP_VALIDATION,
            LIFECYCLE_PACKETS,
        )
    ),
    reason="private JPX artifact integration tests",
)


def test_taxonomy_sidecar_copy_and_exact_application(tmp_path: Path) -> None:
    packet = _copy_validated_lifecycle_taxonomy_repair(
        TAXONOMY_PRODUCER,
        TAXONOMY_VALIDATION,
        tmp_path / "taxonomy",
    )
    source = pd.concat(
        [
            pd.read_csv(
                LIFECYCLE_PACKETS / name / "producer" / "exact_events.csv",
                dtype=str,
            ).fillna("")
            for name in ("packet-0004", "packet-0005")
        ],
        ignore_index=True,
    )
    repaired = _apply_lifecycle_taxonomy_repairs(source, packet.repairs)
    selected = repaired.loc[repaired["event_id"].isin(packet.repairs["event_id"])]
    assert len(selected) == 9
    assert selected["instrument_class"].value_counts().to_dict() == {
        "warrant": 5,
        "foreign_share": 4,
    }
    assert len(repaired) == len(source)


def test_v14_timestamp_sidecar_copy_and_contract(tmp_path: Path) -> None:
    packet = _copy_validated_official_timestamp_v14_extension(
        TIMESTAMP_PRODUCER,
        TIMESTAMP_VALIDATION,
        tmp_path / "timestamp",
    )
    accepted, producer_files, validation_files = (
        _validate_official_timestamp_v14_contract(
            tmp_path / "timestamp" / "producer",
            tmp_path / "timestamp" / "validation",
        )
    )
    assert len(packet.accepted) == len(accepted) == 6
    assert len(producer_files) == 20
    assert len(validation_files) == 4
    assert accepted["event_id"].is_unique


def test_v14_timestamp_sidecar_rejects_declared_output_tamper(
    tmp_path: Path,
) -> None:
    producer = tmp_path / "producer"
    validation = tmp_path / "validation"
    shutil.copytree(TIMESTAMP_PRODUCER, producer)
    shutil.copytree(TIMESTAMP_VALIDATION, validation)
    accepted = producer / "accepted_timestamp_recovery.csv"
    accepted.write_bytes(accepted.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="producer output invalid"):
        _validate_official_timestamp_v14_contract(producer, validation)
