from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_lab.data.japan_reference_only_map import build_japan_reference_only_map


def _map_dir(root, role, rows):
    directory = root / role
    directory.mkdir()
    pd.DataFrame(rows).to_csv(directory / "symbol_map.csv", index=False)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "validation_status": "pass",
                "evidence_scope": "full",
                "scorecard_eligible": True,
                "source_checksums_verified": True,
                "map_role": role,
                "unresolved_rows": 0,
                "overlap_conflict_rows": 0,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_reference_only_map_exactly_partitions_full_reference(tmp_path):
    active_rows = [
        {"stable_listing_id": "a", "refinitiv_ric": "AAA.T", "listing_start": "2020-01-01", "listing_end": "2026-01-01"}
    ]
    reference_rows = [
        *active_rows,
        {"stable_listing_id": "b", "refinitiv_ric": "BBB.T", "listing_start": "2010-01-01", "listing_end": "2019-01-01"},
    ]
    active = _map_dir(tmp_path, "active", active_rows)
    reference = _map_dir(tmp_path, "reference", reference_rows)
    result = build_japan_reference_only_map(
        active_map_dir=active,
        full_reference_map_dir=reference,
        artifact_root=tmp_path / "out",
        run_id="REFERENCE-ONLY",
    )
    assert result.symbol_map["refinitiv_ric"].tolist() == ["BBB.T"]
    assert result.manifest["partition_reconstructs_full_reference"] is True
    assert result.manifest["active_reference_window_overlap_rows"] == 0
    assert result.manifest["active_reference_interval_overlap_rows"] == 0
    assert result.manifest["scorecard_eligible"] is True
    assert result.manifest["independent_audit"] is False
    assert result.manifest["derived_from_independently_validated_maps"] is True
    with pytest.raises(FileExistsError):
        build_japan_reference_only_map(
            active_map_dir=active,
            full_reference_map_dir=reference,
            artifact_root=tmp_path / "out",
            run_id="REFERENCE-ONLY",
        )


def test_reference_only_map_rejects_active_window_missing_from_reference(tmp_path):
    active = _map_dir(
        tmp_path,
        "active",
        [{"refinitiv_ric": "AAA.T", "listing_start": "2020-01-01", "listing_end": "2026-01-01"}],
    )
    reference = _map_dir(
        tmp_path,
        "reference",
        [{"refinitiv_ric": "BBB.T", "listing_start": "2010-01-01", "listing_end": "2019-01-01"}],
    )
    with pytest.raises(ValueError, match="absent from full reference"):
        build_japan_reference_only_map(
            active_map_dir=active,
            full_reference_map_dir=reference,
            artifact_root=tmp_path / "out",
            run_id="BAD",
        )


def test_reference_only_map_rejects_nonidentical_overlapping_interval(tmp_path):
    active_rows = [
        {"refinitiv_ric": "AAA.T", "listing_start": "2020-01-01", "listing_end": "2026-01-01"}
    ]
    reference_rows = [
        *active_rows,
        {"refinitiv_ric": "AAA.T", "listing_start": "2019-01-01", "listing_end": "2021-01-01"},
    ]
    active = _map_dir(tmp_path, "active", active_rows)
    reference = _map_dir(tmp_path, "reference", reference_rows)
    with pytest.raises(ValueError, match="overlapping RIC/date intervals"):
        build_japan_reference_only_map(
            active_map_dir=active,
            full_reference_map_dir=reference,
            artifact_root=tmp_path / "out-overlap",
            run_id="OVERLAP",
        )
