from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g2_vendor_plan import build_japan_g2_vendor_plan
from quant_lab.data.refinitiv_pit_extraction import load_pit_plan_artifacts


def test_vendor_plan_deduplicates_reference_rics_and_builds_both_event_families(tmp_path: Path) -> None:
    active_path = tmp_path / "active.csv"
    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "refinitiv_ric": ["AAA.T"],
            "country": ["Japan"],
            "exchange": ["JPX"],
            "local_code": ["1111"],
            "listing_start": ["2020-01-01"],
            "currency": ["JPY"],
            "pit_usable": [True],
        }
    ).to_csv(active_path, index=False)
    pd.DataFrame(
        {
            "refinitiv_ric": ["AAA.T", "AAA.T", "BBB.T"],
            "country": ["Japan"] * 3,
            "exchange": ["JPX"] * 3,
            "local_code": ["1111", "1111", "2222"],
            "listing_start": ["2010-01-01", "2020-01-01", "2015-01-01"],
            "currency": ["JPY"] * 3,
            "pit_usable": [True] * 3,
        }
    ).to_csv(reference_path, index=False)

    result = build_japan_g2_vendor_plan(
        active_symbol_map_path=active_path,
        reference_symbol_map_path=reference_path,
        artifact_root=tmp_path / "artifacts",
        run_id="PLAN",
        batch_size=1,
    )

    selected = pd.read_csv(result.selected_universe_path)
    actions = pd.read_csv(result.corporate_action_plan_path)
    probes = pd.read_csv(result.liquidity_probe_queue_path)
    assert selected["refinitiv_ric"].tolist() == ["AAA.T", "BBB.T"]
    assert len(actions) == 4
    assert set(actions["event_family"]) == {"dividend", "capital_change"}
    assert len(probes) == 6
    loaded = load_pit_plan_artifacts(result.artifact_dir)
    assert len(loaded.selected_universe) == 2
    assert set(loaded.field_probe_queue["field_role"]) == {
        "market_cap",
        "free_float",
        "shares_outstanding",
    }
    assert actions["run_id"].eq("REFINITIV-JPN-G2-CORPACT-20060101-20260706-V1").all()
    assert result.manifest["promotion_eligible"] is False
    for filename, digest in result.manifest["output_sha256"].items():
        assert hashlib.sha256((result.artifact_dir / filename).read_bytes()).hexdigest() == digest

    with pytest.raises(FileExistsError, match="immutable"):
        build_japan_g2_vendor_plan(
            active_symbol_map_path=active_path,
            reference_symbol_map_path=reference_path,
            artifact_root=tmp_path / "artifacts",
            run_id="PLAN",
        )
