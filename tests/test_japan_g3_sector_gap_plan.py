from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_lab.data.japan_g3_sector_gap_plan import build_japan_g3_sector_gap_plan


def test_sector_gap_plan_reconciles_full_denominator(tmp_path):
    plan = tmp_path / "plan"
    sector = tmp_path / "sector"
    plan.mkdir()
    sector.mkdir()
    pd.DataFrame({"refinitiv_ric": ["AAA.T", "BBB.T", "CCC.T"]}).to_csv(
        plan / "selected_universe.csv", index=False
    )
    (plan / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "scorecard_eligible": True,
                "g1_independent_validation_admitted": True,
                "point_in_time_plan": True,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"ric": ["AAA.T"]}).to_csv(sector / "sector_routing_profile.csv", index=False)
    (sector / "manifest.json").write_text("{}", encoding="utf-8")
    result = build_japan_g3_sector_gap_plan(
        pit_plan_dir=plan,
        existing_sector_dir=sector,
        artifact_root=tmp_path / "out",
        run_id="GAP",
        batch_size=1,
    )
    assert result.missing_rics["refinitiv_ric"].tolist() == ["BBB.T", "CCC.T"]
    assert len(result.plan) == 2
    assert result.manifest["denominator_reconciles"] is True
    assert result.plan["ric_count"].sum() == 2
    with pytest.raises(FileExistsError):
        build_japan_g3_sector_gap_plan(
            pit_plan_dir=plan,
            existing_sector_dir=sector,
            artifact_root=tmp_path / "out",
            run_id="GAP",
        )


def test_sector_gap_plan_rejects_profile_outside_denominator(tmp_path):
    plan = tmp_path / "plan"
    sector = tmp_path / "sector"
    plan.mkdir()
    sector.mkdir()
    pd.DataFrame({"refinitiv_ric": ["AAA.T"]}).to_csv(plan / "selected_universe.csv", index=False)
    (plan / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "scorecard_eligible": True,
                "g1_independent_validation_admitted": True,
                "point_in_time_plan": True,
            }
        ), encoding="utf-8"
    )
    pd.DataFrame({"ric": ["ZZZ.T"]}).to_csv(sector / "sector_routing_profile.csv", index=False)
    (sector / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the PIT plan"):
        build_japan_g3_sector_gap_plan(
            pit_plan_dir=plan,
            existing_sector_dir=sector,
            artifact_root=tmp_path / "out",
            run_id="BAD",
        )
