from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g3_sector_snapshot import build_sector_routing_profile, finalize_sector_snapshot_artifact


def _snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "requested_ric": ["7203.T", "8306.T", "8766.T"],
            "TRBC Industry Group Code": [531010, 551010, 553010],
            "TRBC Industry Group Name": ["Automobiles", "Banking Services", "Insurance"],
            "TRBC Business Sector Code": [5310, 5510, 5530],
            "TRBC Business Sector Name": ["Automobiles", "Banking", "Insurance"],
            "TRBC Industry Code": [53101010, 55101010, 55301020],
            "TRBC Industry Name": ["Automakers", "Banks", "P&C Insurance"],
        }
    )


def test_sector_routes_are_semantically_distinct() -> None:
    profile = build_sector_routing_profile(_snapshot()).set_index("ric")
    assert profile.loc["7203.T", "fundamental_route_class"] == "general"
    assert profile.loc["8306.T", "revenue_source_field"] == "TR.BankTotalRevenue"
    assert profile.loc["8306.T", "profitability_source_field"] == "TR.NetInterestIncome"
    assert profile.loc["8766.T", "profitability_source_field"] == "TR.OperatingIncome"
    assert not profile["point_in_time_sector_claimed"].any()


def test_current_sector_snapshot_never_promotes(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    path = finalize_sector_snapshot_artifact(tmp_path, _snapshot(), requested_rics=3, run_id="SECTOR")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_current_routing"
    assert manifest["coverage"] == 1.0
    assert manifest["scorecard_eligible"] is False
    assert manifest["point_in_time_sector_claimed"] is False
