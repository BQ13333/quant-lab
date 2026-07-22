from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g3_execution_plan import (
    build_japan_g3_execution_plans,
    write_japan_g3_execution_plan_artifacts,
)


def test_plans_are_route_aware_and_blocked_before_g1() -> None:
    universe = pd.DataFrame(
        {"refinitiv_ric": ["A.T", "B.T", "C.T"], "selected_for_u2": [True, True, True]}
    )
    sectors = pd.DataFrame(
        {"ric": ["A.T", "B.T", "C.T"], "fundamental_route_class": ["general", "bank", "insurance"]}
    )
    fundamentals, consensus, earnings, analyst = build_japan_g3_execution_plans(
        universe, sectors, "2025-01-01", "2025-12-31", batch_size=2, window_years=1, g1_ok=False
    )
    assert set(fundamentals["route_class"]) == {"general", "bank", "insurance"}
    assert fundamentals.loc[fundamentals["route_class"].eq("bank"), "source_fields"].str.contains("TR.BankTotalRevenue").all()
    assert set(fundamentals["status"]) == {"blocked_upstream_g1"}
    assert set(consensus["status"]) == {"blocked_upstream_g1"}
    assert set(earnings["status"]) == {"blocked_upstream_g1"}
    assert set(analyst["status"]) == {"blocked_upstream_g1"}


def test_writer_hashes_plans_and_keeps_analyst_lane_fail_closed(tmp_path: Path) -> None:
    plan = tmp_path / "plan"
    sector = tmp_path / "sector"
    routes = tmp_path / "routes"
    g1 = tmp_path / "g1"
    for directory in (plan, sector, routes, g1):
        directory.mkdir()
    pd.DataFrame(
        {"refinitiv_ric": ["A.T"], "selected_for_u2": [True]}
    ).to_csv(plan / "selected_universe.csv", index=False)
    (plan / "manifest.json").write_text(json.dumps({"selected_names": 1}), encoding="utf-8")
    pd.DataFrame(
        {"ric": ["A.T"], "fundamental_route_class": ["general"]}
    ).to_csv(sector / "sector_routing_profile.csv", index=False)
    (sector / "manifest.json").write_text(json.dumps({"coverage": 1.0}), encoding="utf-8")
    pd.DataFrame({"dataset_id": ["fundamentals_pit"]}).to_csv(
        routes / "accepted_route_registry.csv", index=False
    )
    (routes / "manifest.json").write_text(json.dumps({"unresolved_roles": 0}), encoding="utf-8")
    (g1 / "manifest.json").write_text(json.dumps({"status": "failed_validation"}), encoding="utf-8")
    result = write_japan_g3_execution_plan_artifacts(
        plan, sector, routes, g1, tmp_path / "output", "EXEC", "2025-01-01", "2025-12-31", 1, 1
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked_upstream"
    assert manifest["scorecard_eligible"] is False
    assert manifest["full_workspace_analyst_detail_auto_start"] is False
    assert manifest["ordinary_workspace_plan_rows"] > 0
    assert manifest["ordinary_workspace_estimated_field_calls"] > manifest["ordinary_workspace_plan_rows"]
    assert manifest["artifact_sha256"]["analyst_detail_plan"]
