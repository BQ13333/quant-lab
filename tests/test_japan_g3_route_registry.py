from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g3_route_registry import REQUIRED_ROLES, write_japan_g3_route_registry_artifacts


def test_registry_requires_all_semantic_roles_and_never_promotes_qa(tmp_path: Path) -> None:
    historical = tmp_path / "historical"
    critical = tmp_path / "critical"
    revision = tmp_path / "revision"
    fundamentals = tmp_path / "fundamentals"
    for directory in (historical, critical, revision, fundamentals):
        directory.mkdir()

    pd.DataFrame(
        [
            {"dataset_id": dataset, "field_role": role, "candidate_field": f"TR.{role}", "coverage": 1.0}
            for dataset, roles in {
                "ibes_estimates_revisions": ("eps_estimate", "revenue_estimate", "recommendation"),
                "earnings_events_actuals": ("earnings_release",),
            }.items()
            for role in roles
        ]
    ).to_csv(historical / "history_field_map.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": dataset,
                "field_role": role,
                "candidate_field": f"TR.{role}",
                "status": "historical_validated",
                "coverage": 1.0,
                "non_null_values": 10,
            }
            for dataset, roles in {
                "ibes_estimates_revisions": ("target_price",),
                "earnings_events_actuals": ("actual_eps", "surprise"),
            }.items()
            for role in roles
        ]
    ).to_csv(critical / "route_probe_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "country": "JPN",
                "status": "accepted",
                "field": "TR.Revision.CalcDate",
                "unique_date_count": 100,
            }
        ]
    ).to_csv(revision / "accepted_revision_time_map.csv", index=False)
    pd.DataFrame(
        [
            {
                "field_role": role,
                "history_rows": 10,
                "known_at_coverage": 1.0,
                "pit_timestamp_violations": 0,
            }
            for role in ("revenue", "gross_profit", "eps", "book_value")
        ]
    ).to_csv(fundamentals / "quality.csv", index=False)
    (fundamentals / "manifest.json").write_text("{}", encoding="utf-8")

    result = write_japan_g3_route_registry_artifacts(
        historical,
        critical,
        revision,
        fundamentals,
        artifact_root=tmp_path / "output",
        run_id="ROUTES",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert len(result.registry) == sum(len(roles) for roles in REQUIRED_ROLES.values())
    assert result.unresolved.empty
    assert manifest["status"] == "complete_qa_routes"
    assert manifest["scorecard_eligible"] is False
    assert manifest["period_end_as_release_date_allowed"] is False


def test_registry_prefers_self_keyed_ibes_and_parameterized_actuals(tmp_path: Path) -> None:
    historical = tmp_path / "historical"
    critical = tmp_path / "critical"
    revision = tmp_path / "revision"
    fundamentals = tmp_path / "fundamentals"
    ibes_qa = tmp_path / "ibes_qa"
    for directory in (historical, critical, revision, fundamentals, ibes_qa):
        directory.mkdir()
    pd.DataFrame(
        [
            {"dataset_id": "ibes_estimates_revisions", "field_role": role, "candidate_field": f"TR.{role}", "coverage": 1.0}
            for role in ("eps_estimate", "revenue_estimate", "recommendation")
        ]
    ).to_csv(historical / "history_field_map.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": "ibes_estimates_revisions",
                "field_role": "target_price",
                "candidate_field": "TR.PriceTargetMean",
                "status": "historical_validated",
                "coverage": 1.0,
                "non_null_values": 10,
            }
        ]
    ).to_csv(critical / "route_probe_results.csv", index=False)
    pd.DataFrame(columns=["country", "status", "field", "unique_date_count"]).to_csv(
        revision / "accepted_revision_time_map.csv", index=False
    )
    pd.DataFrame(
        [
            {"field_role": role, "history_rows": 10, "known_at_coverage": 1.0, "pit_timestamp_violations": 0}
            for role in ("revenue", "gross_profit", "eps", "book_value")
        ]
    ).to_csv(fundamentals / "quality.csv", index=False)
    (fundamentals / "manifest.json").write_text("{}", encoding="utf-8")
    (ibes_qa / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed_partial",
                "self_keyed_analyst_fields": True,
                "mixed_target_and_recommendation_query_allowed": False,
                "known_at_coverage": 1.0,
                "pit_timestamp_violations": 0,
                "revision_event_rows": 10,
                "earnings_event_rows": 10,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"dataset_id": ["ibes_estimates_revisions", "earnings_events_actuals"]}
    ).to_csv(ibes_qa / "quality.csv", index=False)

    result = write_japan_g3_route_registry_artifacts(
        historical,
        critical,
        revision,
        fundamentals,
        artifact_root=tmp_path / "output",
        run_id="ROUTES",
        ibes_events_qa_dir=ibes_qa,
    )
    registry = result.registry.set_index(["dataset_id", "field_role"])
    assert registry.loc[("ibes_estimates_revisions", "revision_time"), "source_field"] == "TR.RecEstValue.*|TR.TPEstValue.*"
    assert registry.loc[("earnings_events_actuals", "surprise"), "source_field"] == "TR.EPSActSurprise"
    assert result.unresolved.empty
