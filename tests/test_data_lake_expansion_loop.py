from __future__ import annotations

import json

import pandas as pd

from quant_lab.agents.data_lake_expansion_loop import (
    CORE_ACCEPTED_COUNTRIES,
    REMOVED_COUNTRIES,
    DataLakeExpansionConfig,
    build_acceptance_gates,
    build_extraction_queue,
    build_market_targets,
    write_data_lake_expansion_artifacts,
)


def test_market_targets_keep_accepted_core_and_remove_untradable_markets():
    targets = build_market_targets(DataLakeExpansionConfig(run_id="TEST"))

    assert set(CORE_ACCEPTED_COUNTRIES).issubset(set(targets["country"]))
    assert not set(REMOVED_COUNTRIES).intersection(set(targets["country"]))
    assert set(targets.query("stage == 'core_accepted'")["status"]) == {"accepted_5_of_5_baseline"}
    assert {"Hong Kong", "South Korea", "Japan", "United States"}.issubset(
        set(targets["country"])
    )
    assert {"China", "Taiwan"}.isdisjoint(set(targets["country"]))


def test_thailand_is_parked_optional_not_default():
    default_targets = build_market_targets(DataLakeExpansionConfig(run_id="TEST"))
    optional_targets = build_market_targets(DataLakeExpansionConfig(run_id="TEST", include_parked_thailand=True))

    assert "Thailand" not in set(default_targets["country"])
    thailand = optional_targets[optional_targets["country"].eq("Thailand")].iloc[0]
    assert thailand["stage"] == "parked_optional"
    assert bool(thailand["include_by_default"]) is False


def test_acceptance_gates_are_five_critical_lanes_with_ops_health():
    gates = build_acceptance_gates()

    assert gates.shape[0] == 5
    assert set(gates["severity"]) == {"critical"}
    assert gates["gate_id"].tolist() == ["G1", "G2", "G3", "G4", "G5"]
    assert gates["required_evidence"].str.contains("partition_health.csv").any()
    assert gates["required_evidence"].str.contains("compaction_manifest.csv").any()


def test_extraction_queue_is_bounded_and_blocks_alpha_promotion():
    config = DataLakeExpansionConfig(run_id="TEST")
    queue = build_extraction_queue(config, build_market_targets(config))

    assert bool(queue["bounded"].all()) is True
    assert int(queue["promotes_alpha"].sum()) == 0
    assert queue["command"].str.contains("IBKR", case=False).sum() == 0
    assert queue["command"].str.contains("Vietnam|Philippines", case=False).sum() == 0
    assert queue["command"].str.contains("China|Taiwan", case=False).sum() == 0
    assert queue["phase"].str.contains("partition_health_and_compaction").any()
    assert queue.loc[queue["phase"].eq("refinitiv_20y_smoke"), "command"].str.contains("--max-plan-rows 1").all()
    assert queue.loc[queue["phase"].eq("refinitiv_20y_plan_only"), "command"].str.contains("--plan-only").all()


def test_writer_outputs_machine_readable_loop_artifacts(tmp_path):
    config = DataLakeExpansionConfig(run_id="TEST-RUN", artifact_root=tmp_path)
    artifacts = write_data_lake_expansion_artifacts(config)

    assert artifacts.market_targets_path.exists()
    assert artifacts.agent_roster_path.exists()
    assert artifacts.acceptance_gates_path.exists()
    assert artifacts.extraction_queue_path.exists()
    assert artifacts.commands_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.manifest_path.exists()

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    queue = pd.read_csv(artifacts.extraction_queue_path)

    assert manifest["ibkr_status"] == "blocked_not_in_scope"
    assert manifest["alpha_status"] == "blocked_until_new_markets_reach_5_of_5"
    assert manifest["bounded_by_default"] is True
    assert manifest["promotes_alpha_rows"] == 0
    assert set(manifest["removed_countries"]) == set(REMOVED_COUNTRIES)
    assert queue["phase"].tolist() == artifacts.extraction_queue["phase"].tolist()
