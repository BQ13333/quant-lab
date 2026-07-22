from __future__ import annotations

import json

import pandas as pd

from quant_lab.data.pit_data_plan import build_pit_data_acquisition_plan


def _write_fixture_artifact(tmp_path):
    artifact = tmp_path / "l2"
    artifact.mkdir()
    tiers = pd.DataFrame(
        [
            {
                "ticker": "AAA.KL",
                "sid": "Malaysia:AAA",
                "local_symbol": "AAA",
                "name": "AAA",
                "country": "Malaysia",
                "exchange": "Bursa",
                "currency": "MYR",
                "u2_backtest_eligible": True,
                "u3_liquid_proxy": True,
                "adv_60_country_rank": 1,
            },
            {
                "ticker": "BBB.SI",
                "sid": "Singapore:BBB",
                "local_symbol": "BBB",
                "name": "BBB",
                "country": "Singapore",
                "exchange": "SGX",
                "currency": "SGD",
                "u2_backtest_eligible": True,
                "u3_liquid_proxy": False,
                "adv_60_country_rank": 2,
            },
            {
                "ticker": "CCC.PS",
                "sid": "Philippines:CCC",
                "local_symbol": "CCC",
                "name": "CCC",
                "country": "Philippines",
                "exchange": "PSE",
                "currency": "PHP",
                "u2_backtest_eligible": True,
                "u3_liquid_proxy": True,
                "adv_60_country_rank": 1,
            },
        ]
    )
    tiers.to_csv(artifact / "eligible_universe_tiers.csv", index=False)

    symbol_map = tmp_path / "symbol_map.csv"
    pd.DataFrame(
        [
            {"ticker": "AAA", "country": "Malaysia", "refinitiv_ric": "AAA.KL", "refinitiv_symbol_status": "validated_price_history", "validated_close_field": "TRDPRC_1"},
            {"ticker": "BBB", "country": "Singapore", "refinitiv_ric": "BBB.SI", "refinitiv_symbol_status": "validated_price_history", "validated_close_field": "TRDPRC_1"},
            {"ticker": "CCC.PS", "country": "Philippines", "refinitiv_ric": "CCC.PS", "refinitiv_symbol_status": "validated_price_history", "validated_close_field": "TRDPRC_1"},
        ]
    ).to_csv(symbol_map, index=False)
    (artifact / "manifest.json").write_text(json.dumps({"symbol_map_path": str(symbol_map)}), encoding="utf-8")
    return artifact, symbol_map


def test_build_pit_data_acquisition_plan_filters_active_countries_and_writes_artifacts(tmp_path):
    artifact, symbol_map = _write_fixture_artifact(tmp_path)

    result = build_pit_data_acquisition_plan(
        l2_artifact_dir=artifact,
        artifact_root=tmp_path / "plans",
        run_id="PIT-TEST",
        symbol_map_path=symbol_map,
        active_countries=("Malaysia", "Singapore", "Indonesia"),
        tiers=("u3", "u2"),
        start_date="2025-01-01",
        end_date="2025-12-31",
        batch_size=1,
        lseg_connector_status="unauthorized",
    )

    assert set(result.selected_universe["country"]) == {"Malaysia", "Singapore"}
    assert "Philippines" not in set(result.selected_universe["country"])
    assert int(result.selected_universe["selected_for_u2"].sum()) == 2
    assert int(result.selected_universe["selected_for_u3"].sum()) == 1
    assert int(result.selected_universe["has_refinitiv_ric"].sum()) == 2

    required_datasets = {
        "security_master_pit",
        "corporate_actions",
        "delistings_listing_history",
        "fundamentals_pit",
        "ibes_estimates_revisions",
        "earnings_events_actuals",
        "liquidity_market_cap_free_float",
        "fx_country_risk",
    }
    assert required_datasets.issubset(set(result.dataset_requirements["dataset_id"]))
    assert required_datasets.issubset(set(result.readiness_gates["dataset_id"]))
    assert required_datasets.issubset(set(result.field_probe_queue["dataset_id"]))
    assert not result.extraction_batches.empty
    assert result.artifact_dir.joinpath("selected_universe.csv").exists()
    assert result.artifact_dir.joinpath("extraction_batches.csv").exists()
    assert result.manifest_path.exists()
    assert result.summary_path.read_text(encoding="utf-8").count("PIT") >= 1


def test_pit_data_plan_marks_missing_rics_as_blocked(tmp_path):
    artifact = tmp_path / "l2"
    artifact.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "AAA.KL",
                "sid": "Malaysia:AAA",
                "country": "Malaysia",
                "u2_backtest_eligible": True,
                "u3_liquid_proxy": True,
            }
        ]
    ).to_csv(artifact / "eligible_universe_tiers.csv", index=False)

    result = build_pit_data_acquisition_plan(
        l2_artifact_dir=artifact,
        artifact_root=tmp_path / "plans",
        run_id="PIT-MISSING-RIC",
        tiers=("u3",),
        start_date="2025-01-01",
        end_date="2025-01-31",
    )

    blocked = result.extraction_batches[result.extraction_batches["status"].eq("blocked_missing_refinitiv_ric")]
    assert not blocked.empty
    assert "security_master_pit" in set(blocked["dataset_id"])
