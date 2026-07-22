from __future__ import annotations

import json

import pandas as pd

from quant_lab.data.historical_backfill import parse_country_filter, refinitiv_universe_from_symbol_map
from quant_lab.data.usa_research_universe import (
    build_usa_large_liquid_core,
    build_usa_active_liquid_core,
    build_usa_listed_reference_core,
    normalize_usa_liquidity_snapshot,
    snapshot_request_from_symbol_map,
    symbol_map_from_security_master_rows,
    usa_large_liquid_snapshot_field_map,
    write_usa_large_liquid_universe_artifacts,
    write_usa_research_universe_artifacts,
)


def _row(
    ric: str,
    ticker: str,
    exchange_code: str,
    exchange_name: str,
    active_state_bucket: str,
    asset_leaf: str = "Ordinary Share",
    primary_ric: bool = True,
    primary_issue: bool = True,
) -> dict[str, object]:
    return {
        "ric": ric,
        "document_title": f"{ticker} title",
        "common_name": f"{ticker} common",
        "exchange_country": "USA",
        "exchange_name": exchange_name,
        "exchange_code": exchange_code,
        "asset_state": "AC" if active_state_bucket == "active" else "DC",
        "asset_state_name": "Active" if active_state_bucket == "active" else "De-activated",
        "rcs_asset_category_leaf": asset_leaf,
        "instrument_permid": f"PERM-{ric}",
        "ticker_symbol": ticker,
        "is_primary_ric": primary_ric,
        "is_primary_issue_ric": primary_issue,
        "active_state_bucket": active_state_bucket,
        "country_code": "USA",
        "country": "United States",
        "retire_date": "2024-01-02" if active_state_bucket != "active" else pd.NA,
        "first_trade_date": "2001-01-02",
        "isin": f"ISIN-{ticker}",
        "sedol": f"SEDOL-{ticker}",
        "organization_id": f"ORG-{ticker}",
        "perm_id": f"PERMID-{ticker}",
        "price_currency": "USD",
        "active_flag": active_state_bucket == "active",
    }


def _security_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("AAPL.O", "AAPL", "NSQ", "NASDAQ Global Select Consolidated", "active"),
            _row("MSFT.O", "MSFT", "NSQ", "NASDAQ Global Select Consolidated", "active", primary_ric=False),
            _row("IBM.N", "IBM", "NYQ", "NYSE Consolidated", "active"),
            _row("XYZ.A", "XYZ", "ASQ", "NYSE American Consolidated", "active", "Fully Paid Ordinary Share"),
            _row("PINK.PK", "PINK", "PNK", "US 'Other OTC' and Grey Market", "active"),
            _row("OLD.O", "OLD", "NAS", "NASDAQ Stock Exchange Capital Market", "inactive", primary_ric=False),
            _row("OLDNYSE.N", "OLDNYSE", "NYS", "New York Stock Exchange", "inactive", primary_ric=False),
            _row("OLDPINK.PK", "OLDPINK", "PNK", "US 'Other OTC' and Grey Market", "inactive", primary_ric=False),
            _row("ETF.O", "ETF", "NSQ", "NASDAQ Global Select Consolidated", "active", "Exchange Traded Fund"),
        ]
    )


def test_usa_active_liquid_core_keeps_primary_listed_active_common_shares_only():
    core = build_usa_active_liquid_core(_security_master())

    assert set(core["ric"]) == {"AAPL.O", "IBM.N", "XYZ.A"}
    assert set(core["exchange_code"]) == {"NSQ", "NYQ", "ASQ"}
    assert set(core["active_state_bucket"]) == {"active"}


def test_usa_listed_reference_core_retains_dead_listed_names_not_otc():
    reference = build_usa_listed_reference_core(_security_master())

    assert set(reference["ric"]) == {"AAPL.O", "IBM.N", "XYZ.A", "OLD.O", "OLDNYSE.N"}
    assert "OLDPINK.PK" not in set(reference["ric"])
    assert set(reference["active_state_bucket"]) == {"active", "inactive"}


def test_usa_symbol_map_contract_works_with_existing_historical_loader(tmp_path):
    reference = build_usa_listed_reference_core(_security_master())
    symbol_map = symbol_map_from_security_master_rows(reference, tmp_path / "security_master.csv", "USA_LISTED_REFERENCE_CORE_V1")
    path = tmp_path / "symbol_map.csv"
    symbol_map.to_csv(path, index=False)

    universe = refinitiv_universe_from_symbol_map(path, include_countries=parse_country_filter("United States"))

    assert set(universe["refinitiv_ric"]) == {"AAPL.O", "IBM.N", "XYZ.A", "OLD.O", "OLDNYSE.N"}
    assert set(universe["refinitiv_validation_status"]) == {"security_master_discovery_candidate"}
    assert set(universe["country"]) == {"United States"}


def test_usa_large_liquid_core_filters_on_current_cap_adv_and_free_float():
    snapshot = pd.DataFrame(
        [
            {
                "requested_ric": "AAPL.O",
                "Company Market Cap": 3_000_000_000_000,
                "Average Daily Value Traded - 30 Days": 12_000_000_000,
                "Free Float (Percent)": 99,
                "Outstanding Shares": 15_000_000_000,
                "asof_date": "2026-07-09",
            },
            {
                "requested_ric": "IBM.N",
                "Company Market Cap": 200_000_000_000,
                "Average Daily Value Traded - 30 Days": 300_000_000,
                "Free Float (Percent)": 92,
                "Outstanding Shares": 900_000_000,
                "asof_date": "2026-07-09",
            },
            {
                "requested_ric": "XYZ.A",
                "Company Market Cap": 2_000_000_000,
                "Average Daily Value Traded - 30 Days": 4_000_000,
                "Free Float (Percent)": 70,
                "Outstanding Shares": 50_000_000,
                "asof_date": "2026-07-09",
            },
        ]
    )

    core = build_usa_large_liquid_core(_security_master(), snapshot)

    assert set(core["ric"]) == {"AAPL.O", "IBM.N"}
    assert core["market_cap_usd"].min() >= 1_000_000_000
    assert core["adv_value_usd"].min() >= 5_000_000
    assert core["free_float_pct"].min() >= 10
    assert set(core["research_universe_role"]) == {"usa_large_liquid_operational_core"}


def test_usa_large_liquid_snapshot_helpers_build_refinitiv_request(tmp_path):
    active = build_usa_active_liquid_core(_security_master())
    symbol_map = symbol_map_from_security_master_rows(active, tmp_path / "security_master.csv", "USA_IBKR_ACTIVE_LIQUID_CORE_V1")
    request = snapshot_request_from_symbol_map(symbol_map)
    field_map = usa_large_liquid_snapshot_field_map()
    normalized = normalize_usa_liquidity_snapshot(
        pd.DataFrame(
            [
                {
                    "requested_ric": "AAPL.O",
                    "Company Market Cap": 3_000_000_000_000,
                    "Average Daily Value Traded - 30 Days": 12_000_000_000,
                    "Free Float (Percent)": 99,
                }
            ]
        )
    )

    assert set(request["refinitiv_ric"]) == {"AAPL.O", "IBM.N", "XYZ.A"}
    assert {"TR.CompanyMarketCap", "TR.AvgDailyValTraded30D"}.issubset(set(field_map["candidate_field"]))
    assert normalized.loc[0, "market_cap_usd"] == 3_000_000_000_000


def test_write_usa_research_universe_artifacts_creates_backfill_ready_maps(tmp_path):
    security_master_path = tmp_path / "security_master.csv"
    _security_master().to_csv(security_master_path, index=False)

    artifacts = write_usa_research_universe_artifacts(
        security_master_path=security_master_path,
        run_id="USA-CORE-TEST",
        artifact_root=tmp_path / "research_universes",
        symbol_map_artifact_root=tmp_path / "refinitiv_symbol_map",
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert artifacts.active_symbol_map_path.exists()
    assert artifacts.reference_symbol_map_path.exists()
    assert manifest["active_core_rows"] == 3
    assert manifest["reference_core_rows"] == 5
    assert manifest["active_universe_id"] == "USA_IBKR_ACTIVE_LIQUID_CORE_V1"
    assert manifest["reference_universe_id"] == "USA_LISTED_REFERENCE_CORE_V1"


def test_write_usa_large_liquid_universe_artifacts_creates_request_then_filtered_map(tmp_path):
    security_master_path = tmp_path / "security_master.csv"
    _security_master().to_csv(security_master_path, index=False)

    request_only = write_usa_large_liquid_universe_artifacts(
        security_master_path=security_master_path,
        run_id="USA-LARGE-REQUEST-TEST",
        artifact_root=tmp_path / "research_universes",
        symbol_map_artifact_root=tmp_path / "refinitiv_symbol_map",
    )

    assert request_only.large_liquid_symbol_map_path is None
    assert request_only.manifest["status"] == "needs_refinitiv_liquidity_snapshot"
    assert set(pd.read_csv(request_only.snapshot_request_path)["refinitiv_ric"]) == {"AAPL.O", "IBM.N", "XYZ.A"}

    snapshot_path = tmp_path / "snapshot.csv"
    pd.DataFrame(
        [
            {
                "requested_ric": "AAPL.O",
                "Company Market Cap": 3_000_000_000_000,
                "Average Daily Value Traded - 30 Days": 12_000_000_000,
                "Free Float (Percent)": 99,
            },
            {
                "requested_ric": "IBM.N",
                "Company Market Cap": 200_000_000_000,
                "Average Daily Value Traded - 30 Days": 300_000_000,
                "Free Float (Percent)": 92,
            },
            {
                "requested_ric": "XYZ.A",
                "Company Market Cap": 2_000_000_000,
                "Average Daily Value Traded - 30 Days": 4_000_000,
                "Free Float (Percent)": 70,
            },
        ]
    ).to_csv(snapshot_path, index=False)

    built = write_usa_large_liquid_universe_artifacts(
        security_master_path=security_master_path,
        liquidity_snapshot_path=snapshot_path,
        run_id="USA-LARGE-BUILT-TEST",
        artifact_root=tmp_path / "research_universes",
        symbol_map_artifact_root=tmp_path / "refinitiv_symbol_map",
    )

    assert built.large_liquid_symbol_map_path is not None
    symbol_map = pd.read_csv(built.large_liquid_symbol_map_path)
    assert set(symbol_map["refinitiv_ric"]) == {"AAPL.O", "IBM.N"}
    assert built.manifest["status"] == "large_liquid_core_ready"
    assert built.manifest["large_liquid_core_rows"] == 2
