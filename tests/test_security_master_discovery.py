from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import pandas as pd
import pytest

from quant_lab.data.security_master_discovery import (
    DISCOVERY_ENRICHMENT_FIELDS,
    DISCOVERY_SEARCH_SELECT,
    build_security_master_discovery_tables,
    merge_security_master_discovery_data,
    normalize_discovery_search_hits,
    normalize_security_master_enrichment,
    search_security_master_discovery,
    security_master_discovery_filter,
    write_refinitiv_security_master_discovery_artifacts,
)


_MERGE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "merge_refinitiv_security_master_discovery_artifacts.py"
)
_MERGE_SPEC = importlib.util.spec_from_file_location(
    "merge_refinitiv_security_master_discovery_artifacts",
    _MERGE_SCRIPT_PATH,
)
assert _MERGE_SPEC is not None
assert _MERGE_SPEC.loader is not None
_MERGE_MODULE = importlib.util.module_from_spec(_MERGE_SPEC)
_MERGE_SPEC.loader.exec_module(_MERGE_MODULE)
_component_dirs = _MERGE_MODULE._component_dirs
_read_components_file = _MERGE_MODULE._read_components_file


class FakeDiscovery:
    class Views:
        EQUITY_QUOTES = "EQUITY_QUOTES"

    def __init__(self, rows_by_route: dict[tuple[str, str], list[dict[str, object]]]):
        self.rows_by_route = rows_by_route
        self.calls: list[dict[str, object]] = []

    def search(self, view, filter, select, top, skip):
        self.calls.append({"view": view, "filter": filter, "select": select, "top": top, "skip": skip})
        code = re.search(r"ExchangeCountry eq '([^']+)'", filter).group(1)
        bucket = "active" if "AssetState eq 'AC'" in filter else "inactive"
        rows = self.rows_by_route.get((code, bucket), [])
        return pd.DataFrame(rows[skip : skip + top])


class FakeGetData:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def get_data(self, universe, fields):
        universe = list(universe)
        fields = list(fields)
        self.calls.append({"universe": universe, "fields": fields})
        rows = []
        for ric in universe:
            inactive = "DEAD" in ric or "OLD" in ric
            suffix = ric.split(".")[-1]
            exchange_name = {
                "KL": "Bursa Malaysia",
                "SI": "Singapore Exchange",
                "JK": "Indonesia Stock Exchange",
                "T": "Tokyo Stock Exchange",
                "N": "New York Stock Exchange",
                "O": "NASDAQ",
            }.get(suffix, f"{suffix} Exchange")
            exchange_country_code = {"KL": "MY", "SI": "SG", "JK": "ID", "T": "JP", "N": "US", "O": "US"}.get(
                suffix, suffix
            )
            price_currency = {"KL": "MYR", "SI": "SGD", "JK": "IDR", "T": "JPY", "N": "USD", "O": "USD"}.get(
                suffix, "USD"
            )
            rows.append(
                {
                    "Instrument": ric,
                    "Common Name": ric.split(".")[0],
                    "Exchange Name": exchange_name,
                    "Exchange Country Code": exchange_country_code,
                    "Asset Status": "Retired" if inactive else "Active",
                    "Retire Date": pd.Timestamp("2024-07-05 13:45:00") if inactive else pd.NA,
                    "First Trade Date": "2001-02-03",
                    "ISIN": f"ISIN-{ric}",
                    "SEDOL": f"SEDOL-{ric}",
                    "Organization ID": f"ORG-{ric}",
                    "PermID": f"PERM-{ric}",
                    "Price Currency": price_currency,
                    "RIC": ric,
                }
            )
        return pd.DataFrame(rows)


def _search_row(ric: str, country_code: str, asset_state: str, permid: str) -> dict[str, object]:
    return {
        "RIC": ric,
        "DocumentTitle": f"{ric} title",
        "CommonName": f"{ric} common",
        "ExchangeCountry": country_code,
        "ExchangeName": f"{country_code} Exchange",
        "ExchangeCode": country_code,
        "AssetState": asset_state,
        "AssetStateName": "Active" if asset_state == "AC" else "Retired",
        "RCSAssetCategoryLeaf": "Ordinary Share",
        "InstrumentPermID": permid,
        "TickerSymbol": ric.split(".")[0],
        "IsPrimaryRIC": True,
        "IsPrimaryIssueRIC": True,
    }


def test_security_master_discovery_filter_strings_match_discovery_route():
    assert (
        security_master_discovery_filter("MYS", "inactive")
        == "BusinessEntity eq 'QUOTExEQUITY' and RCSAssetCategoryLeaf eq 'Ordinary Share' and "
        "ExchangeCountry eq 'MYS' and AssetState ne 'AC' and AssetState ne null"
    )
    assert (
        security_master_discovery_filter("SGP", "active")
        == "BusinessEntity eq 'QUOTExEQUITY' and RCSAssetCategoryLeaf eq 'Ordinary Share' and "
        "ExchangeCountry eq 'SGP' and AssetState eq 'AC'"
    )
    assert (
        security_master_discovery_filter("jp", "active")
        == "BusinessEntity eq 'QUOTExEQUITY' and RCSAssetCategoryLeaf eq 'Ordinary Share' and "
        "ExchangeCountry eq 'JPN' and AssetState eq 'AC'"
    )
    assert (
        security_master_discovery_filter("jp", "active", primary_only=True)
        == "BusinessEntity eq 'QUOTExEQUITY' and RCSAssetCategoryLeaf eq 'Ordinary Share' and "
        "ExchangeCountry eq 'JPN' and IsPrimaryRIC eq true and IsPrimaryIssueRIC eq true and AssetState eq 'AC'"
    )
    assert (
        security_master_discovery_filter("jp", "inactive", extra_filter="TickerSymbol ge '1000' and TickerSymbol lt '2000'")
        == "BusinessEntity eq 'QUOTExEQUITY' and RCSAssetCategoryLeaf eq 'Ordinary Share' and "
        "ExchangeCountry eq 'JPN' and (TickerSymbol ge '1000' and TickerSymbol lt '2000') and "
        "AssetState ne 'AC' and AssetState ne null"
    )


def test_discovery_search_uses_equity_quotes_filter_select_and_resumable_pagination():
    rows = [_search_row(f"AAA{i}.KL", "MYS", "AC", f"PERM{i}") for i in range(5)]
    discovery = FakeDiscovery({("MYS", "active"): rows})

    hits, page_log, pagination = search_security_master_discovery(
        discovery,
        country_codes=("MYS",),
        state_buckets=("active",),
        top=2,
        max_rows=3,
    )

    assert [call["skip"] for call in discovery.calls] == [0, 2]
    assert all(call["view"] == "EQUITY_QUOTES" for call in discovery.calls)
    assert all(call["select"] == DISCOVERY_SEARCH_SELECT for call in discovery.calls)
    assert discovery.calls[0]["filter"] == security_master_discovery_filter("MYS", "active")
    assert hits["ric"].tolist() == ["AAA0.KL", "AAA1.KL", "AAA2.KL"]
    assert pagination["stopped_by"] == "max_rows"
    assert pagination["resume"]["MYS:active"]["next_skip"] == 3
    assert page_log["rows"].tolist() == [2, 1]


def test_inactive_rows_and_retire_dates_are_normalized_into_listing_windows():
    hits = normalize_discovery_search_hits(
        pd.DataFrame([_search_row("OLD.KL", "MYS", "IA", "PERM-OLD")]),
        country_code="MYS",
        state_bucket="inactive",
    )
    enrichment = normalize_security_master_enrichment(
        pd.DataFrame(
            [
                {
                    "Instrument": "OLD.KL",
                    "Retire Date": pd.Timestamp("2024-07-05 09:15:00"),
                    "First Trade Date": pd.Timestamp("1999-01-02"),
                    "ISIN": "MYOLD",
                    "Price Currency": "MYR",
                }
            ]
        )
    )

    security_master = merge_security_master_discovery_data(hits, enrichment)
    tables = build_security_master_discovery_tables(security_master, requested_country_codes=("MYS",))

    assert security_master.loc[0, "active_state_bucket"] == "inactive"
    assert bool(tables.listing_windows.loc[0, "active_flag"]) is False
    assert tables.listing_windows.loc[0, "delisting_date"] == "2024-07-05"
    assert tables.identifier_history.query("id_type == 'isin'").iloc[0]["effective_to"] == "2024-07-05"
    assert tables.quality.loc[0, "delisted_count"] == 1
    assert tables.quality.loc[0, "asof_year"] == 1999


def test_artifact_writer_outputs_required_files_and_complete_manifest(tmp_path):
    discovery_rows = {
        ("MYS", "active"): [_search_row("AAA.KL", "MYS", "AC", "PERM-MY-A")],
        ("MYS", "inactive"): [_search_row("DEAD.KL", "MYS", "IN", "PERM-MY-D")],
        ("SGP", "active"): [_search_row("AAA.SI", "SGP", "AC", "PERM-SG-A")],
        ("SGP", "inactive"): [_search_row("DEAD.SI", "SGP", "IN", "PERM-SG-D")],
        ("IDN", "active"): [_search_row("AAA.JK", "IDN", "AC", "PERM-ID-A")],
        ("IDN", "inactive"): [_search_row("DEAD.JK", "IDN", "IN", "PERM-ID-D")],
    }
    discovery = FakeDiscovery(discovery_rows)
    get_data = FakeGetData()

    artifacts = write_refinitiv_security_master_discovery_artifacts(
        discovery_client=discovery,
        get_data_client=get_data,
        artifact_root=tmp_path,
        run_id="SECMASTER-DISCOVERY",
        top=10,
        sleep_seconds=0,
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    listing_windows = pd.read_parquet(artifacts.listing_windows_path)

    assert artifacts.entities_path.exists()
    assert artifacts.security_master_path.exists()
    assert artifacts.identifier_history_path.exists()
    assert artifacts.listing_windows_path.exists()
    assert artifacts.universe_membership_path.exists()
    assert artifacts.quality_path.exists()
    assert manifest["status"] == "complete"
    assert manifest["dead_name_rows"] == 3
    assert manifest["artifacts"]["security_master"] == str(artifacts.security_master_path.resolve())
    assert len(manifest["output_sha256"]["security_master"]) == 64
    assert pd.read_csv(artifacts.security_master_path).shape[0] == 6
    assert set(manifest["countries"]) == {"Malaysia", "Singapore", "Indonesia"}
    assert set(artifacts.quality["country"]) == {"Malaysia", "Singapore", "Indonesia"}
    assert set(listing_windows.loc[~listing_windows["active_flag"], "delisting_date"]) == {"2024-07-05"}
    assert get_data.calls[0]["fields"] == list(DISCOVERY_ENRICHMENT_FIELDS)


def test_security_master_discovery_supports_expansion_country_codes(tmp_path):
    discovery_rows = {
        ("JPN", "active"): [_search_row("7203.T", "JPN", "AC", "PERM-JP-A")],
        ("JPN", "inactive"): [_search_row("DEAD.T", "JPN", "IN", "PERM-JP-D")],
    }
    discovery = FakeDiscovery(discovery_rows)
    get_data = FakeGetData()

    artifacts = write_refinitiv_security_master_discovery_artifacts(
        discovery_client=discovery,
        get_data_client=get_data,
        artifact_root=tmp_path,
        run_id="SECMASTER-JPN",
        country_codes=("jp",),
        top=10,
        sleep_seconds=0,
        primary_only=True,
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    listing_windows = pd.read_parquet(artifacts.listing_windows_path)

    assert manifest["status"] == "complete"
    assert manifest["requested_country_codes"] == ["JPN"]
    assert manifest["primary_only"] is True
    assert manifest["extra_filter"] is None
    assert manifest["countries"] == ["Japan"]
    assert manifest["dead_name_rows"] == 1
    assert set(artifacts.quality["country"]) == {"Japan"}
    assert set(listing_windows["currency"]) == {"JPY"}


def test_artifact_writer_records_extra_filter_for_sharded_runs(tmp_path):
    discovery_rows = {
        ("JPN", "inactive"): [_search_row("1500.T", "JPN", "IN", "PERM-JP-D")],
    }
    discovery = FakeDiscovery(discovery_rows)
    get_data = FakeGetData()

    artifacts = write_refinitiv_security_master_discovery_artifacts(
        discovery_client=discovery,
        get_data_client=get_data,
        artifact_root=tmp_path,
        run_id="SECMASTER-JPN-INACTIVE-SHARD",
        country_codes=("jp",),
        state_buckets=("inactive",),
        top=10,
        sleep_seconds=0,
        extra_filter="TickerSymbol ge '1000' and TickerSymbol lt '2000'",
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert manifest["extra_filter"] == "TickerSymbol ge '1000' and TickerSymbol lt '2000'"
    assert manifest["pagination"]["extra_filter"] == "TickerSymbol ge '1000' and TickerSymbol lt '2000'"
    assert "(TickerSymbol ge '1000' and TickerSymbol lt '2000')" in discovery.calls[0]["filter"]


def test_merge_components_file_supports_newline_dirs_and_manifest_paths(tmp_path):
    component_a = tmp_path / "component-a"
    component_b = tmp_path / "component-b"
    component_a.mkdir()
    component_b.mkdir()
    (component_a / "manifest.json").write_text("{}", encoding="utf-8")
    (component_b / "manifest.json").write_text("{}", encoding="utf-8")
    components_file = tmp_path / "components.txt"
    components_file.write_text(
        f"\n{component_a}\n\n{component_b / 'manifest.json'}\n",
        encoding="utf-8",
    )

    component_values = _read_components_file(components_file)
    component_dirs = _component_dirs(tuple(component_values))

    assert component_dirs == [component_a, component_b]


def test_merge_component_dirs_can_skip_failed_shards_without_manifests(tmp_path):
    successful_component = tmp_path / "successful"
    failed_component = tmp_path / "failed"
    successful_component.mkdir()
    failed_component.mkdir()
    (successful_component / "manifest.json").write_text("{}", encoding="utf-8")

    component_dirs = _component_dirs(
        (str(successful_component), str(failed_component)),
        skip_missing_manifest=True,
    )

    assert component_dirs == [successful_component]
    with pytest.raises(SystemExit, match="has no manifest"):
        _component_dirs((str(successful_component), str(failed_component)))
