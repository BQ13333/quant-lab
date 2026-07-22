from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_extraction_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "run_all_source_extraction.py"
    spec = importlib.util.spec_from_file_location("run_all_source_extraction", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_extraction = _load_extraction_module()
_parse_csv_arg = _extraction._parse_csv_arg
_market_field_map_from_catalog = _extraction._market_field_map_from_catalog
_refinitiv_universe_from_symbol_map = _extraction._refinitiv_universe_from_symbol_map


def test_refinitiv_symbol_map_country_filters(tmp_path):
    symbol_map = pd.DataFrame(
        {
            "country": ["Malaysia", "Singapore", "United States", "Japan"],
            "exchange": ["Bursa Malaysia", "SGX", "NYSE", "JPX"],
            "ticker": ["1155", "D05", "A", "7203"],
            "yahoo_ticker": ["1155.KL", "D05.SI", "A", "7203.T"],
            "refinitiv_ric": ["MBBM.KL", "DBSM.SI", "A.N", "7203.T"],
            "refinitiv_symbol_status": ["validated_price_history"] * 4,
        }
    )
    path = tmp_path / "symbol_map.csv"
    symbol_map.to_csv(path, index=False)

    universe = _refinitiv_universe_from_symbol_map(
        str(path),
        max_rows=10,
        include_countries=_parse_csv_arg("Malaysia, Singapore, Japan"),
        exclude_countries=_parse_csv_arg("Japan"),
    )

    assert set(universe["country"]) == {"Malaysia", "Singapore"}
    assert set(universe["yahoo_ticker"]) == {"MBBM.KL", "DBSM.SI"}
    assert set(universe["source_yahoo_ticker"]) == {"1155.KL", "D05.SI"}


def test_refinitiv_symbol_map_country_filter_errors_when_empty(tmp_path):
    symbol_map = pd.DataFrame(
        {
            "country": ["United States"],
            "ticker": ["A"],
            "refinitiv_ric": ["A.N"],
            "refinitiv_symbol_status": ["validated_price_history"],
        }
    )
    path = tmp_path / "symbol_map.csv"
    symbol_map.to_csv(path, index=False)

    try:
        _refinitiv_universe_from_symbol_map(str(path), max_rows=10, include_countries=_parse_csv_arg("Malaysia"))
    except SystemExit as exc:
        assert "removed every validated symbol" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_market_field_map_uses_unambiguous_country_fallback():
    field_catalog = pd.DataFrame(
        {
            "country": ["South Korea", "South Korea", "South Korea", "South Korea", "South Korea"],
            "exchange": ["KRX", "KRX", "KRX", "KRX", "KRX"],
            "field_role": ["Close", "Open", "High", "Low", "Volume"],
            "field": ["TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS"],
        }
    )

    field_map = _market_field_map_from_catalog(field_catalog, {"country": "South Korea", "exchange": "KRX 코스닥"})

    assert field_map == {
        "Close": "TRDPRC_1",
        "Open": "OPEN_PRC",
        "High": "HIGH_1",
        "Low": "LOW_1",
        "Volume": "ACVOL_UNS",
    }


def test_market_field_map_skips_ambiguous_country_fallback():
    field_catalog = pd.DataFrame(
        {
            "country": ["Vietnam", "Vietnam"],
            "exchange": ["HNX", "HOSE"],
            "field_role": ["Close", "Close"],
            "field": ["TR.CLOSEPRICE", "TRDPRC_1"],
        }
    )

    assert _market_field_map_from_catalog(field_catalog, {"country": "Vietnam", "exchange": "UPCoM"}) is None
