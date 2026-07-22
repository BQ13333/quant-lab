from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "quant_lab" / "data" / "asean_sources.py"
SPEC = spec_from_file_location("asean_sources_under_test", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None

asean_sources = module_from_spec(SPEC)
sys.modules[SPEC.name] = asean_sources
SPEC.loader.exec_module(asean_sources)

ASEAN_SOURCE_COLUMNS = asean_sources.ASEAN_SOURCE_COLUMNS
AseanExchangeSource = asean_sources.AseanExchangeSource
asean_source_registry = asean_sources.asean_source_registry
asean_yahoo_suffix_map = asean_sources.asean_yahoo_suffix_map


EXPECTED_COUNTRIES = ["Singapore", "Malaysia", "Indonesia", "Thailand", "Philippines", "Vietnam"]
ACTIVE_COUNTRIES = {"Singapore", "Philippines", "Vietnam"}


def test_asean_source_registry_shape_and_status():
    registry = asean_source_registry()

    assert registry.columns.tolist() == ASEAN_SOURCE_COLUMNS
    assert registry["country"].tolist() == EXPECTED_COUNTRIES
    assert registry["country"].is_unique
    assert registry.shape[0] == 6
    status_by_country = registry.set_index("country")["public_source_status"].to_dict()
    assert {country for country, status in status_by_country.items() if status == "active_public_adapter"} == ACTIVE_COUNTRIES
    assert status_by_country["Malaysia"] == "pending_public_source_validation"
    assert status_by_country["Indonesia"] == "pending_public_source_validation"
    assert status_by_country["Thailand"] == "pending_public_source_validation"


def test_asean_source_registry_suffixes_are_non_empty_tuples():
    registry = asean_source_registry()

    suffixes = registry.set_index("country")["preferred_yahoo_suffixes"].to_dict()

    assert suffixes["Singapore"] == (".SI",)
    assert suffixes["Malaysia"] == (".KL",)
    assert suffixes["Indonesia"] == (".JK",)
    assert suffixes["Thailand"] == (".BK",)
    assert suffixes["Philippines"] == (".PS",)
    assert suffixes["Vietnam"] == (".VN",)
    assert all(isinstance(country_suffixes, tuple) for country_suffixes in suffixes.values())
    assert all(suffix.startswith(".") for country_suffixes in suffixes.values() for suffix in country_suffixes)


def test_asean_yahoo_suffix_map_matches_registry():
    registry = asean_source_registry()
    suffix_map = asean_yahoo_suffix_map()

    assert set(suffix_map) == set(EXPECTED_COUNTRIES)
    assert suffix_map == registry.set_index("country")["preferred_yahoo_suffixes"].to_dict()


def test_asean_exchange_source_as_record_uses_registry_columns():
    source = AseanExchangeSource(
        exchange="Example Exchange",
        country="Example",
        public_source_status="pending_public_source_validation",
        preferred_yahoo_suffixes=(".EX",),
        notes="Pending manual validation.",
    )

    assert list(source.as_record()) == ASEAN_SOURCE_COLUMNS
