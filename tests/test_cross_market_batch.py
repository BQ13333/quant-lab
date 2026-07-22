from __future__ import annotations

import pandas as pd

from quant_lab.research.cross_market import CrossMarketDataset, filter_dataset_by_countries
from quant_lab.research.cross_market_batch import (
    ASIA_L1_ALL_COUNTRIES,
    ASIA_L1_COUNTRIES,
    _models_for_strategy_suite,
    default_asia_l1_sweep_specs,
    select_sweep_specs,
)


def test_filter_dataset_by_countries_keeps_prices_metadata_and_group_map_aligned():
    prices = pd.DataFrame(
        {
            "AAA.KL": [1.0, 1.1],
            "BBB.SI": [2.0, 2.1],
            "CCC.N": [3.0, 3.1],
        },
        index=pd.date_range("2026-01-01", periods=2),
    )
    metadata = pd.DataFrame(
        [
            {"ticker": "AAA.KL", "country": "Malaysia"},
            {"ticker": "BBB.SI", "country": "Singapore"},
            {"ticker": "CCC.N", "country": "United States"},
        ]
    )
    group_map = metadata.set_index("ticker")["country"].reindex(prices.columns)

    filtered = filter_dataset_by_countries(
        CrossMarketDataset(prices=prices, metadata=metadata, group_map=group_map),
        include_countries=["Malaysia", "Singapore"],
    )

    assert list(filtered.prices.columns) == ["AAA.KL", "BBB.SI"]
    assert set(filtered.metadata["country"]) == {"Malaysia", "Singapore"}
    assert filtered.group_map.to_dict() == {"AAA.KL": "Malaysia", "BBB.SI": "Singapore"}


def test_default_asia_l1_sweep_specs_cover_local_usd_and_multiple_windows():
    specs = default_asia_l1_sweep_specs()

    assert len(specs) >= 8
    assert any(not spec.convert_to_usd for spec in specs)
    assert any(spec.convert_to_usd for spec in specs)
    assert {84, 126, 168}.issubset({spec.train_size for spec in specs})
    assert "United States" not in ASIA_L1_COUNTRIES
    assert "Thailand" not in ASIA_L1_COUNTRIES
    assert "Philippines" not in ASIA_L1_COUNTRIES
    assert "Vietnam" not in ASIA_L1_COUNTRIES
    assert "China" not in ASIA_L1_COUNTRIES
    assert "Taiwan" not in ASIA_L1_COUNTRIES
    assert "Thailand" in ASIA_L1_ALL_COUNTRIES
    assert "Philippines" not in ASIA_L1_ALL_COUNTRIES
    assert "Vietnam" not in ASIA_L1_ALL_COUNTRIES
    assert {"Malaysia", "Singapore", "Hong Kong", "South Korea", "Japan"}.issubset(
        set(ASIA_L1_COUNTRIES)
    )


def test_models_for_strategy_suite_source_is_much_broader_than_core():
    core = _models_for_strategy_suite("core")
    source = _models_for_strategy_suite("source")

    assert len(source) > len(core)
    assert any("src_" in model.name for model in source)


def test_select_sweep_specs_filters_and_bounds_specs():
    specs = default_asia_l1_sweep_specs()

    selected = select_sweep_specs(
        specs,
        spec_ids=["asia_l1_local_balanced_126", "asia_l1_usd_fast_84"],
        max_specs=1,
    )

    assert [spec.spec_id for spec in selected] == ["asia_l1_local_balanced_126"]

    try:
        select_sweep_specs(specs, spec_ids=["missing"])
    except ValueError as exc:
        assert "unknown sweep spec ids" in str(exc)
    else:
        raise AssertionError("expected ValueError")
