from __future__ import annotations

from quant_lab.data.japan_market_rule_history import (
    SOURCES,
    price_limit_rule_bands,
    price_limit_transition_evidence,
    rule_amendment_ledger,
    rule_coverage,
    tick_regime_windows,
    tick_rule_bands,
    tick_transition_evidence,
)


def test_tick_regime_windows_cover_the_full_requested_rule_interval() -> None:
    windows = tick_regime_windows()
    assert len(windows) == 8
    assert windows["rule_table_verified"].all()
    assert windows.loc[windows["regime_id"].eq("legacy_20060501"), "valid_from"].iloc[0] == "2006-07-01"
    assert windows.loc[windows["regime_id"].eq("phase_1"), "valid_from"].iloc[0] == "2014-01-14"
    assert windows.loc[windows["regime_id"].eq("phase_3_topix500"), "valid_from"].iloc[0] == "2023-06-05"


def test_tick_tables_match_official_phase_denominators() -> None:
    bands = tick_rule_bands()
    assert len(bands) == 114
    assert bands.groupby("regime_id").size().to_dict() == {
        "legacy_20060501": 9,
        "legacy_20080722": 9,
        "legacy_20100104": 16,
        "other_issues_verified": 16,
        "phase_1": 16,
        "phase_2": 16,
        "phase_3_topix100": 16,
        "phase_3_topix500": 16,
    }
    legacy = bands.loc[bands["regime_id"].eq("legacy_20060501")]
    assert legacy.loc[legacy["price_upper_inclusive_yen"].eq(100_000), "tick_size_yen"].iloc[0] == 100
    narrowed = bands.loc[bands["regime_id"].eq("legacy_20080722")]
    assert narrowed.loc[narrowed["price_upper_inclusive_yen"].eq(300_000), "tick_size_yen"].iloc[0] == 100
    phase2 = bands.loc[bands["regime_id"].eq("phase_2")]
    assert phase2.iloc[0]["tick_size_yen"] == 0.1
    phase3 = bands.loc[bands["regime_id"].eq("phase_3_topix100")]
    assert phase3.loc[phase3["price_upper_inclusive_yen"].eq(5_000), "tick_size_yen"].iloc[0] == 1


def test_price_limit_rule_source_is_official_and_dated() -> None:
    assert SOURCES["price_limit_rules_20141201"].startswith("https://www.jpx.co.jp/")
    assert SOURCES["price_limit_rules_20141201"].endswith("price_limits_20141201.pdf")


def test_price_limit_coverage_passes_only_the_normal_rule_continuity_lane() -> None:
    coverage = rule_coverage(assignments=[])
    row = coverage.loc[coverage["subcomponent"].eq("normal_price_limit_table")].iloc[0]
    assert row["coverage"] == 1.0
    assert row["status"] == "pass_two_regimes_direct_transition_and_complete_2006_2010_amendment_ledger"
    temporary = coverage.loc[coverage["subcomponent"].eq("temporary_price_limit_events")].iloc[0]
    assert temporary["coverage"] == 0.0


def test_pre_program_tick_transition_evidence_has_direct_baseline_and_transitions() -> None:
    evidence = tick_transition_evidence()
    assert len(evidence) == 6
    assert evidence["evidence_date"].tolist() == [
        "2006-05-01",
        "2006-09-26",
        "2008-01-29",
        "2008-07-22",
        "2010-01-04",
        "2013-05-14",
    ]
    assert evidence["continuous_interval_proven"].all()
    coverage = rule_coverage(assignments=[])
    row = coverage.loc[coverage["subcomponent"].eq("pre_2014_tick_rule_table")].iloc[0]
    assert row["coverage"] == 1.0
    assert row["source_rows"] == 6


def test_price_limit_transition_evidence_proves_continuity_beyond_endpoints() -> None:
    evidence = price_limit_transition_evidence()
    assert len(evidence) == 7
    assert evidence["evidence_date"].iloc[0] == "2006-05-01"
    assert evidence["evidence_date"].iloc[-1] == "2026-06-30"
    assert evidence["continuous_interval_proven"].all()


def test_amendment_ledger_is_complete_for_2006_through_2010() -> None:
    ledger = rule_amendment_ledger()
    assert ledger.groupby("rule_family").size().to_dict() == {"normal_price_limit": 9, "tick_size": 6}
    changed = ledger.loc[ledger["normal_grid_effect"].eq("grid_changed")]
    assert changed[["effective_date", "rule_family"]].to_records(index=False).tolist() == [
        ("2008-07-22", "tick_size"),
        ("2010-01-04", "tick_size"),
        ("2010-01-04", "normal_price_limit"),
    ]


def test_normal_price_limit_bands_encode_pre_and_post_arrowhead_rules() -> None:
    bands = price_limit_rule_bands()
    assert bands.groupby("regime_id").size().to_dict() == {
        "arrowhead_20100104": 34,
        "pre_arrowhead_20060701": 29,
    }
    pre = bands.loc[bands["regime_id"].eq("pre_arrowhead_20060701")]
    post = bands.loc[bands["regime_id"].eq("arrowhead_20100104")]
    assert pre.loc[pre["base_price_upper_exclusive_yen"].eq(1_500), "daily_limit_yen"].iloc[0] == 200
    assert post.loc[post["base_price_upper_exclusive_yen"].eq(1_000), "daily_limit_yen"].iloc[0] == 150
    assert pre["valid_from"].unique().tolist() == ["2006-07-01"]
    assert post["valid_to"].unique().tolist() == ["2026-06-30"]
