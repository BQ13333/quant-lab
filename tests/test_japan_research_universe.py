from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from quant_lab.data.japan_research_universe import (
    build_japan_active_operating_core,
    build_japan_historical_listed_reference,
    deduplicate_japan_quote_rows,
    japan_research_universe_quality_metrics,
    jpx_primary_ordinary_share_scope,
    normalize_jpx_listings,
    reconcile_japan_research_universe,
    write_japan_research_universe_artifacts,
)


KNOWN_AT = "2026-07-01T00:00:00Z"


def _master_row(
    ric: str,
    code: str,
    isin: str,
    *,
    exchange: str = "TYO",
    active: bool = True,
    primary: bool = True,
    currency: object = "JPY",
    known_at: object = KNOWN_AT,
) -> dict[str, object]:
    return {
        "ric": ric,
        "document_title": f"{code} ordinary share",
        "common_name": f"Company {code}",
        "exchange_country": "JPN",
        "exchange_name": "Tokyo Stock Exchange" if exchange == "TYO" else "JAPAN - CHI-X",
        "exchange_code": exchange,
        "rcs_asset_category_leaf": "Ordinary Share",
        "instrument_permid": f"PERM-{isin}",
        "ticker_symbol": code,
        "is_primary_ric": primary,
        "is_primary_issue_ric": primary,
        "active_state_bucket": "active" if active else "inactive",
        "country_code": "JPN",
        "country": "Japan",
        "retire_date": pd.NA if active else "2020-01-02",
        "first_trade_date": "2000-01-03",
        "isin": isin,
        "sedol": f"S{code}",
        "price_currency": currency,
        "known_at_utc": known_at,
        "active_flag": active,
    }


def _listings(*codes: str, include_etf: bool = False, known_at: object = KNOWN_AT) -> pd.DataFrame:
    rows = [
        {
            "日付": 20260630,
            "コード": code,
            "銘柄名": f"JPX {code}",
            "市場・商品区分": "プライム（内国株式）",
            "known_at_utc": known_at,
        }
        for code in codes
    ]
    if include_etf:
        rows.append(
            {
                "日付": 20260630,
                "コード": "1305",
                "銘柄名": "ETF",
                "市場・商品区分": "ETF・ETN",
                "known_at_utc": known_at,
            }
        )
    return pd.DataFrame(rows)


def test_scope_normalization_predeclares_domestic_prime_standard_growth_only():
    raw = pd.DataFrame(
        {
            "日付": [20260630] * 5,
            "コード": ["1001", "1002", "1003", "1004", "1005"],
            "銘柄名": list("ABCDE"),
            "市場・商品区分": [
                "プライム（内国株式）",
                "スタンダード（内国株式）",
                "グロース（内国株式）",
                "PRO Market",
                "ETF・ETN",
            ],
        }
    )

    normalized = normalize_jpx_listings(raw)
    scope = jpx_primary_ordinary_share_scope()

    assert normalized["jpx_asof_date"].eq("2026-06-30").all()
    assert normalized["in_primary_ordinary_share_scope"].tolist() == [True, True, True, False, False]
    assert set(scope["scope_id"]) == {
        "prime_domestic_ordinary_shares",
        "standard_domestic_ordinary_shares",
        "growth_domestic_ordinary_shares",
    }


def test_quote_rows_dedupe_to_stable_instrument_and_prefer_primary_tse_quote():
    master = pd.DataFrame(
        [
            _master_row("7203.T", "7203", "JP0000007203"),
            _master_row(
                "7203.CHJ^I25",
                "7203",
                "JP0000007203",
                exchange="CHJ",
                active=False,
                primary=False,
            ),
        ]
    )

    stable = deduplicate_japan_quote_rows(master)

    assert len(stable) == 1
    assert stable.loc[0, "ric"] == "7203.T"
    assert stable.loc[0, "stable_instrument_id"] == "isin:JP0000007203"
    assert stable.loc[0, "quote_row_count"] == 2
    assert stable.loc[0, "jpx_quote_row_count"] == 1


def test_builders_reconcile_current_core_and_keep_inactive_reference_names():
    master = pd.DataFrame(
        [
            _master_row("7203.T", "7203", "JP0000007203"),
            _master_row("6758.T", "6758", "JP0000006758"),
            _master_row("9999.T^A20", "9999", "JP0000009999", active=False, primary=False),
            _master_row("1111.T", "1111", "JP0000001111"),
        ]
    )
    listings = _listings("7203", "6758", "2222", include_etf=True)

    reconciliation = reconcile_japan_research_universe(master, listings)
    active = build_japan_active_operating_core(master, listings)
    reference = build_japan_historical_listed_reference(master, listings)

    assert set(active["refinitiv_ric"]) == {"7203.T", "6758.T"}
    assert set(reference["refinitiv_ric"]) == {"7203.T", "6758.T", "9999.T^A20", "1111.T"}
    assert reference["point_in_time_membership_claimed"].eq(False).all()
    assert set(reconciliation["unmatched_classification"]) >= {
        "missing_active_primary_security_master_instrument",
        "missing_current_public_jpx_listing",
        "out_of_scope_market_category",
    }


def test_quality_gate_requires_complete_jpy_and_source_backed_known_at():
    good_master = pd.DataFrame([_master_row("7203.T", "7203", "JP0000007203")])
    good = japan_research_universe_quality_metrics(good_master, _listings("7203"))
    assert good["passed"].all()

    weak_master = pd.DataFrame(
        [_master_row("7203.T", "7203", "JP0000007203", currency=pd.NA, known_at=pd.NA)]
    )
    weak = japan_research_universe_quality_metrics(
        weak_master, _listings("7203", known_at=pd.NA)
    ).set_index("metric")
    assert not weak.loc["active_core_currency_present_rate", "passed"]
    assert not weak.loc["public_listing_known_at_rate", "passed"]
    assert not weak.loc["reference_security_master_known_at_rate", "passed"]


def test_combined_market_input_is_filtered_before_jpx_duplicate_classification():
    master = pd.DataFrame([_master_row("7203.T", "7203", "JP0000007203")])
    listings = pd.concat(
        [
            _listings("7203").assign(country="Japan", exchange="JPX"),
            _listings("7203").assign(country="United States", exchange="NASDAQ"),
        ],
        ignore_index=True,
    )

    normalized = normalize_jpx_listings(listings)
    reconciliation = reconcile_japan_research_universe(master, listings)

    assert len(normalized) == 1
    assert reconciliation["unmatched_classification"].ne("duplicate_public_listing_code").all()
    assert reconciliation["reconciliation_status"].tolist() == ["matched"]


def test_writer_checksums_inputs_refuses_overwrite_and_does_not_claim_pit(tmp_path):
    master_path = tmp_path / "security_master.csv"
    listings_path = tmp_path / "jpx_listings.csv"
    pd.DataFrame([_master_row("7203.T", "7203", "JP0000007203")]).to_csv(master_path, index=False)
    _listings("7203").to_csv(listings_path, index=False)

    artifacts = write_japan_research_universe_artifacts(
        security_master_path=master_path,
        jpx_listings_path=listings_path,
        run_id="JPN-JPX-TEST",
        artifact_root=tmp_path / "artifacts",
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    expected = hashlib.sha256(master_path.read_bytes()).hexdigest()

    assert manifest["input_sha256"]["merged_japan_security_master"] == expected
    assert manifest["input_checksums_verified_after_build"] is True
    assert manifest["scorecard_eligible"] is True
    assert manifest["historical_pit_ready"] is False
    assert manifest["historical_backtest_eligible"] is False
    assert artifacts.unmatched_path.exists()
    assert artifacts.quality_metrics_path.exists()

    with pytest.raises(FileExistsError):
        write_japan_research_universe_artifacts(
            security_master_path=master_path,
            jpx_listings_path=listings_path,
            run_id="JPN-JPX-TEST",
            artifact_root=tmp_path / "artifacts",
        )
