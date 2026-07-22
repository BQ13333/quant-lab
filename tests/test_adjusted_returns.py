from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.data.adjusted_returns import (
    build_reconstruction_manifest,
    build_daily_adjusted_returns,
    build_event_price_reconciliation,
    normalize_corporate_action_events,
    prepare_daily_prices,
)


def _prices(rows: list[dict[str, object]]) -> pd.DataFrame:
    return prepare_daily_prices(pd.DataFrame(rows))


def test_dividend_price_event_date_uses_ex_date_not_announcement_date():
    prices = _prices(
        [
            {"date": "2024-01-02", "ticker": "AAA", "country": "Malaysia", "currency": "MYR", "close": 100.0},
            {"date": "2024-01-05", "ticker": "AAA", "country": "Malaysia", "currency": "MYR", "close": 99.0},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_key": "div1",
                "ticker": "AAA",
                "country": "Malaysia",
                "event_family": "dividend",
                "Dividend Announcement Date": "2024-01-01",
                "Dividend Ex Date": "2024-01-05",
                "Gross Dividend Amount": 1.0,
                "Dividend Currency": "MYR",
            }
        ]
    )

    normalized = normalize_corporate_action_events(events, prices=prices)

    assert normalized.loc[0, "known_at_date"] == pd.Timestamp("2024-01-01")
    assert normalized.loc[0, "price_event_date"] == pd.Timestamp("2024-01-05")
    assert normalized.loc[0, "price_event_date"] != normalized.loc[0, "known_at_date"]


def test_dividend_currency_falls_back_to_country_price_currency_policy():
    prices = _prices(
        [
            {"date": "2024-01-02", "ticker": "AAA", "country": "Malaysia", "currency": "MYR", "close": 100.0},
            {"date": "2024-01-05", "ticker": "AAA", "country": "Malaysia", "currency": "MYR", "close": 99.0},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_key": "div_missing_ccy",
                "ticker": "AAA",
                "country": "Malaysia",
                "event_family": "dividend",
                "Dividend Announcement Date": "2024-01-01",
                "Dividend Ex Date": "2024-01-05",
                "Gross Dividend Amount": 1.0,
            }
        ]
    )

    normalized = normalize_corporate_action_events(events, prices=prices)
    reconciliation = build_event_price_reconciliation(normalized, prices)

    assert normalized.loc[0, "cash_currency"] == "MYR"
    assert normalized.loc[0, "cash_currency_source"] == "country_price_currency_policy"
    assert reconciliation.loc[0, "reconciliation_status"] == "applied"


def test_us_dividend_currency_falls_back_to_usd():
    prices = _prices(
        [
            {"date": "2024-01-02", "ticker": "AAA", "country": "United States", "currency": "USD", "close": 100.0},
            {"date": "2024-01-05", "ticker": "AAA", "country": "United States", "currency": "USD", "close": 99.0},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_key": "us-dividend",
                "ticker": "AAA",
                "country": "United States",
                "event_family": "dividend",
                "Dividend Announcement Date": "2024-01-01",
                "Dividend Ex Date": "2024-01-05",
                "Gross Dividend Amount": 1.0,
            }
        ]
    )

    normalized = normalize_corporate_action_events(events, prices=prices)
    reconciliation = build_event_price_reconciliation(normalized, prices)

    assert normalized.loc[0, "cash_currency"] == "USD"
    assert normalized.loc[0, "cash_currency_source"] == "country_price_currency_policy"
    assert reconciliation.loc[0, "reconciliation_status"] == "applied"


def test_reconciliation_handles_nullable_ticker_and_event_family():
    prices = _prices(
        [{"date": "2024-01-02", "ticker": "AAA", "country": "United States", "currency": "USD", "close": 100.0}]
    )
    events = pd.DataFrame(
        [
            {
                "event_key": "missing-identifiers",
                "ticker": pd.NA,
                "country": "United States",
                "event_family": pd.NA,
                "price_event_date": pd.NaT,
                "known_at_date": pd.NaT,
                "cash_amount": pd.NA,
                "adjustment_factor": pd.NA,
            }
        ]
    )

    reconciliation = build_event_price_reconciliation(events, prices)

    assert reconciliation.loc[0, "ticker"] == ""
    assert reconciliation.loc[0, "event_family"] == ""
    assert reconciliation.loc[0, "reconciliation_status"] == "diagnostic_no_event"


def test_capital_change_price_event_date_uses_ex_date_then_effective_date_not_announcement_date():
    prices = _prices(
        [
            {"date": "2024-01-02", "ticker": "AAA", "country": "Singapore", "currency": "SGD", "close": 100.0},
            {"date": "2024-01-03", "ticker": "AAA", "country": "Singapore", "currency": "SGD", "close": 50.0},
            {"date": "2024-01-04", "ticker": "BBB", "country": "Singapore", "currency": "SGD", "close": 10.0},
            {"date": "2024-01-08", "ticker": "BBB", "country": "Singapore", "currency": "SGD", "close": 10.0},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_key": "split_ex",
                "ticker": "AAA",
                "country": "Singapore",
                "event_family": "capital_change",
                "Capital Change Announcement Date": "2024-01-01",
                "Capital Change Effective Date": "2024-01-04",
                "Capital Change Ex Date": "2024-01-03",
                "Adjustment Factor": 0.5,
            },
            {
                "event_key": "split_effective",
                "ticker": "BBB",
                "country": "Singapore",
                "event_family": "capital_change",
                "Capital Change Announcement Date": "2024-01-01",
                "Capital Change Effective Date": "2024-01-08",
                "Adjustment Factor": 1.0,
            },
        ]
    )

    normalized = normalize_corporate_action_events(events, prices=prices).set_index("event_key")

    assert normalized.loc["split_ex", "price_event_date"] == pd.Timestamp("2024-01-03")
    assert normalized.loc["split_effective", "price_event_date"] == pd.Timestamp("2024-01-08")
    assert normalized.loc["split_ex", "price_event_date"] != pd.Timestamp("2024-01-01")


def test_adjustment_factor_survives_source_column_with_spaces():
    events = pd.DataFrame(
        [
            {
                "event_key": "split1",
                "ticker": "AAA",
                "country": "Indonesia",
                "event_family": "capital_change",
                "Capital Change Announcement Date": "2024-01-01",
                "Capital Change Effective Date": "2024-01-03",
                "Adjustment Factor": 0.25,
            }
        ]
    )

    normalized = normalize_corporate_action_events(events)

    assert normalized.loc[0, "adjustment_factor"] == pytest.approx(0.25)


def test_non_trading_ex_dates_map_to_next_valid_trading_date():
    prices = _prices(
        [
            {"date": "2024-01-02", "ticker": "AAA", "country": "Malaysia", "currency": "MYR", "close": 100.0},
            {"date": "2024-01-05", "ticker": "AAA", "country": "Malaysia", "currency": "MYR", "close": 99.0},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_key": "div_weekend",
                "ticker": "AAA",
                "country": "Malaysia",
                "event_family": "dividend",
                "Dividend Announcement Date": "2024-01-01",
                "Dividend Ex Date": "2024-01-04",
                "Gross Dividend Amount": 1.0,
                "Dividend Currency": "MYR",
            }
        ]
    )

    normalized = normalize_corporate_action_events(events, prices=prices)
    reconciliation = build_event_price_reconciliation(normalized, prices)

    assert normalized.loc[0, "ex_trade_date"] == pd.Timestamp("2024-01-05")
    assert reconciliation.loc[0, "prev_trade_date"] == pd.Timestamp("2024-01-02")


def test_dividend_total_returns_include_cash_only_with_valid_currency_or_fx():
    prices = _prices(
        [
            {"date": "2024-01-01", "ticker": "AAA", "country": "United States", "currency": "USD", "close": 100.0},
            {"date": "2024-01-02", "ticker": "AAA", "country": "United States", "currency": "USD", "close": 99.0},
            {"date": "2024-01-01", "ticker": "BBB", "country": "United States", "currency": "USD", "close": 100.0},
            {"date": "2024-01-02", "ticker": "BBB", "country": "United States", "currency": "USD", "close": 99.0},
            {"date": "2024-01-01", "ticker": "CCC", "country": "United States", "currency": "USD", "close": 100.0},
            {"date": "2024-01-02", "ticker": "CCC", "country": "United States", "currency": "USD", "close": 99.0},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_key": "usd_cash",
                "ticker": "AAA",
                "country": "United States",
                "event_family": "dividend",
                "Dividend Announcement Date": "2024-01-01",
                "Dividend Ex Date": "2024-01-02",
                "Gross Dividend Amount": 2.0,
                "Dividend Currency": "USD",
            },
            {
                "event_key": "eur_cash_with_fx",
                "ticker": "BBB",
                "country": "United States",
                "event_family": "dividend",
                "Dividend Announcement Date": "2024-01-01",
                "Dividend Ex Date": "2024-01-02",
                "Gross Dividend Amount": 2.0,
                "Dividend Currency": "EUR",
            },
            {
                "event_key": "jpy_cash_no_fx",
                "ticker": "CCC",
                "country": "United States",
                "event_family": "dividend",
                "Dividend Announcement Date": "2024-01-01",
                "Dividend Ex Date": "2024-01-02",
                "Gross Dividend Amount": 2.0,
                "Dividend Currency": "JPY",
            },
        ]
    )
    fx_rates = pd.DataFrame(
        [{"date": "2024-01-02", "from_currency": "EUR", "to_currency": "USD", "rate": 1.1}]
    )

    normalized = normalize_corporate_action_events(events, prices=prices)
    reconciliation = build_event_price_reconciliation(normalized, prices, fx_rates=fx_rates).set_index("event_key")
    daily = build_daily_adjusted_returns(prices, reconciliation.reset_index())

    assert reconciliation.loc["usd_cash", "applied_cash"] == pytest.approx(2.0)
    assert reconciliation.loc["eur_cash_with_fx", "applied_cash"] == pytest.approx(2.2)
    assert reconciliation.loc["jpy_cash_no_fx", "reconciliation_status"] == "quarantined"
    assert reconciliation.loc["jpy_cash_no_fx", "applied_cash"] == pytest.approx(0.0)

    returns = daily.set_index(["ticker", "date"])
    assert returns.loc[("AAA", pd.Timestamp("2024-01-02")), "total_return_local_net"] == pytest.approx(0.01)
    assert returns.loc[("BBB", pd.Timestamp("2024-01-02")), "total_return_local_net"] == pytest.approx(0.012)
    assert returns.loc[("CCC", pd.Timestamp("2024-01-02")), "total_return_local_net"] == pytest.approx(-0.01)


def test_split_factor_is_not_double_applied_when_vendor_close_already_reflects_it():
    prices = _prices(
        [
            {"date": "2024-01-01", "ticker": "AAA", "country": "Singapore", "currency": "SGD", "close": 100.0},
            {"date": "2024-01-02", "ticker": "AAA", "country": "Singapore", "currency": "SGD", "close": 100.0},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_key": "split_adjusted_vendor",
                "ticker": "AAA",
                "country": "Singapore",
                "event_family": "capital_change",
                "Capital Change Announcement Date": "2023-12-15",
                "Capital Change Ex Date": "2024-01-02",
                "Adjustment Factor": 0.5,
            }
        ]
    )

    normalized = normalize_corporate_action_events(events, prices=prices)
    reconciliation = build_event_price_reconciliation(normalized, prices)
    daily = build_daily_adjusted_returns(prices, reconciliation)
    event_row = reconciliation.loc[0]
    return_row = daily.set_index("date").loc[pd.Timestamp("2024-01-02")]

    assert event_row["factor_reflected_status"] == "vendor_close_likely_already_adjusted"
    assert event_row["applied_factor"] == pytest.approx(1.0)
    assert return_row["total_return_local_net"] == pytest.approx(0.0)


def test_manifest_can_pass_with_tiny_audited_quarantine_bucket(tmp_path):
    events = pd.DataFrame(
        {
            "event_key": ["ok", "ambiguous"],
            "price_event_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        }
    )
    reconciliation = pd.DataFrame(
        {
            "event_key": ["ok", "ambiguous"],
            "ex_trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "reconciliation_status": ["applied", "quarantined"],
            "factor_reflected_status": ["not_applicable", "ambiguous_factor_reflection"],
            "applied_cash": [1.0, 0.0],
            "applied_factor": [1.0, 1.0],
        }
    )
    daily = pd.DataFrame({"total_return_local_net": [0.01, 0.02]})

    manifest = build_reconstruction_manifest(
        run_id="TEST",
        artifact_dir=tmp_path,
        normalized_events=events,
        reconciliation=reconciliation,
        daily_adjusted_returns=daily,
        quality=pd.DataFrame(),
        max_match_lag_days=7,
        factor_reflection_tolerance=0.15,
        min_matched_event_share=0.99,
        max_quarantine_share=0.51,
        paths={"daily_adjusted_returns": tmp_path / "daily.parquet"},
    )

    assert manifest["status"] == "pass"
    assert manifest["production_adjusted_return_ready"] is True
    assert manifest["quarantine_share"] == pytest.approx(0.5)
    assert manifest["ambiguous_factor_share"] == pytest.approx(0.5)
