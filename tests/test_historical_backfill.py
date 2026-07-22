from __future__ import annotations

import json

import pandas as pd

from quant_lab.data.historical_backfill import (
    _clip_long_to_listing_windows,
    build_refinitiv_historical_plan,
    execute_refinitiv_historical_plan,
    historical_date_windows,
    parse_country_filter,
    refinitiv_field_map_from_catalog,
    refinitiv_universe_from_symbol_map,
)
from quant_lab.data.vendor_adapters import RefinitivRetryableProviderError
from quant_lab.data.yfinance_loader import DownloadResult


def _symbol_map(tmp_path):
    data = pd.DataFrame(
        {
            "country": ["Malaysia", "Malaysia", "Japan", "United States"],
            "exchange": ["Bursa Malaysia", "Bursa Malaysia", "JPX", "NYSE"],
            "ticker": ["1155", "1023", "7203", "A"],
            "name": ["Maybank", "CIMB", "Toyota", "Agilent"],
            "currency": ["MYR", "MYR", "JPY", "USD"],
            "refinitiv_ric": ["MBBM.KL", "CIMB.KL", "7203.T", "A.N"],
            "refinitiv_symbol_status": ["validated_price_history"] * 4,
        }
    )
    path = tmp_path / "symbol_map.csv"
    data.to_csv(path, index=False)
    return path


def test_historical_date_windows_are_non_overlapping():
    windows = historical_date_windows("2006-01-01", "2016-12-31", window_years=5)

    assert windows == [
        ("2006-01-01", "2010-12-31"),
        ("2011-01-01", "2015-12-31"),
        ("2016-01-01", "2016-12-31"),
    ]


def test_refinitiv_universe_from_symbol_map_filters_countries(tmp_path):
    universe = refinitiv_universe_from_symbol_map(
        _symbol_map(tmp_path),
        include_countries=parse_country_filter("Malaysia,Japan"),
    )

    assert set(universe["refinitiv_ric"]) == {"CIMB.KL", "MBBM.KL", "7203.T"}
    assert set(universe["country"]) == {"Malaysia", "Japan"}
    assert set(universe["yahoo_ticker"]) == {"CIMB.KL", "MBBM.KL", "7203.T"}


def test_refinitiv_universe_accepts_security_master_candidates(tmp_path):
    data = pd.DataFrame(
        {
            "country": ["Japan"],
            "exchange": ["TYO"],
            "ticker": ["7203"],
            "refinitiv_ric": ["7203.T"],
            "refinitiv_symbol_status": ["security_master_discovery_candidate"],
        }
    )
    path = tmp_path / "security_master_symbol_map.csv"
    data.to_csv(path, index=False)

    universe = refinitiv_universe_from_symbol_map(path, include_countries=parse_country_filter("Japan"))

    assert universe["refinitiv_ric"].tolist() == ["7203.T"]
    assert universe["refinitiv_validation_status"].tolist() == ["security_master_discovery_candidate"]


def test_build_refinitiv_historical_plan_batches_by_market_and_window(tmp_path):
    universe = refinitiv_universe_from_symbol_map(
        _symbol_map(tmp_path),
        include_countries=parse_country_filter("Malaysia,Japan"),
    )
    plan = build_refinitiv_historical_plan(
        universe,
        root=tmp_path / "lake",
        run_id="RUN",
        start_date="2016-01-01",
        end_date="2025-12-31",
        batch_size=2,
        window_years=5,
    )

    assert plan.shape[0] == 4
    assert set(plan["status"]) == {"pending"}
    assert set(plan["requested_tickers"]) == {1, 2}
    assert plan["output_path"].str.contains("window=001_2016-01-01_2020-12-31").any()


def test_refinitiv_field_map_country_fallback_is_safe():
    field_catalog = pd.DataFrame(
        {
            "country": ["South Korea"] * 5,
            "exchange": ["KRX"] * 5,
            "field_role": ["Close", "Open", "High", "Low", "Volume"],
            "field": ["TRDPRC_1", "OPEN_PRC", "HIGH_1", "LOW_1", "ACVOL_UNS"],
        }
    )

    assert refinitiv_field_map_from_catalog(field_catalog, "South Korea", "KRX 코스닥")["Close"] == "TRDPRC_1"


def test_execute_refinitiv_historical_plan_writes_manifest_and_resumes(tmp_path):
    universe = refinitiv_universe_from_symbol_map(
        _symbol_map(tmp_path),
        include_countries=parse_country_filter("Malaysia"),
    )
    plan = build_refinitiv_historical_plan(
        universe,
        root=tmp_path / "lake",
        run_id="RUN",
        start_date="2020-01-01",
        end_date="2021-12-31",
        batch_size=2,
        window_years=1,
    )
    calls: list[tuple[list[str], str, str]] = []

    def fake_downloader(tickers, start, end, interval, output_path, ohlcv, field_map=None):
        calls.append((list(tickers), start, end))
        dates = pd.date_range(start, periods=2)
        ohlcv_frame = pd.DataFrame(index=dates)
        for ticker in tickers:
            ohlcv_frame[(ticker, "Open")] = [10.0, 11.0]
            ohlcv_frame[(ticker, "High")] = [12.0, 13.0]
            ohlcv_frame[(ticker, "Low")] = [9.0, 10.0]
            ohlcv_frame[(ticker, "Close")] = [11.0, 12.0]
            ohlcv_frame[(ticker, "Volume")] = [1000.0, 1100.0]
        ohlcv_frame.columns = pd.MultiIndex.from_tuples(ohlcv_frame.columns)
        prices = ohlcv_frame.xs("Close", axis=1, level=1)
        prices.to_parquet(output_path)
        ohlcv_frame.to_parquet(output_path.with_suffix(".ohlcv.parquet"))
        coverage = pd.DataFrame({"ticker": tickers, "rows": [2] * len(tickers), "start": start, "end": end})
        coverage.to_csv(output_path.with_suffix(".coverage.csv"), index=False)
        return DownloadResult(prices=prices, coverage=coverage, failures={}, ohlcv=ohlcv_frame)

    first = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=fake_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
        max_plan_rows=1,
    )
    second = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=fake_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
        max_plan_rows=1,
    )

    assert len(calls) == 2
    assert first.manifest.shape[0] == 1
    assert second.manifest.shape[0] == 2
    assert set(second.manifest["status"]) == {"completed"}
    stored = pd.read_parquet(second.long_ohlcv_root)
    assert stored.shape[0] == 8
    assert set(stored["ticker"]) == {"CIMB.KL", "MBBM.KL"}


def test_execute_refinitiv_historical_plan_retries_empty_completed_rows(tmp_path):
    universe = refinitiv_universe_from_symbol_map(
        _symbol_map(tmp_path),
        include_countries=parse_country_filter("Malaysia"),
    )
    plan = build_refinitiv_historical_plan(
        universe,
        root=tmp_path / "lake",
        run_id="RUN",
        start_date="2020-01-01",
        end_date="2020-12-31",
        batch_size=2,
        window_years=1,
    )
    calls: list[str] = []

    def flaky_downloader(tickers, start, end, interval, output_path, ohlcv, field_map=None):
        calls.append(start)
        if len(calls) == 1:
            coverage = pd.DataFrame(
                {
                    "ticker": tickers,
                    "rows": [0] * len(tickers),
                    "start": [pd.NaT] * len(tickers),
                    "end": [pd.NaT] * len(tickers),
                }
            )
            coverage.to_csv(output_path.with_suffix(".coverage.csv"), index=False)
            return DownloadResult(
                prices=pd.DataFrame(),
                coverage=coverage,
                failures={ticker: "empty history" for ticker in tickers},
                ohlcv=pd.DataFrame(),
            )

        dates = pd.date_range(start, periods=2)
        ohlcv_frame = pd.DataFrame(index=dates)
        for ticker in tickers:
            ohlcv_frame[(ticker, "Open")] = [10.0, 11.0]
            ohlcv_frame[(ticker, "High")] = [12.0, 13.0]
            ohlcv_frame[(ticker, "Low")] = [9.0, 10.0]
            ohlcv_frame[(ticker, "Close")] = [11.0, 12.0]
            ohlcv_frame[(ticker, "Volume")] = [1000.0, 1100.0]
        ohlcv_frame.columns = pd.MultiIndex.from_tuples(ohlcv_frame.columns)
        prices = ohlcv_frame.xs("Close", axis=1, level=1)
        prices.to_parquet(output_path)
        ohlcv_frame.to_parquet(output_path.with_suffix(".ohlcv.parquet"))
        coverage = pd.DataFrame({"ticker": tickers, "rows": [2] * len(tickers), "start": start, "end": end})
        coverage.to_csv(output_path.with_suffix(".coverage.csv"), index=False)
        return DownloadResult(prices=prices, coverage=coverage, failures={}, ohlcv=ohlcv_frame)

    first = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=flaky_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
        max_plan_rows=1,
    )
    second = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=flaky_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
        max_plan_rows=1,
    )

    assert len(calls) == 2
    assert first.manifest["status"].tolist() == ["failed_empty"]
    assert second.manifest["status"].tolist() == ["completed"]
    stored = pd.read_parquet(second.long_ohlcv_root)
    assert stored.shape[0] == 4


def test_execute_refinitiv_historical_plan_accepts_proven_prelisting_empty_window(tmp_path):
    universe = refinitiv_universe_from_symbol_map(
        _symbol_map(tmp_path),
        include_countries=parse_country_filter("Malaysia"),
    ).head(1)
    universe["first_trade_date"] = "2012-03-29"
    plan = build_refinitiv_historical_plan(
        universe,
        root=tmp_path / "lake",
        run_id="RUN",
        start_date="2006-01-01",
        end_date="2010-12-31",
        batch_size=1,
        window_years=5,
    )
    calls = 0

    def empty_downloader(tickers, start, end, interval, output_path, ohlcv, field_map=None):
        nonlocal calls
        calls += 1
        return DownloadResult(
            prices=pd.DataFrame(),
            coverage=pd.DataFrame(),
            failures={ticker: "empty history" for ticker in tickers},
            ohlcv=pd.DataFrame(),
        )

    first = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=empty_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
    )
    second = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=empty_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
    )

    assert calls == 1
    assert first.manifest.loc[0, "status"] == "completed_prelisting"
    assert first.manifest.loc[0, "structural_reason"] == "all_requested_tickers_first_trade_after_window"
    assert first.manifest.loc[0, "first_trade_date_min"] == "2012-03-29"
    assert second.manifest.loc[0, "status"] == "completed_prelisting"
    run_manifest = json.loads((tmp_path / "artifacts" / "RUN" / "historical_run_manifest.json").read_text())
    assert run_manifest["completed_rows"] == 1
    assert run_manifest["failed_rows"] == 0


def test_execute_refinitiv_historical_plan_halts_on_retryable_provider_error(tmp_path):
    universe = refinitiv_universe_from_symbol_map(
        _symbol_map(tmp_path),
        include_countries=parse_country_filter("Malaysia"),
    )
    plan = build_refinitiv_historical_plan(
        universe,
        root=tmp_path / "lake",
        run_id="RUN",
        start_date="2020-01-01",
        end_date="2021-12-31",
        batch_size=1,
        window_years=1,
    )
    calls = 0

    def throttled_downloader(**kwargs):
        nonlocal calls
        calls += 1
        raise RefinitivRetryableProviderError("429 Too many requests")

    result = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=throttled_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
    )

    assert calls == 1
    assert result.manifest["status"].tolist() == ["failed_retryable_provider"]
    run_manifest = json.loads((tmp_path / "artifacts" / "RUN" / "historical_run_manifest.json").read_text())
    assert run_manifest["halted_early"] is True
    assert run_manifest["provider_halt"]["plan_row_id"] == plan.iloc[0]["plan_row_id"]


def test_execute_refinitiv_historical_plan_halts_on_orphan_long_payload(tmp_path):
    universe = refinitiv_universe_from_symbol_map(
        _symbol_map(tmp_path),
        include_countries=parse_country_filter("Malaysia"),
    ).head(1)
    plan = build_refinitiv_historical_plan(
        universe,
        root=tmp_path / "lake",
        run_id="RUN",
        start_date="2020-01-01",
        end_date="2020-12-31",
        batch_size=1,
        window_years=1,
    )
    orphan_root = tmp_path / "lake" / "prices_long" / "RUN" / "ohlcv"
    orphan_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-02")],
            "ticker": ["MBBM.KL"],
            "batch_id": [plan.iloc[0]["plan_row_id"]],
        }
    ).to_parquet(orphan_root / "orphan.parquet", index=False)
    calls = 0

    def downloader(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("orphan guard must run before the downloader")

    result = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
    )

    assert calls == 0
    assert result.manifest.empty
    run_manifest = json.loads((tmp_path / "artifacts" / "RUN" / "historical_run_manifest.json").read_text())
    assert run_manifest["halted_early"] is True
    assert run_manifest["integrity_halt"]["error_type"] == "OrphanLongPayload"
    assert run_manifest["integrity_halt"]["orphan_plan_row_ids"] == [plan.iloc[0]["plan_row_id"]]


def test_clip_long_to_listing_windows_respects_closed_and_right_open_identity():
    long = pd.DataFrame(
        {
            "ticker": ["OLD.T", "OLD.T", "LIVE.T", "LIVE.T"],
            "date": pd.to_datetime(["2019-12-31", "2020-06-30", "2026-06-30", "2026-07-02"]),
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    )
    listings = pd.DataFrame(
        {
            "refinitiv_ric": ["OLD.T", "LIVE.T"],
            "listing_start": ["2020-01-01", "2020-01-01"],
            "listing_end": ["2020-12-31", "2026-06-30"],
            "right_open": [False, True],
        }
    )

    clipped = _clip_long_to_listing_windows(long, listings)

    assert clipped["ticker"].tolist() == ["OLD.T", "LIVE.T", "LIVE.T"]
    assert clipped["date"].tolist() == [
        pd.Timestamp("2020-06-30"),
        pd.Timestamp("2026-06-30"),
        pd.Timestamp("2026-07-02"),
    ]


def test_historical_plan_assigns_each_ric_window_to_one_batch(tmp_path):
    universe = pd.DataFrame(
        {
            "country": ["Japan", "Japan", "Japan"],
            "exchange": ["JPX", "JPX", "JPX"],
            "ticker": ["OLD", "NEW", "OTHER"],
            "refinitiv_ric": ["SAME.T", "SAME.T", "OTHER.T"],
        }
    )

    plan = build_refinitiv_historical_plan(
        universe,
        root=tmp_path / "lake",
        run_id="RUN",
        start_date="2020-01-01",
        end_date="2020-12-31",
        batch_size=1,
        window_years=1,
    )

    owners = [ticker for value in plan["tickers"] for ticker in value.split(",")]
    assert sorted(owners) == ["OTHER.T", "SAME.T"]


def test_execute_refinitiv_historical_plan_accepts_proven_postretirement_empty_window(tmp_path):
    universe = refinitiv_universe_from_symbol_map(
        _symbol_map(tmp_path),
        include_countries=parse_country_filter("Malaysia"),
    ).head(1)
    universe["retire_date"] = "2020-05-11"
    plan = build_refinitiv_historical_plan(
        universe,
        root=tmp_path / "lake",
        run_id="RUN",
        start_date="2021-01-01",
        end_date="2025-12-31",
        batch_size=1,
        window_years=5,
    )
    calls = 0

    def empty_downloader(tickers, start, end, interval, output_path, ohlcv, field_map=None):
        nonlocal calls
        calls += 1
        return DownloadResult(
            prices=pd.DataFrame(),
            coverage=pd.DataFrame(),
            failures={ticker: "empty history" for ticker in tickers},
            ohlcv=pd.DataFrame(),
        )

    first = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=empty_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
    )
    execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=empty_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
    )

    assert calls == 1
    assert first.manifest.loc[0, "status"] == "completed_postretirement"
    assert first.manifest.loc[0, "structural_reason"] == "all_requested_tickers_retired_before_window"
    assert first.manifest.loc[0, "retire_date_max"] == "2020-05-11"


def test_execute_refinitiv_historical_plan_can_quarantine_rows(tmp_path):
    universe = refinitiv_universe_from_symbol_map(
        _symbol_map(tmp_path),
        include_countries=parse_country_filter("Malaysia"),
    )
    plan = build_refinitiv_historical_plan(
        universe,
        root=tmp_path / "lake",
        run_id="RUN",
        start_date="2020-01-01",
        end_date="2021-12-31",
        batch_size=2,
        window_years=1,
    )
    calls: list[tuple[str, str]] = []

    def fake_downloader(tickers, start, end, interval, output_path, ohlcv, field_map=None):
        calls.append((start, end))
        dates = pd.date_range(start, periods=2)
        ohlcv_frame = pd.DataFrame(index=dates)
        for ticker in tickers:
            ohlcv_frame[(ticker, "Open")] = [10.0, 11.0]
            ohlcv_frame[(ticker, "High")] = [12.0, 13.0]
            ohlcv_frame[(ticker, "Low")] = [9.0, 10.0]
            ohlcv_frame[(ticker, "Close")] = [11.0, 12.0]
            ohlcv_frame[(ticker, "Volume")] = [1000.0, 1100.0]
        ohlcv_frame.columns = pd.MultiIndex.from_tuples(ohlcv_frame.columns)
        prices = ohlcv_frame.xs("Close", axis=1, level=1)
        prices.to_parquet(output_path)
        ohlcv_frame.to_parquet(output_path.with_suffix(".ohlcv.parquet"))
        coverage = pd.DataFrame({"ticker": tickers, "rows": [2] * len(tickers), "start": start, "end": end})
        coverage.to_csv(output_path.with_suffix(".coverage.csv"), index=False)
        return DownloadResult(prices=prices, coverage=coverage, failures={}, ohlcv=ohlcv_frame)

    result = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=fake_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
        max_plan_rows=1,
        exclude_plan_row_ids={str(plan.iloc[0]["plan_row_id"])},
    )
    run_manifest = pd.read_json(result.artifact_dir / "historical_run_manifest.json", typ="series")

    assert calls == [("2021-01-01", "2021-12-31")]
    assert result.manifest["plan_row_id"].tolist() == [str(plan.iloc[1]["plan_row_id"])]
    assert int(run_manifest["excluded_rows"]) == 1


def test_execute_refinitiv_historical_plan_can_retry_only_included_rows(tmp_path):
    universe = refinitiv_universe_from_symbol_map(
        _symbol_map(tmp_path),
        include_countries=parse_country_filter("Malaysia"),
    )
    plan = build_refinitiv_historical_plan(
        universe,
        root=tmp_path / "lake",
        run_id="RUN",
        start_date="2020-01-01",
        end_date="2022-12-31",
        batch_size=2,
        window_years=1,
    )
    target_plan_row_id = str(plan.iloc[1]["plan_row_id"])
    calls: list[tuple[str, str]] = []

    def fake_downloader(tickers, start, end, interval, output_path, ohlcv, field_map=None):
        calls.append((start, end))
        dates = pd.date_range(start, periods=2)
        ohlcv_frame = pd.DataFrame(index=dates)
        for ticker in tickers:
            ohlcv_frame[(ticker, "Open")] = [10.0, 11.0]
            ohlcv_frame[(ticker, "High")] = [12.0, 13.0]
            ohlcv_frame[(ticker, "Low")] = [9.0, 10.0]
            ohlcv_frame[(ticker, "Close")] = [11.0, 12.0]
            ohlcv_frame[(ticker, "Volume")] = [1000.0, 1100.0]
        ohlcv_frame.columns = pd.MultiIndex.from_tuples(ohlcv_frame.columns)
        prices = ohlcv_frame.xs("Close", axis=1, level=1)
        prices.to_parquet(output_path)
        ohlcv_frame.to_parquet(output_path.with_suffix(".ohlcv.parquet"))
        coverage = pd.DataFrame({"ticker": tickers, "rows": [2] * len(tickers), "start": start, "end": end})
        coverage.to_csv(output_path.with_suffix(".coverage.csv"), index=False)
        return DownloadResult(prices=prices, coverage=coverage, failures={}, ohlcv=ohlcv_frame)

    result = execute_refinitiv_historical_plan(
        plan=plan,
        root=tmp_path / "lake",
        run_id="RUN",
        downloader=fake_downloader,
        universe=universe,
        artifact_dir=tmp_path / "artifacts" / "RUN",
        include_plan_row_ids={target_plan_row_id},
    )
    run_manifest = pd.read_json(result.artifact_dir / "historical_run_manifest.json", typ="series")

    assert calls == [("2021-01-01", "2021-12-31")]
    assert result.manifest["plan_row_id"].tolist() == [target_plan_row_id]
    assert int(run_manifest["included_rows"]) == 1
