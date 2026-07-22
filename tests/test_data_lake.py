from __future__ import annotations

import pandas as pd

from quant_lab.data.data_lake import (
    _slug,
    backfill_prices_in_batches,
    iter_listing_batches,
    wide_ohlcv_to_long,
    write_ohlcv_long_partitions,
    write_universe_partitions,
)
from quant_lab.data.yfinance_loader import DownloadResult


def _listings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "yahoo_ticker": ["A", "B", "C", "D", "E"],
            "country": ["US", "US", "JP", "JP", "JP"],
            "exchange": ["NYSE", "NYSE", "JPX", "JPX", "JPX"],
        }
    )


def test_iter_listing_batches_groups_and_bounds():
    batches = list(iter_listing_batches(_listings(), batch_size=2))

    assert len(batches) == 3
    assert batches[0][0]["country"] == "JP"
    assert batches[0][0]["rows"] == 2
    assert batches[-1][1].shape[0] == 2


def test_write_universe_partitions(tmp_path):
    path = write_universe_partitions(_listings(), tmp_path, "TEST")

    assert path.exists()
    assert path.parent.joinpath("schema.csv").exists()
    assert path.parent.joinpath("partitions.csv").exists()
    assert list(path.glob("country_partition=*/exchange_partition=*/*.parquet"))


def test_write_universe_partitions_requires_overwrite_for_existing_path(tmp_path):
    write_universe_partitions(_listings(), tmp_path, "TEST")

    try:
        write_universe_partitions(_listings(), tmp_path, "TEST")
    except FileExistsError as exc:
        assert "overwrite=True" in str(exc)
    else:
        raise AssertionError("expected FileExistsError")

    path = write_universe_partitions(_listings().head(2), tmp_path, "TEST", overwrite=True)

    assert path.exists()


def test_slug_keeps_non_ascii_exchange_paths_distinct():
    assert _slug("KRX") == "krx"
    assert _slug("KRX 유가").startswith("krx_")
    assert _slug("KRX 코스닥").startswith("krx_")
    assert len({_slug("KRX"), _slug("KRX 유가"), _slug("KRX 코스닥")}) == 3
    assert _slug("Hong Kong") == "hong_kong"


def _ohlcv_frame(tickers: list[str]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=2)
    data = {}
    for ticker in tickers:
        data[(ticker, "Open")] = [10.0, 11.0]
        data[(ticker, "High")] = [12.0, 13.0]
        data[(ticker, "Low")] = [9.0, 10.0]
        data[(ticker, "Close")] = [11.0, 12.0]
        data[(ticker, "Volume")] = [1000, 1100]
    return pd.DataFrame(data, index=index)


def test_wide_ohlcv_to_long_preserves_metadata():
    listings = _listings().assign(currency=["USD", "USD", "JPY", "JPY", "JPY"])
    long = wide_ohlcv_to_long(
        _ohlcv_frame(["A", "C"]),
        listings=listings,
        provider="mock",
        interval="1d",
        adjustment="raw",
        data_tier="bronze",
        run_id="RUN",
        batch_id="BATCH",
        fetched_at_utc="2026-07-05T00:00:00+00:00",
    )

    assert long.shape[0] == 4
    assert set(["date", "ticker", "sid", "local_symbol", "country", "exchange", "currency"]).issubset(long.columns)
    assert long.loc[long["ticker"] == "A", "country"].iloc[0] == "US"
    assert long.loc[long["ticker"] == "C", "currency"].iloc[0] == "JPY"
    assert long["adj_close"].equals(long["close"])
    assert set(long["provider"]) == {"mock"}
    assert set(long["data_tier"]) == {"bronze"}
    assert set(long["adjustment"]) == {"raw"}
    assert set(long["batch_id"]) == {"BATCH"}


def test_wide_ohlcv_to_long_drops_rows_without_close():
    index = pd.date_range("2024-01-01", periods=2)
    ohlcv = pd.DataFrame(
        {
            ("A", "Volume"): [1000, 1100],
        },
        index=index,
    )

    long = wide_ohlcv_to_long(ohlcv, listings=_listings(), provider="mock")

    assert long.empty


def test_write_ohlcv_long_partitions(tmp_path):
    long = wide_ohlcv_to_long(
        _ohlcv_frame(["A"]),
        listings=_listings(),
        provider="mock",
        interval="1d",
        adjustment="raw",
        run_id="RUN",
    )

    path = write_ohlcv_long_partitions(long, root=tmp_path, run_id="RUN")

    assert path.exists()
    assert path.parent.joinpath("ohlcv_schema.csv").exists()
    assert path.parent.joinpath("ohlcv_partitions.csv").exists()
    assert list(path.glob("provider_partition=mock/interval_partition=1d/adjustment_partition=raw/**/*.parquet"))
    stored = pd.read_parquet(path)
    assert stored.shape[0] == 2
    assert set(stored["ticker"]) == {"A"}


def test_write_ohlcv_long_partitions_reads_mixed_null_metadata(tmp_path):
    null_name_listing = pd.DataFrame(
        {
            "ticker": ["A"],
            "yahoo_ticker": ["A"],
            "name": [pd.NA],
            "country": ["US"],
            "exchange": ["NYSE"],
            "currency": ["USD"],
        }
    )
    named_listing = pd.DataFrame(
        {
            "ticker": ["C"],
            "yahoo_ticker": ["C"],
            "name": ["Company C"],
            "country": ["JP"],
            "exchange": ["JPX"],
            "currency": ["JPY"],
        }
    )
    first = wide_ohlcv_to_long(_ohlcv_frame(["A"]), listings=null_name_listing, provider="mock", run_id="RUN")
    second = wide_ohlcv_to_long(_ohlcv_frame(["C"]), listings=named_listing, provider="mock", run_id="RUN")

    path = write_ohlcv_long_partitions(first, root=tmp_path, run_id="RUN")
    write_ohlcv_long_partitions(second, root=tmp_path, run_id="RUN")

    stored = pd.read_parquet(path)
    assert stored.shape[0] == 4
    assert set(stored["ticker"]) == {"A", "C"}
    assert "name" in stored.columns


def test_write_ohlcv_long_partitions_reads_mixed_numeric_volume_types(tmp_path):
    first = wide_ohlcv_to_long(_ohlcv_frame(["A"]), listings=_listings(), provider="mock", run_id="RUN")
    second = wide_ohlcv_to_long(_ohlcv_frame(["C"]), listings=_listings(), provider="mock", run_id="RUN")
    second["volume"] = second["volume"] + 0.5

    path = write_ohlcv_long_partitions(first, root=tmp_path, run_id="RUN")
    write_ohlcv_long_partitions(second, root=tmp_path, run_id="RUN")

    stored = pd.read_parquet(path)
    assert stored.shape[0] == 4
    assert stored["volume"].dtype == "float64"


def test_backfill_prices_in_batches_uses_bounded_batches(tmp_path):
    calls = []

    def fake_downloader(tickers, period, interval, output_path, fallback_periods, ohlcv):
        calls.append(list(tickers))
        prices = pd.DataFrame({ticker: [1.0, 2.0] for ticker in tickers}, index=pd.date_range("2024-01-01", periods=2))
        prices.to_parquet(output_path)
        coverage = pd.DataFrame({"ticker": tickers, "rows": [2] * len(tickers)})
        coverage.to_csv(output_path.with_suffix(".coverage.csv"), index=False)
        return DownloadResult(prices=prices, coverage=coverage, failures={})

    result = backfill_prices_in_batches(
        _listings(),
        root=tmp_path,
        run_id="RUN",
        batch_size=2,
        max_tickers=4,
        downloader=fake_downloader,
    )

    assert result.manifest_path.exists()
    assert len(calls) == 3
    assert all(len(call) <= 2 for call in calls)
    assert sum(len(call) for call in calls) == 4
    assert result.manifest["price_columns"].sum() == 4


def test_backfill_prices_in_batches_passes_batch_download_kwargs(tmp_path):
    received = []

    def fake_downloader(tickers, period, interval, output_path, fallback_periods, ohlcv, field_map=None):
        received.append((list(tickers), field_map))
        prices = pd.DataFrame({ticker: [1.0] for ticker in tickers}, index=pd.date_range("2024-01-01", periods=1))
        prices.to_parquet(output_path)
        coverage = pd.DataFrame({"ticker": tickers, "rows": [1] * len(tickers)})
        coverage.to_csv(output_path.with_suffix(".coverage.csv"), index=False)
        return DownloadResult(prices=prices, coverage=coverage, failures={})

    backfill_prices_in_batches(
        _listings().head(2),
        root=tmp_path,
        run_id="RUN",
        batch_size=2,
        downloader=fake_downloader,
        download_kwargs_by_batch=lambda meta, batch: {"field_map": {"Close": f"{meta['exchange']}_CLOSE"}},
    )

    assert received == [(["A", "B"], {"Close": "NYSE_CLOSE"})]


def test_backfill_prices_in_batches_writes_long_ohlcv_when_requested(tmp_path):
    calls = []

    def fake_downloader(tickers, period, interval, output_path, fallback_periods, ohlcv):
        calls.append(list(tickers))
        prices = pd.DataFrame({ticker: [1.0, 2.0] for ticker in tickers}, index=pd.date_range("2024-01-01", periods=2))
        prices.to_parquet(output_path)
        coverage = pd.DataFrame({"ticker": tickers, "rows": [2] * len(tickers)})
        coverage.to_csv(output_path.with_suffix(".coverage.csv"), index=False)
        ohlcv_frame = _ohlcv_frame(tickers)
        ohlcv_frame.to_parquet(output_path.with_suffix(".ohlcv.parquet"))
        return DownloadResult(prices=prices, coverage=coverage, failures={}, ohlcv=ohlcv_frame)

    result = backfill_prices_in_batches(
        _listings().head(2).assign(currency=["USD", "USD"]),
        root=tmp_path,
        run_id="RUN",
        batch_size=2,
        downloader=fake_downloader,
        provider="mock",
        adjustment="raw",
        data_tier="bronze",
        write_long_ohlcv=True,
    )

    assert result.long_ohlcv_root is not None
    assert result.long_ohlcv_root.exists()
    assert result.manifest["long_ohlcv_rows"].sum() == 4
    assert result.manifest["long_ohlcv_path"].notna().all()
    stored = pd.read_parquet(result.long_ohlcv_root)
    assert stored.shape[0] == 4
    assert set(stored["provider"]) == {"mock"}
    assert set(stored["data_tier"]) == {"bronze"}
    assert set(stored["country"]) == {"US"}
    assert set(result.manifest["data_tier"]) == {"bronze"}
