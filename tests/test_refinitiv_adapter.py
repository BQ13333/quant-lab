from __future__ import annotations

import pandas as pd

from quant_lab.data.vendor_adapters import (
    RefinitivAdapter,
    RefinitivRetryableProviderError,
    refinitiv_history_to_ohlcv,
)


class FakeLsegData:
    def __init__(self, history: pd.DataFrame):
        self.history = history
        self.open_calls: list[dict] = []
        self.get_history_calls: list[dict] = []
        self.closed = False

    def open_session(self, **kwargs):
        self.open_calls.append(kwargs)

    def close_session(self):
        self.closed = True

    def get_history(self, **kwargs):
        self.get_history_calls.append(kwargs)
        return self.history


class FallbackFakeLsegData:
    def __init__(self, histories: dict[str, pd.DataFrame]):
        self.histories = histories
        self.open_calls: list[dict] = []
        self.get_history_calls: list[dict] = []

    def open_session(self, **kwargs):
        self.open_calls.append(kwargs)

    def close_session(self):
        pass

    def get_history(self, **kwargs):
        self.get_history_calls.append(kwargs)
        universe = kwargs["universe"]
        if len(universe) > 1:
            raise TypeError("'UniverseContainer' object is not subscriptable")
        ticker = universe[0]
        if ticker not in self.histories:
            raise RuntimeError("not found")
        return self.histories[ticker]


class ErrorFakeLsegData:
    def open_session(self, **kwargs):
        pass

    def close_session(self):
        pass

    def get_history(self, **kwargs):
        raise RuntimeError("429 Too many requests")


def test_refinitiv_history_to_ohlcv_multiindex():
    dates = pd.date_range("2024-01-01", periods=2)
    history = pd.DataFrame(
        {
            ("AAA.N", "OPEN_PRC"): [10.0, 11.0],
            ("AAA.N", "HIGH_1"): [12.0, 13.0],
            ("AAA.N", "LOW_1"): [9.0, 10.0],
            ("AAA.N", "TRDPRC_1"): [11.0, 12.0],
            ("AAA.N", "ACVOL_UNS"): [100, 110],
            ("BBB.N", "TRDPRC_1"): [20.0, 21.0],
        },
        index=dates,
    )

    ohlcv = refinitiv_history_to_ohlcv(history, tickers=["AAA.N", "BBB.N"])

    assert ("AAA.N", "Open") in ohlcv.columns
    assert ("AAA.N", "Close") in ohlcv.columns
    assert ("AAA.N", "Volume") in ohlcv.columns
    assert ("BBB.N", "Close") in ohlcv.columns
    assert ohlcv.loc[dates[0], ("AAA.N", "Close")] == 11.0


def test_refinitiv_history_to_ohlcv_single_ticker_flat_columns():
    dates = pd.date_range("2024-01-01", periods=2)
    history = pd.DataFrame(
        {
            "OPEN_PRC": [10.0, 11.0],
            "TRDPRC_1": [11.0, 12.0],
        },
        index=dates,
    )

    ohlcv = refinitiv_history_to_ohlcv(history, tickers=["AAA.N"])

    assert list(ohlcv.columns) == [("AAA.N", "Open"), ("AAA.N", "Close")]


def test_refinitiv_history_to_ohlcv_accepts_display_field_names():
    dates = pd.date_range("2024-01-01", periods=2)
    history = pd.DataFrame(
        {
            "Price Close": [11.0, 12.0],
            "Accumulated Volume": [100, 110],
        },
        index=dates,
    )

    ohlcv = refinitiv_history_to_ohlcv(
        history,
        tickers=["AAA.N"],
        field_map={"Close": "TR.PriceClose", "Volume": "TR.AccumulatedVolume"},
    )

    assert list(ohlcv.columns) == [("AAA.N", "Close"), ("AAA.N", "Volume")]
    assert ohlcv.loc[dates[1], ("AAA.N", "Close")] == 12.0


def test_refinitiv_adapter_download_uses_local_session_config():
    dates = pd.date_range("2024-01-01", periods=2)
    history = pd.DataFrame({("AAA.N", "TRDPRC_1"): [11.0, 12.0]}, index=dates)
    fake = FakeLsegData(history)
    adapter = RefinitivAdapter(
        session_name="desktop.workspace",
        config_name="/private/lseg-data.config.json",
        library=fake,
    )

    result = adapter.download(
        tickers=["AAA.N"],
        start="2024-01-01",
        end="2024-01-31",
        fields=["TRDPRC_1"],
    )

    assert result.equals(history)
    assert fake.open_calls == [{"name": "desktop.workspace", "config_name": "/private/lseg-data.config.json"}]
    assert fake.get_history_calls[0]["universe"] == ["AAA.N"]
    assert fake.get_history_calls[0]["fields"] == ["TRDPRC_1"]


def test_refinitiv_adapter_health_check_retains_last_error():
    adapter = RefinitivAdapter(library=ErrorFakeLsegData())

    assert adapter.health_check() is False
    assert "RuntimeError" in adapter.last_health_error
    assert "429 Too many requests" in adapter.last_health_error


def test_refinitiv_adapter_rate_limit_does_not_fan_out_to_single_tickers():
    fake = ErrorFakeLsegData()
    adapter = RefinitivAdapter(library=fake)

    try:
        adapter.download_price_history(
            tickers=["AAA.N", "BBB.N"],
            period="2024-01-01:2024-01-31",
        )
    except RefinitivRetryableProviderError as exc:
        assert "429 Too many requests" in str(exc)
    else:
        raise AssertionError("Expected a retryable provider failure")


def test_refinitiv_adapter_download_price_history_shapes_download_result(tmp_path):
    dates = pd.date_range("2024-01-01", periods=2)
    history = pd.DataFrame(
        {
            ("AAA.N", "OPEN_PRC"): [10.0, 11.0],
            ("AAA.N", "HIGH_1"): [12.0, 13.0],
            ("AAA.N", "LOW_1"): [9.0, 10.0],
            ("AAA.N", "TRDPRC_1"): [11.0, 12.0],
            ("AAA.N", "ACVOL_UNS"): [100, 110],
        },
        index=dates,
    )
    fake = FakeLsegData(history)
    adapter = RefinitivAdapter(library=fake)
    output_path = tmp_path / "prices.parquet"

    result = adapter.download_price_history(
        tickers=["AAA.N"],
        period="1y",
        interval="1d",
        output_path=output_path,
        ohlcv=True,
    )

    assert list(result.prices.columns) == ["AAA.N"]
    assert ("AAA.N", "Close") in result.ohlcv.columns
    assert result.coverage.loc[result.coverage["ticker"] == "AAA.N", "rows"].iloc[0] == 2
    assert result.failures == {}
    assert output_path.exists()
    assert output_path.with_suffix(".ohlcv.parquet").exists()
    assert output_path.with_suffix(".coverage.csv").exists()


def test_refinitiv_adapter_download_price_history_accepts_field_map_override(tmp_path):
    dates = pd.date_range("2024-01-01", periods=2)
    history = pd.DataFrame(
        {
            ("0700.HK", "Price Close"): [300.0, 301.0],
        },
        index=dates,
    )
    fake = FakeLsegData(history)
    adapter = RefinitivAdapter(library=fake)

    result = adapter.download_price_history(
        tickers=["0700.HK"],
        period="1mo",
        interval="1d",
        output_path=tmp_path / "prices.parquet",
        ohlcv=True,
        field_map={"Close": "TR.CLOSEPRICE"},
    )

    assert fake.get_history_calls[0]["fields"] == ["TR.CLOSEPRICE"]
    assert result.prices.loc[dates[1], "0700.HK"] == 301.0


def test_refinitiv_adapter_download_price_history_range_uses_explicit_dates(tmp_path):
    dates = pd.date_range("2010-01-01", periods=2)
    history = pd.DataFrame(
        {
            ("AAA.N", "TRDPRC_1"): [10.0, 11.0],
        },
        index=dates,
    )
    fake = FakeLsegData(history)
    adapter = RefinitivAdapter(library=fake)

    result = adapter.download_price_history_range(
        tickers=["AAA.N"],
        start="2006-01-01",
        end="2010-12-31",
        interval="1d",
        output_path=tmp_path / "prices.parquet",
        ohlcv=True,
        field_map={"Close": "TRDPRC_1"},
    )

    assert fake.get_history_calls[0]["start"] == "2006-01-01"
    assert fake.get_history_calls[0]["end"] == "2010-12-31"
    assert result.coverage.loc[result.coverage["ticker"] == "AAA.N", "period_used"].iloc[0] == "2006-01-01:2010-12-31"
    assert result.prices.loc[dates[1], "AAA.N"] == 11.0


def test_refinitiv_adapter_download_price_history_falls_back_after_batch_error(tmp_path):
    dates = pd.date_range("2024-01-01", periods=2)
    fake = FallbackFakeLsegData(
        {
            "AAA.N": pd.DataFrame({"TRDPRC_1": [10.0, 11.0]}, index=dates),
            "BBB.N": pd.DataFrame({"TRDPRC_1": [20.0, 21.0]}, index=dates),
        }
    )
    adapter = RefinitivAdapter(library=fake)

    result = adapter.download_price_history(
        tickers=["AAA.N", "BBB.N", "BAD.N"],
        period="1mo",
        interval="1d",
        output_path=tmp_path / "prices.parquet",
        ohlcv=True,
        field_map={"Close": "TRDPRC_1"},
    )

    assert list(result.prices.columns) == ["AAA.N", "BBB.N"]
    assert result.coverage.loc[result.coverage["ticker"] == "BAD.N", "rows"].iloc[0] == 0
    assert result.failures["BAD.N"] == "empty history"
    assert fake.get_history_calls[0]["universe"] == ["AAA.N", "BBB.N", "BAD.N"]
