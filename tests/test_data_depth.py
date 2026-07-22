from __future__ import annotations

import pandas as pd

from quant_lab.data.data_depth import build_data_depth_tables, load_long_ohlcv, write_data_depth_report
from quant_lab.data.data_lake import wide_ohlcv_to_long, write_ohlcv_long_partitions


def _ohlcv_frame(tickers: list[str]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3)
    data = {}
    for ticker in tickers:
        data[(ticker, "Open")] = [10.0, 11.0, 12.0]
        data[(ticker, "High")] = [12.0, 13.0, 14.0]
        data[(ticker, "Low")] = [9.0, 10.0, 11.0]
        data[(ticker, "Close")] = [11.0, 12.0, 13.0]
        data[(ticker, "Volume")] = [1000.0, 0.0, None]
    return pd.DataFrame(data, index=index)


def _listings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "yahoo_ticker": ["AAA.KL", "BBB.SI"],
            "country": ["Malaysia", "Singapore"],
            "exchange": ["Bursa Malaysia", "SGX"],
            "currency": ["MYR", "SGD"],
            "name": ["AAA Bhd", "BBB Ltd"],
        }
    )


def test_build_data_depth_tables_summarizes_long_ohlcv():
    long = wide_ohlcv_to_long(_ohlcv_frame(["AAA.KL", "BBB.SI"]), listings=_listings(), provider="mock")

    overall, country_summary, ticker_summary = build_data_depth_tables(long, long_ohlcv_root="lake")

    assert overall.loc[0, "rows"] == 6
    assert overall.loc[0, "tickers"] == 2
    assert overall.loc[0, "countries"] == 2
    assert set(country_summary["country"]) == {"Malaysia", "Singapore"}
    assert set(ticker_summary["rows"]) == {3}
    assert ticker_summary["zero_volume_pct"].gt(0).all()


def test_write_data_depth_report_reads_partitioned_lake(tmp_path):
    long = wide_ohlcv_to_long(_ohlcv_frame(["AAA.KL"]), listings=_listings().head(1), provider="mock", run_id="RUN")
    root = write_ohlcv_long_partitions(long, root=tmp_path / "lake", run_id="RUN")

    loaded = load_long_ohlcv(root)
    report = write_data_depth_report(root, artifact_root=tmp_path / "artifacts", run_id="DEPTH")

    assert loaded.shape[0] == 3
    assert report.manifest_path.exists()
    assert report.country_summary_path.exists()
    assert report.ticker_summary_path.exists()
    assert "Rows: 3" in report.markdown_path.read_text(encoding="utf-8")
