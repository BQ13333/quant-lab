from __future__ import annotations

import json

import pandas as pd

from quant_lab.data.fx_country_risk import build_currency_reference, write_fx_country_risk_artifacts


def _daily_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 3),
            "ticker": ["AAA.KL", "AAA.KL", "BBB.SI", "BBB.SI", "CCC.JK", "CCC.JK"],
            "country": ["Malaysia", "Malaysia", "Singapore", "Singapore", "Indonesia", "Indonesia"],
            "exchange": ["Bursa", "Bursa", "SGX", "SGX", "IDX", "IDX"],
            "close": [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        }
    )


def test_build_currency_reference_covers_active_u2_countries():
    ref = build_currency_reference(_daily_features())

    assert dict(zip(ref["country"], ref["currency"], strict=True)) == {
        "Indonesia": "IDR",
        "Malaysia": "MYR",
        "Singapore": "SGD",
    }
    assert ref["security_coverage"].min() == 1.0


def test_write_fx_country_risk_artifacts_with_injected_downloader(tmp_path):
    daily_path = tmp_path / "daily.parquet"
    _daily_features().to_parquet(daily_path, index=False)

    def fake_downloader(tickers, start_date, end_date):
        dates = pd.date_range(start_date, end_date, freq="D")
        return pd.DataFrame({ticker: [1.0 + idx for idx, _ in enumerate(dates)] for ticker in tickers}, index=dates)

    artifacts = write_fx_country_risk_artifacts(
        daily_features_path=daily_path,
        artifact_root=tmp_path / "fx",
        run_id="FX-TEST",
        start_date="2024-01-01",
        end_date="2024-01-03",
        downloader=fake_downloader,
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert artifacts.currency_reference_path.exists()
    assert artifacts.fx_rates_path.exists()
    assert artifacts.local_usd_returns_path.exists()
    assert artifacts.policy_path.exists()
    assert manifest["currency_coverage"] == 1.0
    assert manifest["fx_calendar_coverage"] == 1.0
    assert manifest["usd_return_coverage"] == 1.0
    assert manifest["status"] == "pass_currency_fx_returns_partial_tca"
