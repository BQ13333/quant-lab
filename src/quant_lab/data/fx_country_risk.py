from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd
import yfinance as yf

from quant_lab.data.fx import FX_TICKERS


ASEAN_U2_CURRENCY_REFERENCE = {
    "Indonesia": {"currency": "IDR", "exchange_pattern": "Indonesia Stock Exchange", "source": "exchange_reference_policy"},
    "Malaysia": {"currency": "MYR", "exchange_pattern": "Kuala Lumpur Stock Exchange", "source": "exchange_reference_policy"},
    "Singapore": {"currency": "SGD", "exchange_pattern": "Singapore Exchange", "source": "exchange_reference_policy"},
}


@dataclass(frozen=True)
class FXCountryRiskArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    currency_reference_path: Path
    fx_rates_path: Path
    local_usd_returns_path: Path
    fx_quality_path: Path
    policy_path: Path
    currency_reference: pd.DataFrame
    fx_rates: pd.DataFrame
    local_usd_returns: pd.DataFrame
    fx_quality: pd.DataFrame
    policy: pd.DataFrame


def write_fx_country_risk_artifacts(
    daily_features_path: str | Path,
    artifact_root: str | Path,
    run_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    downloader: Callable[[Sequence[str], str, str], pd.DataFrame] | None = None,
) -> FXCountryRiskArtifacts:
    daily = pd.read_parquet(daily_features_path, columns=["date", "ticker", "country", "exchange", "close"])
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    if start_date is None:
        start_date = str(daily["date"].min().date())
    if end_date is None:
        end_date = str(daily["date"].max().date())

    currency_reference = build_currency_reference(daily)
    currencies = sorted(set(currency_reference["currency"].dropna().astype(str)))
    fx_tickers = [FX_TICKERS[currency] for currency in currencies if currency in FX_TICKERS]
    downloader = downloader or download_yfinance_fx_rates
    fx_rates = downloader(fx_tickers, start_date, end_date) if fx_tickers else pd.DataFrame()
    fx_rates = _normalize_fx_frame(fx_rates)
    local_usd_returns = build_local_usd_return_paths(daily, currency_reference, fx_rates)
    fx_quality = build_fx_quality(fx_rates, fx_tickers=fx_tickers, start_date=start_date, end_date=end_date)
    policy = build_country_policy_table(currency_reference)

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    currency_reference_path = artifact_dir / "currency_reference.csv"
    fx_rates_path = artifact_dir / "fx_rates.parquet"
    local_usd_returns_path = artifact_dir / "local_usd_returns.parquet"
    fx_quality_path = artifact_dir / "fx_quality.csv"
    policy_path = artifact_dir / "country_risk_tca_policy.csv"
    manifest_path = artifact_dir / "manifest.json"

    currency_reference.to_csv(currency_reference_path, index=False)
    fx_rates.to_parquet(fx_rates_path, index=True)
    local_usd_returns.to_parquet(local_usd_returns_path, index=False)
    fx_quality.to_csv(fx_quality_path, index=False)
    policy.to_csv(policy_path, index=False)

    currency_coverage = _safe_min(currency_reference, "security_coverage")
    fx_coverage = _safe_min(fx_quality, "calendar_coverage")
    usd_return_coverage = _usd_return_coverage(local_usd_returns)
    status = (
        "pass_currency_fx_returns_partial_tca"
        if currency_coverage >= 0.999 and fx_coverage >= 0.90 and usd_return_coverage >= 0.90
        else "incomplete"
    )
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": status,
        "currency_coverage": float(currency_coverage),
        "fx_calendar_coverage": float(fx_coverage),
        "usd_return_coverage": float(usd_return_coverage),
        "currencies": currencies,
        "fx_tickers": fx_tickers,
        "start_date": start_date,
        "end_date": end_date,
        "tca_calendar_status": "policy_stub_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "currency_reference": str(currency_reference_path.resolve()),
            "fx_rates": str(fx_rates_path.resolve()),
            "local_usd_returns": str(local_usd_returns_path.resolve()),
            "fx_quality": str(fx_quality_path.resolve()),
            "country_risk_tca_policy": str(policy_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return FXCountryRiskArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        currency_reference_path=currency_reference_path,
        fx_rates_path=fx_rates_path,
        local_usd_returns_path=local_usd_returns_path,
        fx_quality_path=fx_quality_path,
        policy_path=policy_path,
        currency_reference=currency_reference,
        fx_rates=fx_rates,
        local_usd_returns=local_usd_returns,
        fx_quality=fx_quality,
        policy=policy,
    )


def build_currency_reference(daily_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, group in daily_features.groupby("country", dropna=False):
        country_text = str(country)
        ref = ASEAN_U2_CURRENCY_REFERENCE.get(country_text, {})
        security_count = int(group["ticker"].nunique()) if "ticker" in group.columns else int(len(group))
        rows.append(
            {
                "country": country_text,
                "currency": ref.get("currency", pd.NA),
                "source": ref.get("source", "missing_policy"),
                "exchange_pattern": ref.get("exchange_pattern", ""),
                "security_count": security_count,
                "security_coverage": 1.0 if ref.get("currency") else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def download_yfinance_fx_rates(tickers: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(list(tickers), start=start_date, end=_end_exclusive(end_date), interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=list(tickers))
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
        else:
            close = raw.xs(raw.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]}) if "Close" in raw else raw
    return close


def build_fx_quality(fx_rates: pd.DataFrame, fx_tickers: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    rows = []
    for ticker in fx_tickers:
        series = pd.to_numeric(fx_rates.get(ticker, pd.Series(dtype=float)), errors="coerce")
        valid = series.dropna()
        rows.append(
            {
                "fx_ticker": ticker,
                "rows": int(valid.shape[0]),
                "first_date": valid.index.min().date().isoformat() if not valid.empty else "",
                "last_date": valid.index.max().date().isoformat() if not valid.empty else "",
                "calendar_days": int(len(dates)),
                "calendar_coverage": float(valid.shape[0] / max(1, len(dates))),
                "non_positive_rows": int(series.le(0).fillna(False).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_local_usd_return_paths(
    daily_features: pd.DataFrame,
    currency_reference: pd.DataFrame,
    fx_rates: pd.DataFrame,
    stale_limit_days: int = 5,
) -> pd.DataFrame:
    if daily_features.empty:
        return pd.DataFrame()
    data = daily_features.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    country_currency = dict(zip(currency_reference["country"], currency_reference["currency"], strict=False))
    data["currency"] = data["country"].map(country_currency)
    data["fx_ticker"] = data["currency"].map(FX_TICKERS)
    rates = fx_rates.sort_index().ffill(limit=stale_limit_days)
    parts = []
    for fx_ticker, group in data.groupby("fx_ticker", dropna=False):
        block = group.copy()
        if pd.isna(fx_ticker) or fx_ticker not in rates.columns:
            block["fx_local_per_usd"] = pd.NA
        else:
            fx = rates[fx_ticker].reindex(pd.DatetimeIndex(block["date"])).ffill(limit=stale_limit_days)
            block["fx_local_per_usd"] = fx.to_numpy()
        parts.append(block)
    out = pd.concat(parts, ignore_index=True, sort=False) if parts else data
    out["close_local"] = pd.to_numeric(out["close"], errors="coerce")
    out["close_usd"] = out["close_local"] / pd.to_numeric(out["fx_local_per_usd"], errors="coerce")
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["simple_return_local"] = out.groupby("ticker")["close_local"].pct_change(fill_method=None)
    out["simple_return_usd"] = out.groupby("ticker")["close_usd"].pct_change(fill_method=None)
    keep = [
        "date",
        "ticker",
        "country",
        "exchange",
        "currency",
        "fx_ticker",
        "fx_local_per_usd",
        "close_local",
        "close_usd",
        "simple_return_local",
        "simple_return_usd",
    ]
    return out[[col for col in keep if col in out.columns]]


def build_country_policy_table(currency_reference: pd.DataFrame) -> pd.DataFrame:
    policy = currency_reference[["country", "currency"]].copy()
    policy["withholding_tax_policy"] = "not_modelled_yet"
    policy["stamp_duty_policy"] = "not_modelled_yet"
    policy["exchange_fee_policy"] = "not_modelled_yet"
    policy["holiday_calendar_policy"] = "not_modelled_yet"
    policy["tca_readiness"] = "partial_policy_stub"
    return policy


def _normalize_fx_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out.index = out.index.normalize()
    return out


def _safe_min(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else 0.0


def _usd_return_coverage(local_usd_returns: pd.DataFrame) -> float:
    if local_usd_returns.empty or "close_local" not in local_usd_returns or "close_usd" not in local_usd_returns:
        return 0.0
    valid_local = pd.to_numeric(local_usd_returns["close_local"], errors="coerce").notna()
    if not valid_local.any():
        return 0.0
    valid_usd = pd.to_numeric(local_usd_returns["close_usd"], errors="coerce").notna()
    return float((valid_local & valid_usd).sum() / valid_local.sum())


def _end_exclusive(end_date: str) -> str:
    return str((pd.Timestamp(end_date) + pd.Timedelta(days=1)).date())
