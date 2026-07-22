from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _coalesce(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=data.index, dtype="object")
    for column in columns:
        if column in data.columns:
            result = result.where(result.notna(), data[column])
    return result


def _load_security_master(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _active_filter(data: pd.DataFrame, include_active: bool, include_inactive: bool) -> pd.DataFrame:
    if include_active and include_inactive:
        return data
    bucket = data.get("active_state_bucket", pd.Series(pd.NA, index=data.index)).astype("string").str.casefold()
    active = bucket.eq("active")
    if include_active:
        return data.loc[active].copy()
    if include_inactive:
        return data.loc[~active].copy()
    return data.head(0).copy()


def build_symbol_map_from_security_master(
    security_master: pd.DataFrame,
    source_path: Path,
    include_active: bool = True,
    include_inactive: bool = True,
    max_symbols: int | None = None,
) -> pd.DataFrame:
    if security_master.empty:
        return pd.DataFrame()
    data = security_master.copy()
    if "ric" not in data.columns:
        raise ValueError("Security master must include a ric column")
    data = data.loc[data["ric"].notna()].copy()
    data = _active_filter(data, include_active=include_active, include_inactive=include_inactive)
    if data.empty:
        return pd.DataFrame()

    data["refinitiv_ric"] = data["ric"].astype("string").str.strip()
    data = data.loc[data["refinitiv_ric"].notna() & data["refinitiv_ric"].ne("")]
    data = data.drop_duplicates("refinitiv_ric", keep="first").copy()
    data["ticker"] = _coalesce(data, ["ticker_symbol", "refinitiv_ric"])
    data["exchange"] = _coalesce(data, ["exchange_code", "exchange_name"])
    data["name"] = _coalesce(data, ["common_name", "document_title"])
    data["currency"] = _coalesce(data, ["price_currency"])

    symbol_map = pd.DataFrame(
        {
            "country": _coalesce(data, ["country"]),
            "country_code": _coalesce(data, ["country_code", "exchange_country"]),
            "exchange": data["exchange"],
            "exchange_name": _coalesce(data, ["exchange_name"]),
            "exchange_code": _coalesce(data, ["exchange_code"]),
            "ticker": data["ticker"],
            "name": data["name"],
            "currency": data["currency"],
            "refinitiv_ric": data["refinitiv_ric"],
            "refinitiv_symbol_status": "security_master_discovery_candidate",
            "validated_close_field": pd.NA,
            "candidate_method": "refinitiv_discovery_security_master",
            "symbol_map_source": str(source_path.resolve()),
            "security_master_active_bucket": _coalesce(data, ["active_state_bucket"]),
            "security_master_active_flag": _coalesce(data, ["active_flag"]),
            "first_trade_date": _coalesce(data, ["first_trade_date"]),
            "retire_date": _coalesce(data, ["retire_date"]),
            "instrument_permid": _coalesce(data, ["instrument_permid"]),
            "perm_id": _coalesce(data, ["perm_id"]),
            "organization_id": _coalesce(data, ["organization_id"]),
            "isin": _coalesce(data, ["isin"]),
            "sedol": _coalesce(data, ["sedol"]),
        }
    )
    sort_cols = [col for col in ("country", "exchange", "security_master_active_bucket", "ticker", "refinitiv_ric") if col in symbol_map.columns]
    symbol_map = symbol_map.sort_values(sort_cols).reset_index(drop=True)
    if max_symbols is not None:
        symbol_map = symbol_map.head(max_symbols).copy()
    return symbol_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Refinitiv Discovery security master into a historical symbol map.")
    parser.add_argument("--security-master", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_symbol_map")
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--inactive-only", action="store_true")
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args()

    if args.active_only and args.inactive_only:
        raise SystemExit("--active-only and --inactive-only cannot both be set")

    source_path = Path(args.security_master)
    security_master = _load_security_master(source_path)
    symbol_map = build_symbol_map_from_security_master(
        security_master,
        source_path=source_path,
        include_active=not args.inactive_only,
        include_inactive=not args.active_only,
        max_symbols=args.max_symbols,
    )
    artifact_dir = Path(args.artifact_root) / args.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    symbol_map_path = artifact_dir / "symbol_map.csv"
    manifest_path = artifact_dir / "manifest.json"
    symbol_map.to_csv(symbol_map_path, index=False)

    manifest = {
        "run_id": args.run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "symbol_map_path": str(symbol_map_path.resolve()),
        "security_master_source": str(source_path.resolve()),
        "rows": int(len(symbol_map)),
        "active_rows": int(symbol_map["security_master_active_bucket"].astype("string").str.casefold().eq("active").sum())
        if not symbol_map.empty
        else 0,
        "inactive_rows": int(symbol_map["security_master_active_bucket"].astype("string").str.casefold().ne("active").sum())
        if not symbol_map.empty
        else 0,
        "status_counts": symbol_map["refinitiv_symbol_status"].value_counts(dropna=False).to_dict()
        if not symbol_map.empty
        else {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Run ID: {args.run_id}")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Symbol map: {symbol_map_path}")
    print(f"Rows: {manifest['rows']}")
    print(f"Active rows: {manifest['active_rows']}")
    print(f"Inactive rows: {manifest['inactive_rows']}")
    if not symbol_map.empty:
        print(symbol_map["security_master_active_bucket"].value_counts(dropna=False).to_string())
        print(symbol_map[["country", "exchange", "ticker", "refinitiv_ric", "security_master_active_bucket"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
