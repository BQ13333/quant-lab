from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.data_lake import backfill_prices_in_batches, default_data_lake_root, write_universe_partitions
from quant_lab.data.refinitiv_symbols import add_refinitiv_ric
from quant_lab.data.universe_mining import (
    MinedUniverseResult,
    save_mined_asean_universe,
    save_mined_asia_universe,
    save_mined_us_universe,
)
from quant_lab.data.vendor_adapters import REFINITIV_OHLCV_FIELD_MAP, RefinitivAdapter


VALIDATED_REFINITIV_RIC_STATUSES = {"validated_price_history"}


def _now_run_id(prefix: str) -> str:
    return datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%d-%H%M%S")


def _read_private_config(path: str | None) -> str | None:
    if not path:
        return None
    resolved = Path(path).expanduser()
    return str(resolved) if resolved.exists() else None


def _parse_csv_arg(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().casefold() for item in value.split(",") if item.strip()}


def _filter_refinitiv_countries(
    universe: pd.DataFrame,
    include_countries: set[str] | None = None,
    exclude_countries: set[str] | None = None,
) -> pd.DataFrame:
    include_countries = include_countries or set()
    exclude_countries = exclude_countries or set()
    if not include_countries and not exclude_countries:
        return universe
    if "country" not in universe.columns:
        raise SystemExit("Refinitiv country filtering requires a country column in the symbol map.")

    country = universe["country"].astype("string").str.strip().str.casefold()
    filtered = universe.copy()
    if include_countries:
        filtered = filtered.loc[country.isin(include_countries)].copy()
        country = filtered["country"].astype("string").str.strip().str.casefold()
    if exclude_countries:
        filtered = filtered.loc[~country.isin(exclude_countries)].copy()
    if filtered.empty:
        raise SystemExit("Refinitiv country filters removed every validated symbol.")
    return filtered


def _result_record(name: str, result: MinedUniverseResult) -> dict[str, object]:
    return {
        "name": name,
        "run_id": result.run_id,
        "artifact_dir": str(result.artifact_dir),
        "listings_path": str(result.artifact_dir / "listings.csv"),
        "manifest_path": str(result.artifact_dir / "manifest.csv"),
        "rows": int(len(result.listings)),
        "countries": sorted(result.listings["country"].dropna().unique().tolist()) if "country" in result.listings else [],
        "exchanges": sorted(result.listings["exchange"].dropna().unique().tolist()) if "exchange" in result.listings else [],
    }


def _combine_listings(results: list[MinedUniverseResult]) -> pd.DataFrame:
    frames = []
    for result in results:
        if result.listings.empty:
            continue
        frame = result.listings.copy()
        frame["universe_run_id"] = result.run_id
        frame["universe_artifact_dir"] = str(result.artifact_dir)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = add_refinitiv_ric(combined)
    dedupe_cols = [col for col in ("country", "exchange", "ticker", "yahoo_ticker", "refinitiv_ric") if col in combined.columns]
    return combined.drop_duplicates(dedupe_cols, keep="first").reset_index(drop=True)


def _source_sample(listings: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or listings.empty:
        return listings.head(0).copy()
    if len(listings) <= max_rows:
        return listings.copy()
    group_cols = [col for col in ("country", "exchange") if col in listings.columns]
    if not group_cols:
        return listings.head(max_rows).copy()
    per_group = max(1, max_rows // max(1, listings.groupby(group_cols, dropna=False).ngroups))
    sample = (
        listings.sort_values(group_cols + ["ticker"])
        .groupby(group_cols, dropna=False, group_keys=False)
        .head(per_group)
        .head(max_rows)
        .copy()
    )
    return sample


def _refinitiv_universe_from_validation(
    combined: pd.DataFrame,
    validation_csv: str | None,
    allow_inferred_rics: bool,
    max_rows: int,
) -> pd.DataFrame:
    if combined.empty or "refinitiv_ric" not in combined.columns:
        return combined.head(0).copy()

    if validation_csv:
        validation = pd.read_csv(validation_csv, low_memory=False)
        required_cols = {"refinitiv_ric", "status"}
        missing = required_cols - set(validation.columns)
        if missing:
            raise SystemExit(f"Refinitiv validation CSV is missing columns: {sorted(missing)}")
        validated = validation[
            validation["status"].isin(VALIDATED_REFINITIV_RIC_STATUSES) & validation["refinitiv_ric"].notna()
        ].copy()
        if validated.empty:
            raise SystemExit("Refinitiv validation CSV has no validated price-history RICs.")
        validated["refinitiv_ric"] = validated["refinitiv_ric"].astype(str)
        valid_rics = set(validated["refinitiv_ric"])
        universe = combined[combined["refinitiv_ric"].astype(str).isin(valid_rics)].copy()
        if universe.empty:
            raise SystemExit("No combined-listing RICs matched the supplied Refinitiv validation CSV.")
        validation_cols = ["refinitiv_ric", "status", "successful_fields", "max_coverage", "last_available"]
        validation_cols = [col for col in validation_cols if col in validated.columns]
        validation_join = validated[validation_cols].drop_duplicates("refinitiv_ric").rename(
            columns={"status": "refinitiv_validation_status"}
        )
        universe = universe.merge(
            validation_join,
            on="refinitiv_ric",
            how="left",
        )
        return _source_sample(universe, max_rows)

    if allow_inferred_rics:
        universe = combined[combined["refinitiv_ric"].notna()].copy()
        if "refinitiv_ric_status" in universe.columns:
            universe["refinitiv_validation_status"] = universe["refinitiv_ric_status"]
        else:
            universe["refinitiv_validation_status"] = "inferred_not_validated"
        return _source_sample(universe, max_rows)

    raise SystemExit(
        "Refinitiv live extraction requires --refinitiv-validation-csv from a field probe. "
        "Use --allow-inferred-refinitiv-rics only for a small smoke test."
    )


def _refinitiv_universe_from_symbol_map(
    symbol_map_csv: str,
    max_rows: int,
    include_countries: set[str] | None = None,
    exclude_countries: set[str] | None = None,
) -> pd.DataFrame:
    symbol_map = pd.read_csv(symbol_map_csv, low_memory=False)
    required_cols = {"refinitiv_ric", "refinitiv_symbol_status"}
    missing = required_cols - set(symbol_map.columns)
    if missing:
        raise SystemExit(f"Refinitiv symbol-map CSV is missing columns: {sorted(missing)}")
    universe = symbol_map[
        symbol_map["refinitiv_symbol_status"].eq("validated_price_history") & symbol_map["refinitiv_ric"].notna()
    ].copy()
    if universe.empty:
        raise SystemExit("Refinitiv symbol-map CSV has no validated symbols.")
    universe = _filter_refinitiv_countries(
        universe,
        include_countries=include_countries,
        exclude_countries=exclude_countries,
    )
    if "yahoo_ticker" not in universe.columns:
        universe["yahoo_ticker"] = universe["refinitiv_ric"]
    else:
        universe["source_yahoo_ticker"] = universe["yahoo_ticker"]
        universe["yahoo_ticker"] = universe["refinitiv_ric"]
    universe["refinitiv_validation_status"] = universe["refinitiv_symbol_status"]
    universe["refinitiv_symbol_map_path"] = str(symbol_map_csv)
    return _source_sample(universe, max_rows)


def _load_refinitiv_field_catalog(field_catalog_csv: str | None, validation_csv: str | None) -> tuple[pd.DataFrame, str | None]:
    candidate = Path(field_catalog_csv).expanduser() if field_catalog_csv else None
    if candidate is None and validation_csv:
        sibling = Path(validation_csv).expanduser().with_name("field_catalog.csv")
        if sibling.exists():
            candidate = sibling
    if candidate is None:
        return pd.DataFrame(), None
    if not candidate.exists():
        raise SystemExit(f"Refinitiv field catalog not found: {candidate}")
    catalog = pd.read_csv(candidate, low_memory=False)
    return catalog, str(candidate)


def _market_field_map_from_catalog(field_catalog: pd.DataFrame, meta: dict[str, object]) -> dict[str, str] | None:
    if field_catalog.empty:
        return None
    required = {"country", "exchange", "field", "field_role"}
    if not required.issubset(field_catalog.columns):
        return None
    country = str(meta.get("country"))
    exchange = str(meta.get("exchange"))
    country_catalog = field_catalog[field_catalog["country"].astype(str).eq(country)].copy()
    market = country_catalog[country_catalog["exchange"].astype(str).eq(exchange)].copy()
    if market.empty and not country_catalog.empty:
        role_fields = country_catalog.groupby("field_role")["field"].nunique(dropna=True)
        if not role_fields.empty and role_fields.le(1).all():
            market = country_catalog.copy()
    if market.empty:
        return None
    field_map = dict(zip(market["field_role"].astype(str), market["field"].astype(str), strict=False))
    return field_map or None


def _refinitiv_download_kwargs_builder(field_catalog: pd.DataFrame):
    def build(meta: dict[str, object], batch: pd.DataFrame) -> dict[str, object]:
        field_map = _market_field_map_from_catalog(field_catalog, meta)
        if field_map is None and "validated_close_field" in batch.columns:
            close_fields = batch["validated_close_field"].dropna().astype(str).unique().tolist()
            if len(close_fields) == 1:
                field_map = {**REFINITIV_OHLCV_FIELD_MAP, "Close": close_fields[0]}
        return {"field_map": field_map} if field_map else {}

    return build


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled all-source universe and price extraction.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/data_extraction")
    parser.add_argument("--universe-root", default="artifacts/universes")
    parser.add_argument("--lake-root", default=None)
    parser.add_argument("--regions", default="asia,asean,us", help="Comma-separated: asia,asean,us.")
    parser.add_argument("--asia-sources", default="twse,tpex,krx,hkex,jpx,sse,szse")
    parser.add_argument("--asean-sources", default="sgx")
    parser.add_argument("--write-universe-lake", action="store_true")
    parser.add_argument("--download-yahoo", action="store_true")
    parser.add_argument("--download-refinitiv", action="store_true")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-yahoo-tickers", type=int, default=250)
    parser.add_argument("--max-refinitiv-tickers", type=int, default=250)
    parser.add_argument("--refinitiv-session", default="desktop.workspace")
    parser.add_argument("--refinitiv-config", default=None)
    parser.add_argument("--refinitiv-validation-csv", default=None)
    parser.add_argument("--refinitiv-field-catalog-csv", default=None)
    parser.add_argument("--refinitiv-symbol-map-csv", default=None)
    parser.add_argument("--refinitiv-include-countries", default=None, help="Comma-separated country filter for symbol-map downloads.")
    parser.add_argument("--refinitiv-exclude-countries", default=None, help="Comma-separated country exclusion for symbol-map downloads.")
    parser.add_argument("--allow-inferred-refinitiv-rics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or _now_run_id("ALL-SOURCES")
    artifact_dir = Path(args.artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    lake_root = Path(args.lake_root) if args.lake_root else default_data_lake_root(Path.cwd())
    regions = {region.strip().lower() for region in args.regions.split(",") if region.strip()}
    refinitiv_include_countries = _parse_csv_arg(args.refinitiv_include_countries)
    refinitiv_exclude_countries = _parse_csv_arg(args.refinitiv_exclude_countries)

    results: list[MinedUniverseResult] = []
    records: list[dict[str, object]] = []
    if "asia" in regions:
        result = save_mined_asia_universe(
            artifact_root=args.universe_root,
            sources=tuple(source.strip() for source in args.asia_sources.split(",") if source.strip()),
        )
        results.append(result)
        records.append(_result_record("asia", result))
    if "asean" in regions:
        result = save_mined_asean_universe(
            artifact_root=args.universe_root,
            sources=tuple(source.strip() for source in args.asean_sources.split(",") if source.strip()),
        )
        results.append(result)
        records.append(_result_record("asean", result))
    if "us" in regions:
        result = save_mined_us_universe(artifact_root=args.universe_root)
        results.append(result)
        records.append(_result_record("us", result))

    combined = _combine_listings(results)
    combined_path = artifact_dir / "combined_listings.csv"
    combined.to_csv(combined_path, index=False)

    price_runs: list[dict[str, object]] = []
    if args.write_universe_lake and not combined.empty:
        partition_path = write_universe_partitions(
            combined,
            root=lake_root,
            universe_id=run_id,
            overwrite=True,
        )
        price_runs.append({"type": "universe_partitions", "path": str(partition_path), "rows": int(len(combined))})

    if args.download_yahoo and not args.dry_run:
        yahoo_universe = _source_sample(combined[combined["yahoo_ticker"].notna()].copy(), args.max_yahoo_tickers)
        result = backfill_prices_in_batches(
            yahoo_universe,
            root=lake_root,
            run_id=f"PRICE-BACKFILL-YAHOO-{run_id}",
            period=args.period,
            interval=args.interval,
            batch_size=args.batch_size,
            provider="yahoo",
            adjustment="auto_adjusted",
            data_tier="bronze",
            write_long_ohlcv=True,
        )
        price_runs.append(
            {
                "type": "yahoo_prices",
                "run_id": result.run_id,
                "root": str(result.root),
                "manifest_path": str(result.manifest_path),
                "long_ohlcv_root": str(result.long_ohlcv_root),
                "requested_tickers": int(result.manifest["requested_tickers"].sum()) if not result.manifest.empty else 0,
                "price_columns": int(result.manifest["price_columns"].sum()) if not result.manifest.empty else 0,
                "failures": int(result.manifest["failures"].sum()) if not result.manifest.empty else 0,
            }
        )

    refinitiv_config = _read_private_config(args.refinitiv_config)
    refinitiv_ready = bool(refinitiv_config or args.refinitiv_session)
    resolved_refinitiv_field_catalog_path: str | None = None
    if args.download_refinitiv and not args.dry_run:
        if not refinitiv_ready:
            raise SystemExit("Refinitiv requested but no session/config was supplied.")
        if args.refinitiv_symbol_map_csv:
            ric_universe = _refinitiv_universe_from_symbol_map(
                symbol_map_csv=args.refinitiv_symbol_map_csv,
                max_rows=args.max_refinitiv_tickers,
                include_countries=refinitiv_include_countries,
                exclude_countries=refinitiv_exclude_countries,
            )
        else:
            ric_universe = _refinitiv_universe_from_validation(
                combined=combined,
                validation_csv=args.refinitiv_validation_csv,
                allow_inferred_rics=args.allow_inferred_refinitiv_rics,
                max_rows=args.max_refinitiv_tickers,
            )
            ric_universe = ric_universe.assign(yahoo_ticker=ric_universe["refinitiv_ric"])
        adapter = RefinitivAdapter(
            session_name=args.refinitiv_session,
            config_name=refinitiv_config,
        )
        field_catalog, field_catalog_path = _load_refinitiv_field_catalog(
            field_catalog_csv=args.refinitiv_field_catalog_csv,
            validation_csv=args.refinitiv_validation_csv,
        )
        resolved_refinitiv_field_catalog_path = field_catalog_path
        result = backfill_prices_in_batches(
            ric_universe,
            root=lake_root,
            run_id=f"PRICE-BACKFILL-REFINITIV-{run_id}",
            period=args.period,
            interval=args.interval,
            batch_size=args.batch_size,
            provider="refinitiv",
            adjustment="vendor_default",
            data_tier="bronze",
            write_long_ohlcv=True,
            downloader=adapter.download_price_history,
            download_kwargs_by_batch=_refinitiv_download_kwargs_builder(field_catalog),
        )
        price_runs.append(
            {
                "type": "refinitiv_prices",
                "run_id": result.run_id,
                "root": str(result.root),
                "manifest_path": str(result.manifest_path),
                "long_ohlcv_root": str(result.long_ohlcv_root),
                "requested_tickers": int(result.manifest["requested_tickers"].sum()) if not result.manifest.empty else 0,
                "price_columns": int(result.manifest["price_columns"].sum()) if not result.manifest.empty else 0,
                "failures": int(result.manifest["failures"].sum()) if not result.manifest.empty else 0,
                "field_catalog_path": field_catalog_path,
            }
        )

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "lake_root": str(lake_root),
        "regions": sorted(regions),
        "combined_listings_path": str(combined_path),
        "combined_rows": int(len(combined)),
        "with_yahoo_ticker": int(combined["yahoo_ticker"].notna().sum()) if "yahoo_ticker" in combined else 0,
        "with_proposed_refinitiv_ric": int(combined["refinitiv_ric"].notna().sum())
        if "refinitiv_ric" in combined
        else 0,
        "refinitiv_ric_status_counts": combined["refinitiv_ric_status"].value_counts(dropna=False).to_dict()
        if "refinitiv_ric_status" in combined
        else {},
        "universe_runs": records,
        "price_runs": price_runs,
        "refinitiv": {
            "package_installed": True,
            "session": args.refinitiv_session,
            "config_supplied_and_exists": bool(refinitiv_config),
            "live_extraction_requested": bool(args.download_refinitiv),
            "validation_csv": args.refinitiv_validation_csv,
            "field_catalog_csv": args.refinitiv_field_catalog_csv,
            "symbol_map_csv": args.refinitiv_symbol_map_csv,
            "include_countries": sorted(refinitiv_include_countries),
            "exclude_countries": sorted(refinitiv_exclude_countries),
            "resolved_field_catalog_csv": resolved_refinitiv_field_catalog_path,
            "inferred_rics_allowed": bool(args.allow_inferred_refinitiv_rics),
            "credential_policy": "Credentials must remain outside this repo and outside chat.",
        },
        "caveat": "Public listings are current snapshots, not point-in-time security masters. Refinitiv RICs are proposed unless supplied by an audited symbol map or validated by a live probe.",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = artifact_dir / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Combined listings: {len(combined)}")
    if not combined.empty and {"country", "exchange"}.issubset(combined.columns):
        print(combined.groupby(["country", "exchange"], dropna=False).size().sort_values(ascending=False).head(25).to_string())
    print(f"Manifest: {manifest_path}")
    for price_run in price_runs:
        print(json.dumps(price_run, indent=2))


if __name__ == "__main__":
    main()
