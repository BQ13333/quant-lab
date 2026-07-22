from __future__ import annotations

import argparse
import warnings

from quant_lab.data.security_master_discovery import (
    DEFAULT_DISCOVERY_COUNTRY_CODES,
    SUPPORTED_DISCOVERY_COUNTRY_CODES,
    write_refinitiv_security_master_discovery_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def _parse_country_codes(value: str) -> tuple[str, ...]:
    if not value:
        return DEFAULT_DISCOVERY_COUNTRY_CODES
    allowed = {code.casefold(): code for code in SUPPORTED_DISCOVERY_COUNTRY_CODES}
    aliases = {
        "my": "MYS",
        "malaysia": "MYS",
        "sg": "SGP",
        "singapore": "SGP",
        "id": "IDN",
        "indonesia": "IDN",
        "cn": "CHN",
        "china": "CHN",
        "hk": "HKG",
        "hong kong": "HKG",
        "tw": "TWN",
        "taiwan": "TWN",
        "kr": "KOR",
        "korea": "KOR",
        "south korea": "KOR",
        "jp": "JPN",
        "japan": "JPN",
        "us": "USA",
        "usa": "USA",
        "united states": "USA",
        "th": "THA",
        "thailand": "THA",
    }
    codes: list[str] = []
    for item in value.split(","):
        key = item.strip().casefold()
        if not key:
            continue
        code = aliases.get(key, allowed.get(key))
        if code is None:
            raise SystemExit(f"Unsupported Refinitiv Discovery country code: {item}")
        codes.append(code)
    return tuple(dict.fromkeys(codes))


def _parse_state_buckets(value: str) -> tuple[str, ...]:
    if not value:
        return ("active", "inactive")
    allowed = {"active", "inactive"}
    buckets: list[str] = []
    for item in value.split(","):
        bucket = item.strip().casefold()
        if not bucket:
            continue
        if bucket not in allowed:
            raise SystemExit(f"Unsupported state bucket: {item}")
        buckets.append(bucket)
    return tuple(dict.fromkeys(buckets))


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"lseg\.data\..*")
    parser = argparse.ArgumentParser(
        description="Extract active and inactive ordinary equities via LSEG Discovery Search."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/data_lake/pit/security_master_history")
    parser.add_argument("--countries", default=",".join(DEFAULT_DISCOVERY_COUNTRY_CODES))
    parser.add_argument("--state-buckets", default="active,inactive")
    parser.add_argument("--top", type=int, default=1000)
    parser.add_argument("--start-skip", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--enrichment-chunk-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--primary-only", action="store_true", help="Filter Discovery to primary RIC/primary issue rows.")
    parser.add_argument(
        "--extra-filter",
        default=None,
        help="Additional LSEG Discovery filter expression, wrapped in parentheses and appended with AND.",
    )
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    countries = _parse_country_codes(args.countries)
    state_buckets = _parse_state_buckets(args.state_buckets)
    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        artifact = write_refinitiv_security_master_discovery_artifacts(
            discovery_client=adapter.ld,
            get_data_client=adapter.ld,
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            country_codes=countries,
            state_buckets=state_buckets,
            top=args.top,
            start_skip=args.start_skip,
            max_pages=args.max_pages,
            max_rows=args.max_rows,
            enrichment_chunk_size=args.enrichment_chunk_size,
            sleep_seconds=args.sleep_seconds,
            primary_only=args.primary_only,
            extra_filter=args.extra_filter,
        )
    finally:
        adapter.close()

    print(f"Run ID: {artifact.run_id}")
    print(f"Artifact dir: {artifact.artifact_dir}")
    print(f"Status: {artifact.manifest['status']}")
    print(f"Rows: {artifact.manifest['rows']}")
    print(f"Dead/inactive rows: {artifact.manifest['dead_name_rows']}")
    print(f"Countries with rows: {artifact.manifest['countries']}")
    print(f"Manifest: {artifact.manifest_path}")
    print(artifact.quality.to_string(index=False))


if __name__ == "__main__":
    main()
