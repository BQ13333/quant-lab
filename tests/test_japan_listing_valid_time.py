from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_listing_valid_time import (
    ACTIVE_MAP_FILE,
    INPUT_CHECKSUMS_FILE,
    NORMALIZED_FILE,
    REFERENCE_MAP_FILE,
    STATUS_FILE,
    build_japan_listing_valid_time_tables,
    write_japan_listing_valid_time_artifacts,
)


NOW = datetime(2026, 7, 12, 8, 0, tzinfo=timezone.utc)
OBSERVED = "2026-07-11T16:00:00+00:00"


def _map_row(
    ric: str,
    stable_id: str,
    *,
    active: bool,
    first_trade_date: str | None,
    retire_date: str | None,
    known_at: str | None = None,
) -> dict[str, object]:
    return {
        "ric": ric,
        "stable_instrument_id": stable_id,
        "stable_identifier_type": "isin",
        "isin": stable_id.split(":", 1)[1],
        "exchange_code": "TYO",
        "is_direct_jpx_quote": True,
        "has_direct_jpx_listing_evidence": True,
        "is_active": active,
        "local_code": ric.removesuffix(".T"),
        "first_trade_date": first_trade_date,
        "retire_date": retire_date,
        "jpx_first_trade_date": first_trade_date,
        "jpx_known_at_utc": known_at,
        "security_master_known_at_utc": None,
    }


def _normalized_row(
    ric: str,
    isin: str,
    *,
    first_trade_date: str | None,
    retire_date: str | None,
) -> dict[str, object]:
    return {
        "ric": ric,
        "first_trade_date": first_trade_date,
        "retire_date": retire_date,
        "isin": isin,
        "sedol": f"S-{ric}",
        "price_currency": "JPY",
        "observed_at_utc": OBSERVED,
        "evidence_scope": "refinitiv_current_observation_only",
    }


def _status(rics: list[str], status: str = "success") -> pd.DataFrame:
    return pd.DataFrame(
        {"ric": rics, "status": status, "observed_at_utc": OBSERVED}
    )


def test_builds_stable_and_ric_intervals_without_inventing_pit_history() -> None:
    stable_id = "isin:JP0000000001"
    active = pd.DataFrame(
        [
            _map_row(
                "NEW.T",
                stable_id,
                active=True,
                first_trade_date="2020-01-01",
                retire_date="2026-08-01",
            )
        ]
    )
    reference = pd.DataFrame(
        [
            _map_row(
                "OLD.T",
                stable_id,
                active=False,
                first_trade_date="2000-01-01",
                retire_date="2019-12-31",
            ),
            active.iloc[0].to_dict(),
        ]
    )
    normalized = pd.DataFrame(
        [
            _normalized_row(
                "OLD.T",
                "JP0000000001",
                first_trade_date="2000-01-01",
                retire_date="2019-12-31",
            ),
            _normalized_row(
                "NEW.T",
                "JP0000000001",
                first_trade_date="2020-01-01",
                retire_date="2026-08-01",
            ),
        ]
    )

    tables = build_japan_listing_valid_time_tables(
        active,
        reference,
        normalized,
        _status(["OLD.T", "NEW.T"]),
    )

    listing = tables.listing_intervals.iloc[0]
    assert listing["listing_start"] == "2000-01-01"
    assert listing["retire_date"] == "2026-08-01"
    assert bool(listing["scheduled_retirement"]) is True
    assert bool(listing["retired_as_of"]) is False
    assert listing["boundary_state"] == "scheduled_retirement"
    assert listing["ric_count"] == 2

    quotes = tables.ric_quote_intervals.set_index("ric")
    assert quotes.loc["OLD.T", "boundary_state"] == "retired"
    assert quotes.loc["NEW.T", "boundary_state"] == "scheduled_retirement"
    assert bool(quotes.loc["NEW.T", "retired_as_of"]) is False

    observations = tables.current_observations
    assert set(observations["attribute_name"]) == {"isin", "sedol", "currency"}
    assert observations[["valid_from", "valid_to"]].isna().all().all()
    assert observations["undated_current_observation"].astype(bool).all()
    assert not observations["pit_usable"].astype(bool).any()
    assert observations["known_at_utc"].notna().all()

    assert tables.valid_time_membership.iloc[0]["membership_end"] == "2026-08-01"
    pit = tables.pit_membership.iloc[0]
    assert pd.isna(pit["known_at_utc"])
    assert bool(pit["pit_usable"]) is False
    assert pit["pit_block_reason"] == "missing_source_backed_known_at_utc"


def test_explicit_source_known_at_is_preserved_and_null_never_pit_usable() -> None:
    known = "2026-07-10T00:00:00+00:00"
    active = pd.DataFrame(
        [
            _map_row(
                "KNOWN.T",
                "isin:JP0000000002",
                active=True,
                first_trade_date="2022-01-01",
                retire_date=None,
                known_at=known,
            ),
            _map_row(
                "NULL.T",
                "isin:JP0000000003",
                active=True,
                first_trade_date=None,
                retire_date=None,
            ),
        ]
    )
    normalized = pd.DataFrame(
        [
            _normalized_row(
                "KNOWN.T", "JP0000000002", first_trade_date="2022-01-01", retire_date=None
            ),
            _normalized_row(
                "NULL.T", "JP0000000003", first_trade_date=None, retire_date=None
            ),
        ]
    )
    tables = build_japan_listing_valid_time_tables(
        active, active.copy(), normalized, _status(["KNOWN.T", "NULL.T"])
    )
    pit = tables.pit_membership.set_index("stable_instrument_id")
    assert pit.loc["isin:JP0000000002", "known_at_utc"] == known
    assert bool(pit.loc["isin:JP0000000002", "pit_usable"]) is True
    assert pd.isna(pit.loc["isin:JP0000000003", "known_at_utc"])
    assert bool(pit.loc["isin:JP0000000003", "pit_usable"]) is False
    listing = tables.listing_intervals.set_index("stable_instrument_id")
    assert bool(listing.loc["isin:JP0000000003", "unknown_start"]) is True
    assert bool(listing.loc["isin:JP0000000003", "unknown_end"]) is True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _writer_inputs(root: Path) -> tuple[Path, Path]:
    reconciliation = root / "reconciliation"
    enrichment = root / "enrichment"
    reconciliation.mkdir(parents=True)
    enrichment.mkdir(parents=True)

    active = pd.DataFrame(
        [
            _map_row(
                "7203.T",
                "isin:JP3633400001",
                active=True,
                first_trade_date="1949-05-16",
                retire_date="2026-09-01",
            )
        ]
    )
    active.to_csv(reconciliation / ACTIVE_MAP_FILE, index=False)
    active.to_csv(reconciliation / REFERENCE_MAP_FILE, index=False)
    (reconciliation / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_sha256": {
                    "active_operating_core": _sha256(reconciliation / ACTIVE_MAP_FILE),
                    "historical_listed_reference": _sha256(
                        reconciliation / REFERENCE_MAP_FILE
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    normalized = pd.DataFrame(
        [
            _normalized_row(
                "7203.T",
                "JP3633400001",
                first_trade_date="1949-05-16",
                retire_date="2026-09-01",
            )
        ]
    )
    normalized.to_csv(enrichment / NORMALIZED_FILE, index=False)
    _status(["7203.T"]).to_csv(enrichment / STATUS_FILE, index=False)
    checksums = enrichment / INPUT_CHECKSUMS_FILE
    checksums.write_text(
        f"{_sha256(enrichment / NORMALIZED_FILE)}  {NORMALIZED_FILE}\n"
        f"{_sha256(enrichment / STATUS_FILE)}  {STATUS_FILE}\n",
        encoding="utf-8",
    )
    (enrichment / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete_current_observation",
                "current_observation_only": True,
                "historical_pit_ready": False,
                "artifact_sha256": {
                    "normalized_responses": _sha256(enrichment / NORMALIZED_FILE),
                    "ric_status": _sha256(enrichment / STATUS_FILE),
                    "checksums": _sha256(checksums),
                },
            }
        ),
        encoding="utf-8",
    )
    return reconciliation, enrichment


def test_writer_is_atomic_immutable_hashed_and_never_promotes(tmp_path: Path) -> None:
    reconciliation, enrichment = _writer_inputs(tmp_path)
    result = write_japan_listing_valid_time_artifacts(
        reconciliation_dir=reconciliation,
        enrichment_dir=enrichment,
        artifact_root=tmp_path / "artifacts",
        run_id="JP-VALID-TIME",
        clock=lambda: NOW,
    )
    assert result.manifest["immutable"] is True
    assert result.manifest["atomic_publish"] is True
    assert result.manifest["historical_backtest_eligible"] is False
    assert result.manifest["alpha_promotion_allowed"] is False
    assert result.manifest["counts"]["scheduled_retirement_rows"] == 1
    assert not any(result.tables.pit_membership["pit_usable"].astype(bool))
    assert not list((tmp_path / "artifacts").glob(".*.tmp-*"))

    for line in result.checksums_path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(result.artifact_dir / name) == expected
    assert result.manifest["artifact_sha256"]["checksums"] == _sha256(
        result.checksums_path
    )
    assert set(result.manifest["input_sha256"]) == {
        "active_map",
        "reference_map",
        "normalized",
        "status",
        "enrichment_checksums",
        "enrichment_manifest",
        "reconciliation_manifest",
    }

    with pytest.raises(FileExistsError, match="immutable artifact directory"):
        write_japan_listing_valid_time_artifacts(
            reconciliation_dir=reconciliation,
            enrichment_dir=enrichment,
            artifact_root=tmp_path / "artifacts",
            run_id="JP-VALID-TIME",
        )


def test_writer_rejects_mutated_or_incomplete_enrichment(tmp_path: Path) -> None:
    reconciliation, enrichment = _writer_inputs(tmp_path)
    with (enrichment / NORMALIZED_FILE).open("a", encoding="utf-8") as handle:
        handle.write("MUTATED.T,,,,,,,\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        write_japan_listing_valid_time_artifacts(
            reconciliation_dir=reconciliation,
            enrichment_dir=enrichment,
            artifact_root=tmp_path / "out-a",
            run_id="MUTATED",
        )

    reconciliation, enrichment = _writer_inputs(tmp_path / "second")
    manifest_path = enrichment / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "bounded_in_progress"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not complete_current_observation"):
        write_japan_listing_valid_time_artifacts(
            reconciliation_dir=reconciliation,
            enrichment_dir=enrichment,
            artifact_root=tmp_path / "out-b",
            run_id="INCOMPLETE",
        )


def test_runner_exposes_owned_cli() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "build_japan_listing_valid_time.py"
    spec = importlib.util.spec_from_file_location("build_japan_listing_valid_time", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    args = parser.parse_args(["--run-id", "TEST-RUN", "--as-of-date", "2026-07-11"])
    assert args.run_id == "TEST-RUN"
    assert args.as_of_date == "2026-07-11"
