from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.parse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
OFFICIAL_PRODUCER_RUN_ID = (
    "JPN-G4-HALT-OFFICIAL-PUBLICATION-TIMESTAMPS-V8-20260712-V1"
)
OFFICIAL_VALIDATION_RUN_ID = (
    "JPN-G4-HALT-OFFICIAL-PUBLICATION-TIMESTAMPS-V8-"
    "INDEPENDENT-VALIDATION-20260712-V1"
)
PRODUCER_COMPONENT = "japan_g4_halt_v8_official_publication_timestamp_recovery"
VALIDATION_COMPONENT = (
    "japan_g4_halt_v8_official_publication_timestamp_recovery_validation"
)
PUBLICATION_PRECISION = "official_page_date_only"
CONSERVATIVE_DATE_ONLY_RULE = (
    "date_only_conservative_upper_bound_next_midnight_jst"
)

EXPECTED_EVENT_IDS = frozenset(
    {
        "jpx-report-d129a47fac42d1338b0cdf10",
        "jpx-report-e05f17d2e939486e13fde78d",
        "jpx-report-5211314f9df91d3614fbf85c",
        "jpx-report-c41cf5f4ef754f5adfede406",
        "jpx-report-bd31eeba13db47900ee0cf58",
        "jpx-report-40ff9f970778ba0e64673c62",
        "jpx-delist-f21453f0f4aff06cbceedf0d",
        "jpx-delist-8f88eba45e321e35ff26a1c7",
        "jpx-delist-3197ccfb203a833a61027944",
        "jpx-delist-f81fc020ff0855d311c690e3",
    }
)
EXPECTED_PAGE_IDS = frozenset(
    {
        "detail_696",
        "detail_933",
        "detail_1022",
        "detail_1207",
        "20160401-11",
        "20160630-02",
        "20180905-02",
        "20190205-01",
        "20260603-01",
    }
)
SUPPORT_PAGE_IDS = frozenset({"20180905-02", "20190205-01", "20260603-01"})
EXPECTED_SOURCE_IDS = frozenset(
    {f"{page_id}-page" for page_id in EXPECTED_PAGE_IDS}
    | {f"{page_id}-support" for page_id in SUPPORT_PAGE_IDS}
)
PRODUCER_OUTPUTS = frozenset(
    {
        "input_event_rows.csv",
        "timestamp_recovery.csv",
        "accepted_timestamp_recovery.csv",
        "rejected_timestamp_recovery.csv",
        "official_source_manifest.csv",
        "quarantine/rows.csv",
        "summary.md",
    }
)
VALIDATION_OUTPUTS = frozenset(
    {"claim_validation.csv", "source_validation.csv", "validation_issues.csv"}
)
ACCEPTED_COLUMNS = (
    "event_id",
    "known_at_utc",
    "publication_date",
    "source_sha256",
    "source_url",
    "known_at_rule",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class TimestampSidecarAdmissionError(ValueError):
    """The timestamp sidecar failed a closed admission check."""


@dataclass(frozen=True)
class ValidatedJapanG4HaltTimestampSidecar:
    accepted: pd.DataFrame
    destination: Path
    copied_paths: tuple[Path, ...]
    lineage: dict[str, Any]

    @property
    def timestamp_rows(self) -> pd.DataFrame:
        return self.accepted


ValidatedHaltTimestampSidecar = ValidatedJapanG4HaltTimestampSidecar


@dataclass(frozen=True)
class _VerifiedPacket:
    producer: Path
    validation: Path
    accepted: pd.DataFrame
    producer_manifest: dict[str, Any]
    validation_manifest: dict[str, Any]
    producer_inventory: dict[str, str]
    validation_inventory: dict[str, str]
    lineage: dict[str, Any]


def verify_japan_g4_halt_timestamp_sidecar(
    producer_dir: str | Path,
    validation_dir: str | Path,
    *,
    existing_event_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Verify the official packet and return its normalized accepted rows."""
    packet = _verify_packet(Path(producer_dir), Path(validation_dir))
    return normalize_accepted_timestamp_sidecar(
        packet.accepted, existing_event_ids=existing_event_ids
    )


def copy_validated_japan_g4_halt_timestamp_sidecar(
    producer_dir: str | Path,
    validation_dir: str | Path,
    destination: str | Path,
    *,
    existing_event_ids: Iterable[str] = (),
) -> ValidatedJapanG4HaltTimestampSidecar:
    """Verify and atomically seal both immutable trees under ``destination``."""
    packet = _verify_packet(Path(producer_dir), Path(validation_dir))
    accepted = normalize_accepted_timestamp_sidecar(
        packet.accepted, existing_event_ids=existing_event_ids
    )
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        raise TimestampSidecarAdmissionError("destination_already_exists")
    if _paths_overlap(target, packet.producer) or _paths_overlap(target, packet.validation):
        raise TimestampSidecarAdmissionError("destination_overlaps_source_packet")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink() or not target.parent.is_dir():
        raise TimestampSidecarAdmissionError("destination_parent_is_unsafe")

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    copied_relatives: list[Path] = []
    try:
        for label, source, inventory in (
            ("producer", packet.producer, packet.producer_inventory),
            ("validation", packet.validation, packet.validation_inventory),
        ):
            for relative, expected_hash in sorted(inventory.items()):
                payload = _read_safe_file(source, relative)
                if _sha256_bytes(payload) != expected_hash:
                    raise TimestampSidecarAdmissionError(
                        f"source_changed_during_copy:{label}/{relative}"
                    )
                copied = temporary / label / relative
                copied.parent.mkdir(parents=True, exist_ok=True)
                with copied.open("xb") as handle:
                    handle.write(payload)
                if _sha256_file(copied) != expected_hash:
                    raise TimestampSidecarAdmissionError(
                        f"sealed_copy_hash_mismatch:{label}/{relative}"
                    )
                copied_relatives.append(Path(label) / relative)

        lineage_path = temporary / "lineage.json"
        lineage_payload = (
            json.dumps(packet.lineage, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with lineage_path.open("xb") as handle:
            handle.write(lineage_payload)
        copied_relatives.append(Path("lineage.json"))
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    copied = tuple(target / relative for relative in copied_relatives)
    return ValidatedJapanG4HaltTimestampSidecar(
        accepted=accepted,
        destination=target,
        copied_paths=copied,
        lineage=packet.lineage,
    )


def admit_japan_g4_halt_timestamp_sidecar(
    producer_dir: str | Path,
    validation_dir: str | Path,
    destination: str | Path,
    *,
    existing_event_ids: Iterable[str] = (),
) -> ValidatedJapanG4HaltTimestampSidecar:
    """Public admission alias used by primary-application callers."""
    return copy_validated_japan_g4_halt_timestamp_sidecar(
        producer_dir,
        validation_dir,
        destination,
        existing_event_ids=existing_event_ids,
    )


def copy_validated_halt_timestamp_sidecar(
    producer_dir: str | Path,
    validation_dir: str | Path,
    destination: str | Path,
    *,
    existing_event_ids: Iterable[str] = (),
) -> ValidatedHaltTimestampSidecar:
    """Concise alias consistent with the other halt sidecar adapters."""
    return copy_validated_japan_g4_halt_timestamp_sidecar(
        producer_dir,
        validation_dir,
        destination,
        existing_event_ids=existing_event_ids,
    )


def admit_halt_timestamp_sidecar(
    producer_dir: str | Path,
    validation_dir: str | Path,
    destination: str | Path,
    *,
    existing_event_ids: Iterable[str] = (),
) -> ValidatedHaltTimestampSidecar:
    return copy_validated_halt_timestamp_sidecar(
        producer_dir,
        validation_dir,
        destination,
        existing_event_ids=existing_event_ids,
    )


def normalize_accepted_timestamp_sidecar(
    accepted_claims: pd.DataFrame,
    *,
    existing_event_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """Return the narrow primary-facing sidecar with conservative PIT provenance."""
    required = {
        "event_id",
        "known_at_utc",
        "publication_date",
        "page_sha256",
        "page_url",
        "known_at_rule",
        "publication_precision",
        "event_date",
        "status",
    }
    if not required.issubset(accepted_claims.columns):
        missing = sorted(required - set(accepted_claims.columns))
        raise TimestampSidecarAdmissionError(
            f"accepted_claim_columns_missing:{','.join(missing)}"
        )
    claims = accepted_claims.copy().fillna("")
    ids = claims["event_id"].astype(str)
    if len(claims) != len(EXPECTED_EVENT_IDS) or set(ids) != EXPECTED_EVENT_IDS:
        raise TimestampSidecarAdmissionError("accepted_event_denominator_invalid")
    if ids.duplicated().any():
        raise TimestampSidecarAdmissionError("duplicate_accepted_event_id")
    overlap = set(ids) & {str(value) for value in existing_event_ids}
    if overlap:
        raise TimestampSidecarAdmissionError(
            "existing_event_id_overlap:" + ",".join(sorted(overlap))
        )

    normalized: list[dict[str, str]] = []
    for row in claims.to_dict("records"):
        event_id = str(row["event_id"])
        publication_date = _strict_date(row["publication_date"], event_id)
        event_date = _strict_date(row["event_date"], event_id)
        known_at = _strict_utc(row["known_at_utc"], event_id)
        expected_known_at = datetime.combine(
            publication_date + timedelta(days=1), time(), JST
        ).astimezone(UTC)
        event_date_end = datetime.combine(event_date, time.max, JST).astimezone(UTC)
        source_hash = str(row["page_sha256"])
        source_url = str(row["page_url"])
        if (
            str(row["status"]) != "accepted"
            or str(row["publication_precision"]) != PUBLICATION_PRECISION
            or str(row["known_at_rule"]) != CONSERVATIVE_DATE_ONLY_RULE
            or known_at != expected_known_at
        ):
            raise TimestampSidecarAdmissionError(
                f"conservative_date_only_rule_invalid:{event_id}"
            )
        if known_at > event_date_end:
            raise TimestampSidecarAdmissionError(f"known_at_after_event_date:{event_id}")
        if not SHA256_PATTERN.fullmatch(source_hash):
            raise TimestampSidecarAdmissionError(f"source_sha256_invalid:{event_id}")
        parsed_url = urllib.parse.urlparse(source_url)
        if not (
            parsed_url.scheme == "https"
            and parsed_url.hostname == "www.jpx.co.jp"
            and not parsed_url.username
            and not parsed_url.password
        ):
            raise TimestampSidecarAdmissionError(f"source_url_invalid:{event_id}")
        normalized.append(
            {
                "event_id": event_id,
                "known_at_utc": known_at.isoformat(),
                "publication_date": publication_date.isoformat(),
                "source_sha256": source_hash,
                "source_url": source_url,
                "known_at_rule": CONSERVATIVE_DATE_ONLY_RULE,
            }
        )
    return (
        pd.DataFrame(normalized, columns=ACCEPTED_COLUMNS)
        .sort_values("event_id")
        .reset_index(drop=True)
    )


def _verify_packet(producer_input: Path, validation_input: Path) -> _VerifiedPacket:
    producer = _safe_root(producer_input, "producer")
    validation = _safe_root(validation_input, "validation")
    if _paths_overlap(producer, validation):
        raise TimestampSidecarAdmissionError("producer_validation_tree_overlap")

    producer_manifest_path = _safe_path(producer, "manifest.json")
    validation_manifest_path = _safe_path(validation, "manifest.json")
    producer_manifest = _read_json(producer_manifest_path, "producer_manifest")
    validation_manifest = _read_json(validation_manifest_path, "validation_manifest")
    producer_sha = _sha256_file(producer_manifest_path)
    validation_sha = _sha256_file(validation_manifest_path)

    _verify_manifest_contracts(
        producer_manifest,
        validation_manifest,
        producer_manifest_path,
        producer_sha,
    )
    producer_outputs = _verify_declared_outputs(
        producer, producer_manifest.get("outputs"), PRODUCER_OUTPUTS, "producer"
    )
    validation_outputs = _verify_declared_outputs(
        validation,
        validation_manifest.get("outputs"),
        VALIDATION_OUTPUTS,
        "validation",
    )

    claims = _csv(_safe_path(producer, "timestamp_recovery.csv"), "claims")
    accepted = _csv(
        _safe_path(producer, "accepted_timestamp_recovery.csv"), "accepted"
    )
    rejected = _csv(
        _safe_path(producer, "rejected_timestamp_recovery.csv"), "rejected"
    )
    quarantine = _csv(
        _safe_path(producer, "quarantine/rows.csv"), "quarantine"
    )
    sources = _csv(
        _safe_path(producer, "official_source_manifest.csv"), "sources"
    )
    _verify_counts_and_partitions(
        producer_manifest, validation_manifest, claims, accepted, rejected, quarantine, sources
    )
    raw_inventory = _verify_sources(producer, producer_manifest, sources)
    _verify_claim_source_links(claims, sources)
    _verify_validation_rows(validation, validation_manifest, claims, sources)

    producer_inventory = {
        "manifest.json": producer_sha,
        **producer_outputs,
        **raw_inventory,
    }
    validation_inventory = {
        "manifest.json": validation_sha,
        **validation_outputs,
    }
    _verify_sealed_tree(producer, producer_inventory, "producer")
    _verify_sealed_tree(validation, validation_inventory, "validation")

    lineage = {
        "sidecar_run_id": OFFICIAL_PRODUCER_RUN_ID,
        "producer_run_id": producer_manifest["run_id"],
        "producer_manifest_sha256": producer_sha,
        "validation_run_id": validation_manifest["run_id"],
        "validation_manifest_sha256": validation_sha,
        "target_events": len(EXPECTED_EVENT_IDS),
        "source_responses": len(EXPECTED_SOURCE_IDS),
        "publication_precision": PUBLICATION_PRECISION,
        "known_at_rule": CONSERVATIVE_DATE_ONLY_RULE,
        "refinitiv_calls": False,
    }
    return _VerifiedPacket(
        producer=producer,
        validation=validation,
        accepted=accepted,
        producer_manifest=producer_manifest,
        validation_manifest=validation_manifest,
        producer_inventory=producer_inventory,
        validation_inventory=validation_inventory,
        lineage=lineage,
    )


def _verify_manifest_contracts(
    producer: Mapping[str, Any],
    validation: Mapping[str, Any],
    producer_manifest_path: Path,
    producer_sha: str,
) -> None:
    producer_policy = producer.get("source_policy")
    if not isinstance(producer_policy, Mapping):
        raise TimestampSidecarAdmissionError("producer_source_policy_invalid")
    prohibited = (
        "http_response_date_used_as_publication_time",
        "http_last_modified_used_as_publication_time",
        "archive_capture_time_used",
        "report_month_used",
        "same_code_prior_observation_used",
        "inferred_publication_time_used",
    )
    producer_ok = bool(
        producer.get("schema_version") == "1.0-v8-bounded-immutable"
        and producer.get("run_id") == OFFICIAL_PRODUCER_RUN_ID
        and producer.get("component") == PRODUCER_COMPONENT
        and producer.get("status") == "complete"
        and producer.get("refinitiv_calls") is False
        and producer.get("primary_halt_history_modified") is False
        and producer.get("independent_validation_required") is True
        and producer.get("promotion_eligible") is False
        and producer_policy.get("official_jpx_only") is True
        and producer_policy.get("exact_page_count") == len(EXPECTED_PAGE_IDS)
        and producer_policy.get("known_at_rule") == CONSERVATIVE_DATE_ONLY_RULE
        and all(producer_policy.get(key) is False for key in prohibited)
    )
    if not producer_ok:
        raise TimestampSidecarAdmissionError("producer_manifest_contract_invalid")

    descriptor = validation.get("producer_manifest")
    descriptor_ok = isinstance(descriptor, Mapping) and _descriptor_matches_exact_path(
        producer_manifest_path, descriptor
    )
    validation_ok = bool(
        validation.get("schema_version")
        == "1.0-independent-retained-byte-validation"
        and validation.get("run_id") == OFFICIAL_VALIDATION_RUN_ID
        and validation.get("component") == VALIDATION_COMPONENT
        and validation.get("producer_run_id") == OFFICIAL_PRODUCER_RUN_ID
        and validation.get("status") == "pass"
        and validation.get("validation_issues") == []
        and validation.get("independent_implementation") is True
        and validation.get("retained_bytes_reparsed") is True
        and validation.get("refinitiv_calls") is False
        and validation.get("promotion_eligible") is False
        and descriptor_ok
        and descriptor.get("sha256") == producer_sha
    )
    if not validation_ok:
        raise TimestampSidecarAdmissionError("producer_validator_lineage_invalid")


def _verify_declared_outputs(
    root: Path,
    declared_value: object,
    expected_names: frozenset[str],
    label: str,
) -> dict[str, str]:
    if not isinstance(declared_value, Mapping) or set(declared_value) != expected_names:
        raise TimestampSidecarAdmissionError(f"{label}_declared_output_set_invalid")
    inventory: dict[str, str] = {}
    for relative, descriptor in declared_value.items():
        path = _safe_path(root, relative)
        if not isinstance(descriptor, Mapping) or not _descriptor_matches_exact_path(
            path, descriptor
        ):
            raise TimestampSidecarAdmissionError(
                f"{label}_declared_output_invalid:{relative}"
            )
        inventory[str(relative)] = str(descriptor["sha256"])
    return inventory


def _verify_counts_and_partitions(
    producer_manifest: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
    claims: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    quarantine: pd.DataFrame,
    sources: pd.DataFrame,
) -> None:
    required_claim_columns = {
        "event_id",
        "event_date",
        "publication_date",
        "publication_precision",
        "known_at_utc",
        "known_at_rule",
        "page_id",
        "page_url",
        "page_source_path",
        "page_sha256",
        "support_url",
        "support_source_path",
        "support_sha256",
        "status",
    }
    required_source_columns = {
        "source_id",
        "page_id",
        "source_role",
        "requested_url",
        "final_url",
        "http_status",
        "content_type",
        "raw_path",
        "sha256",
        "bytes",
        "metadata_path",
        "metadata_sha256",
        "accepted",
    }
    if not required_claim_columns.issubset(claims.columns):
        raise TimestampSidecarAdmissionError("producer_claim_schema_invalid")
    if not required_source_columns.issubset(sources.columns):
        raise TimestampSidecarAdmissionError("producer_source_schema_invalid")
    claim_ids = claims["event_id"].astype(str)
    accepted_ids = accepted.get("event_id", pd.Series(dtype=str)).astype(str)
    rejected_ids = rejected.get("event_id", pd.Series(dtype=str)).astype(str)
    if (
        len(claims) != 10
        or set(claim_ids) != EXPECTED_EVENT_IDS
        or claim_ids.duplicated().any()
        or len(accepted) != 10
        or set(accepted_ids) != EXPECTED_EVENT_IDS
        or accepted_ids.duplicated().any()
        or len(rejected) != 0
        or len(quarantine) != 0
        or set(accepted_ids) & set(rejected_ids)
        or not claims["status"].eq("accepted").all()
    ):
        raise TimestampSidecarAdmissionError("producer_event_partition_invalid")
    if not accepted.reset_index(drop=True).equals(
        claims.loc[claims["status"].eq("accepted")].reset_index(drop=True)
    ):
        raise TimestampSidecarAdmissionError("accepted_claim_copy_mismatch")
    source_ids = sources["source_id"].astype(str)
    if (
        len(sources) != 12
        or set(source_ids) != EXPECTED_SOURCE_IDS
        or source_ids.duplicated().any()
        or set(sources["page_id"].astype(str)) != EXPECTED_PAGE_IDS
        or not sources["accepted"].eq("True").all()
    ):
        raise TimestampSidecarAdmissionError("producer_source_denominator_invalid")

    producer_counts = producer_manifest.get("counts")
    validation_counts = validation_manifest.get("counts")
    expected_producer_counts = {
        "target_events": 10,
        "exact_pages": 9,
        "linked_support_documents": 3,
        "retained_source_responses": 12,
        "accepted_events": 10,
        "rejected_events": 0,
        "accepted_source_responses": 12,
        "quarantined_source_responses": 0,
        "publication_time_stated_events": 0,
        "publication_date_only_events": 10,
    }
    expected_validation_counts = {
        "target_events": 10,
        "producer_accepted_events": 10,
        "producer_rejected_events": 0,
        "independently_validated_events": 10,
        "source_responses": 12,
        "validated_source_responses": 12,
        "validation_issues": 0,
    }
    if not isinstance(producer_counts, Mapping) or any(
        producer_counts.get(key) != value
        for key, value in expected_producer_counts.items()
    ):
        raise TimestampSidecarAdmissionError("producer_exact_count_mismatch")
    if not isinstance(validation_counts, Mapping) or any(
        validation_counts.get(key) != value
        for key, value in expected_validation_counts.items()
    ):
        raise TimestampSidecarAdmissionError("validation_exact_count_mismatch")


def _verify_sources(
    producer: Path, manifest: Mapping[str, Any], sources: pd.DataFrame
) -> dict[str, str]:
    raw_hashes = manifest.get("raw_source_sha256")
    metadata_hashes = manifest.get("response_metadata_sha256")
    if (
        not isinstance(raw_hashes, Mapping)
        or set(raw_hashes) != EXPECTED_SOURCE_IDS
        or not isinstance(metadata_hashes, Mapping)
        or set(metadata_hashes) != EXPECTED_SOURCE_IDS
    ):
        raise TimestampSidecarAdmissionError("source_hash_manifest_set_invalid")
    inventory: dict[str, str] = {}
    for row in sources.to_dict("records"):
        source_id = str(row["source_id"])
        page_id = str(row["page_id"])
        role = str(row["source_role"])
        expected_role = "linked_official_pdf" if source_id.endswith("-support") else "page_html"
        expected_source_id = f"{page_id}-{'support' if role == 'linked_official_pdf' else 'page'}"
        expected_content_type = (
            "application/pdf" if role == "linked_official_pdf" else "text/html"
        )
        raw_path = _declared_source_path(producer, row["raw_path"], "raw", source_id)
        metadata_path = _declared_source_path(
            producer, row["metadata_path"], "raw/metadata", source_id
        )
        raw_relative = raw_path.relative_to(producer).as_posix()
        metadata_relative = metadata_path.relative_to(producer).as_posix()
        raw_sha = str(row["sha256"])
        metadata_sha = str(row["metadata_sha256"])
        if (
            page_id not in EXPECTED_PAGE_IDS
            or role != expected_role
            or source_id != expected_source_id
            or str(row["http_status"]) != "200"
            or str(row["content_type"]).split(";", 1)[0].strip().lower()
            != expected_content_type
            or str(row["requested_url"]) != str(row["final_url"])
            or not _official_url(str(row["requested_url"]))
            or not SHA256_PATTERN.fullmatch(raw_sha)
            or not SHA256_PATTERN.fullmatch(metadata_sha)
            or raw_hashes.get(source_id) != raw_sha
            or metadata_hashes.get(source_id) != metadata_sha
            or _sha256_file(raw_path) != raw_sha
            or _sha256_file(metadata_path) != metadata_sha
            or raw_path.stat().st_size != _strict_int(row["bytes"], source_id)
        ):
            raise TimestampSidecarAdmissionError(f"source_contract_invalid:{source_id}")
        metadata = _read_json(metadata_path, f"response_metadata:{source_id}")
        if not (
            metadata.get("source_id") == source_id
            and metadata.get("requested_url") == row["requested_url"]
            and metadata.get("final_url") == row["final_url"]
            and str(metadata.get("http_status")) == str(row["http_status"])
            and metadata.get("payload_sha256") == raw_sha
            and metadata.get("payload_bytes") == _strict_int(row["bytes"], source_id)
            and metadata.get("accepted") is True
            and metadata.get("publication_timestamp_input") is False
        ):
            raise TimestampSidecarAdmissionError(
                f"response_metadata_contract_invalid:{source_id}"
            )
        inventory[raw_relative] = raw_sha
        inventory[metadata_relative] = metadata_sha
    return inventory


def _verify_claim_source_links(claims: pd.DataFrame, sources: pd.DataFrame) -> None:
    source_by_id = sources.set_index("source_id", drop=False).to_dict("index")
    for row in claims.to_dict("records"):
        page_id = str(row["page_id"])
        page = source_by_id.get(f"{page_id}-page")
        support = source_by_id.get(f"{page_id}-support")
        support_expected = page_id in SUPPORT_PAGE_IDS
        if not page or not (
            row["page_url"] == page["requested_url"]
            and row["page_source_path"] == page["raw_path"]
            and row["page_sha256"] == page["sha256"]
        ):
            raise TimestampSidecarAdmissionError(
                f"claim_page_source_link_invalid:{row['event_id']}"
            )
        support_valid = bool(
            support
            and row["support_url"] == support["requested_url"]
            and row["support_source_path"] == support["raw_path"]
            and row["support_sha256"] == support["sha256"]
        )
        if support_expected != support_valid:
            raise TimestampSidecarAdmissionError(
                f"claim_support_source_link_invalid:{row['event_id']}"
            )
        if not support_expected and any(
            str(row[column])
            for column in ("support_url", "support_source_path", "support_sha256")
        ):
            raise TimestampSidecarAdmissionError(
                f"unexpected_claim_support_source:{row['event_id']}"
            )


def _verify_validation_rows(
    validation: Path,
    manifest: Mapping[str, Any],
    claims: pd.DataFrame,
    sources: pd.DataFrame,
) -> None:
    claim_validation = _csv(
        _safe_path(validation, "claim_validation.csv"), "claim_validation"
    )
    source_validation = _csv(
        _safe_path(validation, "source_validation.csv"), "source_validation"
    )
    issues = _csv(
        _safe_path(validation, "validation_issues.csv"), "validation_issues"
    )
    claim_ids = claim_validation.get("event_id", pd.Series(dtype=str)).astype(str)
    source_ids = source_validation.get("source_id", pd.Series(dtype=str)).astype(str)
    required_claim_bools = (
        "v8_event_valid",
        "source_hashes_valid",
        "response_metadata_valid",
        "page_claims_valid",
        "event_claims_valid",
        "publication_precision_valid",
        "known_at_valid",
        "partition_valid",
        "passed",
    )
    required_source_bools = (
        "hash_valid",
        "metadata_valid",
        "route_valid",
        "content_type_valid",
        "location_valid",
        "passed",
    )
    if (
        set(claim_validation.columns).isdisjoint(required_claim_bools)
        or any(column not in claim_validation for column in required_claim_bools)
        or len(claim_validation) != len(claims)
        or set(claim_ids) != EXPECTED_EVENT_IDS
        or claim_ids.duplicated().any()
        or not claim_validation["producer_status"].eq("accepted").all()
        or any(not _all_true(claim_validation[column]) for column in required_claim_bools)
        or len(source_validation) != len(sources)
        or set(source_ids) != EXPECTED_SOURCE_IDS
        or source_ids.duplicated().any()
        or any(column not in source_validation for column in required_source_bools)
        or any(not _all_true(source_validation[column]) for column in required_source_bools)
        or len(issues) != 0
        or manifest.get("validation_issues") != []
    ):
        raise TimestampSidecarAdmissionError("independent_validation_rows_invalid")


def _verify_sealed_tree(root: Path, inventory: Mapping[str, str], label: str) -> None:
    actual: set[str] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories) + list(files):
            path = current_path / name
            if path.is_symlink():
                raise TimestampSidecarAdmissionError(
                    f"{label}_tree_symlink_rejected:{path.relative_to(root).as_posix()}"
                )
        for name in files:
            path = current_path / name
            mode = path.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode):
                raise TimestampSidecarAdmissionError(
                    f"{label}_tree_nonregular_file:{path.relative_to(root).as_posix()}"
                )
            actual.add(path.relative_to(root).as_posix())
    if actual != set(inventory):
        raise TimestampSidecarAdmissionError(f"{label}_tree_is_not_sealed")
    for relative, expected_hash in inventory.items():
        if _sha256_file(_safe_path(root, relative)) != expected_hash:
            raise TimestampSidecarAdmissionError(
                f"{label}_sealed_hash_mismatch:{relative}"
            )


def _safe_root(path: Path, label: str) -> Path:
    absolute = path.absolute()
    if absolute.is_symlink() or not absolute.is_dir():
        raise TimestampSidecarAdmissionError(f"{label}_root_is_unsafe")
    return absolute.resolve()


def _safe_path(root: Path, relative_value: object) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise TimestampSidecarAdmissionError(f"unsafe_relative_path:{relative_value}")
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise TimestampSidecarAdmissionError(
                f"symlink_path_rejected:{relative.as_posix()}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise TimestampSidecarAdmissionError(
            f"declared_file_missing:{relative.as_posix()}"
        ) from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise TimestampSidecarAdmissionError(f"unsafe_declared_file:{relative.as_posix()}")
    return resolved


def _declared_source_path(
    root: Path, declared_value: object, required_parent: str, source_id: str
) -> Path:
    declared = Path(str(declared_value))
    if not declared.is_absolute():
        raise TimestampSidecarAdmissionError(f"source_path_not_absolute:{source_id}")
    expected_parent = root / required_parent
    try:
        relative = declared.relative_to(root)
    except ValueError as exc:
        raise TimestampSidecarAdmissionError(f"source_path_escape:{source_id}") from exc
    path = _safe_path(root, relative.as_posix())
    if path.parent.resolve() != expected_parent.resolve():
        raise TimestampSidecarAdmissionError(f"source_path_location_invalid:{source_id}")
    return path


def _descriptor_matches_exact_path(path: Path, descriptor: Mapping[str, Any]) -> bool:
    try:
        declared_path = Path(str(descriptor.get("path", "")))
        return bool(
            declared_path.is_absolute()
            and declared_path.absolute() == path.absolute()
            and SHA256_PATTERN.fullmatch(str(descriptor.get("sha256", "")))
            and descriptor.get("sha256") == _sha256_file(path)
            and _strict_int(descriptor.get("bytes"), str(path)) == path.stat().st_size
        )
    except (OSError, TimestampSidecarAdmissionError):
        return False


def _read_safe_file(root: Path, relative: str) -> bytes:
    path = _safe_path(root, relative)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(mode):
            raise TimestampSidecarAdmissionError(f"nonregular_source_file:{relative}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TimestampSidecarAdmissionError(f"{label}_invalid_json") from exc
    if not isinstance(value, dict):
        raise TimestampSidecarAdmissionError(f"{label}_not_an_object")
    return value


def _csv(path: Path, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise TimestampSidecarAdmissionError(f"{label}_invalid_csv") from exc


def _strict_date(value: object, event_id: str) -> date:
    text = str(value)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise TimestampSidecarAdmissionError(f"date_invalid:{event_id}") from exc
    if parsed.isoformat() != text:
        raise TimestampSidecarAdmissionError(f"date_not_canonical:{event_id}")
    return parsed


def _strict_utc(value: object, event_id: str) -> datetime:
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TimestampSidecarAdmissionError(f"known_at_invalid:{event_id}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise TimestampSidecarAdmissionError(f"known_at_not_utc:{event_id}")
    return parsed.astimezone(UTC)


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TimestampSidecarAdmissionError(f"integer_invalid:{label}")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise TimestampSidecarAdmissionError(f"integer_invalid:{label}") from exc
    if parsed < 0 or str(parsed) != str(value):
        raise TimestampSidecarAdmissionError(f"integer_invalid:{label}")
    return parsed


def _official_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname == "www.jpx.co.jp"
        and not parsed.username
        and not parsed.password
    )


def _all_true(series: pd.Series) -> bool:
    return series.astype(str).str.strip().str.casefold().eq("true").all()


def _paths_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return left_resolved == right_resolved or left_resolved.is_relative_to(
        right_resolved
    ) or right_resolved.is_relative_to(left_resolved)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
