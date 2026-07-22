from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup


UTC = timezone.utc
JST = ZoneInfo("Asia/Tokyo")
TARGET_CODES = ("3313", "5289", "3654", "6697", "8648", "6534", "8685", "8686")
TRANSITION_CODES = frozenset({"3313", "5289", "3654"})
POST_EVENT_ONLY_CODES = frozenset({"8648", "6534", "8685", "8686"})
DIRECT_UNRESOLVED_CODES = frozenset({"3313", "5289", "3654", "6697"})
MAPPING_GAP_TYPES = frozenset({"ric_mapping_unresolved", "ric_mapping_not_historical_pit"})
TIMESTAMP_GAP_TYPE = "historical_pit_timestamp_unresolved"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024

RESULT_COLUMNS = (
    "event_id",
    "local_code",
    "target_class",
    "issue_name",
    "event_date",
    "source_gap_types",
    "event_mapping_status",
    "event_candidate_ric",
    "official_evidence_kind",
    "official_publication_date",
    "conservative_known_at_utc",
    "official_source_url",
    "official_source_sha256",
    "g1_identity_windows",
    "g1_covering_windows",
    "g1_covering_rics",
    "transition_validation_status",
    "transition_successor_code",
    "transition_successor_ric",
    "candidate_evidence_timing",
    "mapping_acceptance",
    "accepted_refinitiv_ric",
    "acceptance_reason",
    "timestamp_acceptance",
    "result_status",
    "promotion_eligible",
    "next_action",
)
SOURCE_COLUMNS = (
    "source_id",
    "codes",
    "evidence_kind",
    "source_url",
    "publication_date",
    "raw_path",
    "sha256",
    "bytes",
    "hash_verified",
    "content_verified",
)
INPUT_COLUMNS = (
    "input_id",
    "path",
    "sha256",
    "expected_sha256",
    "hash_verified",
)
PLAN_COLUMNS = (
    "local_code",
    "event_id",
    "gap_types",
    "mapping_status",
    "timestamp_status",
    "blocking_evidence_class",
    "required_evidence",
    "completion_status",
    "promotion_eligible",
)
CHECK_COLUMNS = ("check", "passed", "detail")

Fetcher = Callable[[str, float], bytes]


@dataclass(frozen=True)
class JapanG4HaltIdentityRecoveryArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    results_path: Path
    completion_plan_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _SourceSpec:
    codes: tuple[str, ...]
    evidence_kind: str
    url: str
    publication_date: str
    required_tokens: tuple[str, ...]


SOURCE_SPECS = (
    _SourceSpec(
        ("3313", "5289"),
        "validated_corporate_transition_notice",
        "https://www.jpx.co.jp/english/news/1030/20180926-01.html",
        "2018-09-26",
        ("9278", "bookoff group holdings", "5290", "vertex corporation"),
    ),
    _SourceSpec(
        ("3654",),
        "validated_corporate_transition_notice",
        "https://www.jpx.co.jp/english/news/1030/20190226-01.html",
        "2019-02-26",
        ("4433", "hito-communications holdings"),
    ),
    _SourceSpec(
        ("6697",),
        "direct_delisting_decision",
        "https://www.jpx.co.jp/english/news/1023/20250503-12.html",
        "2025-05-03",
        ("6697", "techpoint", "may 29, 2025"),
    ),
    _SourceSpec(
        ("8648",),
        "direct_delisting_decision",
        "https://www.jpx.co.jp/english/news/1021/20170330-04.html",
        "2017-03-30",
        ("8648", "bank of america", "may 1, 2017"),
    ),
    _SourceSpec(
        ("6534",),
        "direct_delisting_decision",
        "https://www.jpx.co.jp/english/news/1021/20180926-12.html",
        "2018-09-26",
        ("6534", "d.a.consortium holdings", "oct. 26, 2018"),
    ),
    _SourceSpec(
        ("8685",),
        "direct_delisting_decision",
        "https://www.jpx.co.jp/english/news/1021/20181102-01.html",
        "2018-11-02",
        ("8685", "american international group", "dec. 03, 2018"),
    ),
    _SourceSpec(
        ("8686",),
        "direct_delisting_decision",
        "https://www.jpx.co.jp/english/news/1021/20190902_01.html",
        "2019-09-02",
        ("8686", "aflac incorporated", "oct. 03, 2019"),
    ),
)


def recover_japan_g4_halt_identities(
    *,
    run_id: str,
    unresolved_gaps_path: str | Path,
    transition_validation_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_identity_recovery",
    timeout: float = 30.0,
    fetcher: Fetcher | None = None,
) -> JapanG4HaltIdentityRecoveryArtifacts:
    """Investigate the bounded G4 identity set without mutating admitted artifacts."""
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    gaps_path = Path(unresolved_gaps_path).resolve()
    producer_dir = gaps_path.parent
    transition_dir = Path(transition_validation_dir).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    raw_dir = root / "raw" / "official_jpx"
    raw_dir.mkdir(parents=True, exist_ok=True)

    producer_manifest_path = producer_dir / "manifest.json"
    producer_manifest = _json(producer_manifest_path)
    _require_manifest_output(producer_manifest_path, gaps_path, "unresolved_gaps.csv")
    if producer_manifest.get("component") != "halt_suspension_events":
        raise ValueError("unresolved gaps are not from the G4 halt producer")

    input_paths = {
        "halt_producer_manifest": producer_manifest_path,
        "unresolved_gaps": gaps_path,
        "halt_events": producer_dir / "halt_suspension_events.parquet",
        "g1_reference_symbol_map": producer_dir / "inputs" / "g1_reference_symbol_map.csv",
        "g1_security_master_candidates": producer_dir / "inputs" / "g1_security_master_candidates.csv",
        "g1_lineage": producer_dir / "inputs" / "g1_lineage.json",
        "transition_validation_manifest": transition_dir / "manifest.json",
        "validated_transition_ledger": transition_dir / "validated_transition_ledger.csv",
    }
    for key in (
        "halt_suspension_events.parquet",
        "inputs/g1_reference_symbol_map.csv",
        "inputs/g1_security_master_candidates.csv",
        "inputs/g1_lineage.json",
    ):
        _require_manifest_output(producer_manifest_path, producer_dir / key, key)

    transition_manifest_path = input_paths["transition_validation_manifest"]
    transition_manifest = _json(transition_manifest_path)
    if not (
        transition_manifest.get("validation_status") == "pass"
        and transition_manifest.get("component_scorecard_eligible") is True
        and transition_manifest.get("independent_validation") is True
    ):
        raise ValueError("corporate transition evidence is not independently admitted")
    _require_manifest_output(
        transition_manifest_path,
        input_paths["validated_transition_ledger"],
        "validated_transition_ledger.csv",
    )
    _verify_g1_lineage(input_paths["g1_lineage"])

    gaps = pd.read_csv(gaps_path, dtype=str).fillna("")
    scoped_gaps = gaps.loc[gaps["local_code"].isin(TARGET_CODES)].copy()
    _validate_gap_scope(scoped_gaps)
    events = pd.read_parquet(input_paths["halt_events"]).fillna("")
    events = events.loc[events["local_code"].isin(TARGET_CODES)].copy()
    _validate_event_scope(events)
    g1_map = pd.read_csv(input_paths["g1_reference_symbol_map"], dtype=str).fillna("")
    security = pd.read_csv(input_paths["g1_security_master_candidates"], dtype=str).fillna("")
    transitions = pd.read_csv(input_paths["validated_transition_ledger"], dtype=str).fillna("")

    input_inventory = _input_inventory(input_paths, producer_manifest, transition_manifest)
    if not input_inventory["hash_verified"].all():
        failed = input_inventory.loc[~input_inventory["hash_verified"], "input_id"].tolist()
        raise ValueError(f"input hash verification failed: {failed}")

    source_manifest, source_by_code = _acquire_sources(
        raw_dir,
        timeout=timeout,
        fetcher=fetcher or _fetch,
    )
    results = _build_results(
        scoped_gaps,
        events,
        g1_map,
        security,
        transitions,
        source_by_code,
    )
    completion_plan = _build_completion_plan(results)
    checks = _independent_checks(
        scoped_gaps,
        results,
        completion_plan,
        source_manifest,
        input_inventory,
        raw_dir,
    )

    results_path = root / "identity_results.csv"
    completion_path = root / "completion_plan.csv"
    sources_path = root / "official_source_manifest.csv"
    inputs_path = root / "input_inventory.csv"
    checks_path = root / "independent_checks.csv"
    _atomic_csv(results_path, results)
    _atomic_csv(completion_path, completion_plan)
    _atomic_csv(sources_path, source_manifest)
    _atomic_csv(inputs_path, input_inventory)
    _atomic_csv(checks_path, checks)
    outputs = (results_path, completion_path, sources_path, inputs_path, checks_path)

    checks_pass = bool(checks["passed"].all())
    accepted_mappings = int(results["mapping_acceptance"].eq("accepted_historical_pit").sum())
    recovered_timestamps = int(
        results["timestamp_acceptance"].eq(
            "accepted_historical_pit_conservative_next_day"
        ).sum()
    )
    manifest = {
        "schema_version": "1.0-bounded-fail-closed",
        "run_id": run_id,
        "artifact_type": "japan_g4_halt_identity_recovery",
        "component": "halt_suspension_identity_recovery_plan",
        "status": "pass_recovery_plan" if checks_pass else "failed_independent_checks",
        "scope": {"local_codes": list(TARGET_CODES), "exact": True},
        "official_jpx_source": True,
        "refinitiv_calls": False,
        "hash_verified_inputs": bool(input_inventory["hash_verified"].all()),
        "independent_checks_passed": checks_pass,
        "scorecard_eligible": False,
        "component_scorecard_eligible": False,
        "promotion_eligible": False,
        "promotion_policy": {
            "alpha": False,
            "historical_backtest": False,
            "production": False,
            "ibkr": False,
            "live_trading": False,
        },
        "counts": {
            "target_codes": len(TARGET_CODES),
            "result_rows": len(results),
            "historical_pit_mapping_accepted": accepted_mappings,
            "mapping_rejected": len(results) - accepted_mappings,
            "historical_pit_timestamp_recovered": recovered_timestamps,
            "completion_plan_rows": len(completion_plan),
            "official_source_pages": len(source_manifest),
            "independent_checks": len(checks),
        },
        "acceptance_policy": {
            "mapping": "one admitted G1 window must cover the pre-delisting mapping day, be known by the event date, be PIT-usable, and carry a code-aligned Tokyo RIC",
            "successor_only": "never accepted as predecessor listing identity",
            "post_event_only": "never accepted as historical PIT mapping",
            "publication_date": "official JPX publication date is made conservative at 00:00 JST on the following calendar day",
        },
        "artifacts": {path.stem: str(path) for path in outputs},
        "input_sha256": {
            row["input_id"]: row["sha256"] for row in input_inventory.to_dict("records")
        },
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    if not checks_pass:
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise ValueError(f"independent checks failed: {failed}")
    return JapanG4HaltIdentityRecoveryArtifacts(
        run_id,
        root,
        manifest_path,
        results_path,
        completion_path,
        manifest,
    )


def _build_results(
    gaps: pd.DataFrame,
    events: pd.DataFrame,
    g1_map: pd.DataFrame,
    security: pd.DataFrame,
    transitions: pd.DataFrame,
    source_by_code: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for event in events.sort_values("local_code").to_dict("records"):
        code = event["local_code"]
        event_day = date.fromisoformat(event["event_date"])
        mapping_day = (event_day - timedelta(days=1)).isoformat()
        windows = g1_map.loc[g1_map["local_code"].eq(code)].copy()
        covering = _covering_g1_windows(windows, mapping_day, event["event_date"])
        code_aligned = covering.loc[
            covering["refinitiv_ric"].str.match(
                rf"^{re.escape(code)}\.T(?:\^.*)?$", case=False, na=False
            )
        ]
        accepted_ric = ""
        if len(code_aligned) == 1 and code_aligned["refinitiv_ric"].nunique() == 1:
            accepted_ric = code_aligned.iloc[0]["refinitiv_ric"]

        transition = transitions.loc[
            transitions.get("old_code", pd.Series(dtype=str)).eq(code)
            & transitions.get("validation_pass", pd.Series(dtype=str)).map(_truth)
        ]
        if len(transition) > 1:
            raise ValueError(f"multiple validated transition rows for {code}")
        transition_row = transition.iloc[0].to_dict() if len(transition) else {}
        transition_ric = str(transition_row.get("successor_primary_rics", ""))
        direct_post_event = _post_event_candidate(event, security)
        if accepted_ric:
            timing = "pre_event_admitted_g1_window"
            acceptance = "accepted_historical_pit"
            reason = "unique_code_aligned_tokyo_ric_in_event_covering_g1_window"
            next_action = "none"
        elif transition_row:
            timing = "successor_only_corporate_transition"
            acceptance = "rejected_successor_only"
            reason = "validated_transition_identifies_successor_not_predecessor_listing_ric"
            next_action = "acquire an admitted pre-event G1 window for the predecessor Tokyo listing RIC"
        elif direct_post_event or event.get("refinitiv_ric", ""):
            timing = "post_event_only"
            acceptance = "rejected_post_event_only"
            reason = "candidate_was_observed_after_the_event_without_a_covering_g1_window"
            next_action = "admit a contemporaneous code-to-Tokyo-RIC observation known no later than the event"
        else:
            timing = "none"
            acceptance = "rejected_missing_identity"
            reason = "no_admitted_code_aligned_tokyo_ric_candidate"
            next_action = "acquire an admitted pre-event security-master or official identifier observation"

        source = source_by_code[code]
        known_at = _conservative_next_day(source["publication_date"])
        gap_types = sorted(set(gaps.loc[gaps["local_code"].eq(code), "gap_type"]))
        candidate_ric = event.get("refinitiv_ric", "") or direct_post_event
        rows.append(
            {
                "event_id": event["event_id"],
                "local_code": code,
                "target_class": "post_event_only_mapping" if code in POST_EVENT_ONLY_CODES else "unresolved_local_code",
                "issue_name": event["issue_name"],
                "event_date": event["event_date"],
                "source_gap_types": "|".join(gap_types),
                "event_mapping_status": event["mapping_status"],
                "event_candidate_ric": candidate_ric,
                "official_evidence_kind": source["evidence_kind"],
                "official_publication_date": source["publication_date"],
                "conservative_known_at_utc": known_at,
                "official_source_url": source["source_url"],
                "official_source_sha256": source["sha256"],
                "g1_identity_windows": len(windows),
                "g1_covering_windows": len(covering),
                "g1_covering_rics": "|".join(sorted(set(covering["refinitiv_ric"]))) if len(covering) else "",
                "transition_validation_status": "pass" if transition_row else "not_applicable",
                "transition_successor_code": transition_row.get("successor_code", ""),
                "transition_successor_ric": transition_ric,
                "candidate_evidence_timing": timing,
                "mapping_acceptance": acceptance,
                "accepted_refinitiv_ric": accepted_ric,
                "acceptance_reason": reason,
                "timestamp_acceptance": "accepted_historical_pit_conservative_next_day",
                "result_status": "mapping_accepted_timestamp_recovered" if accepted_ric else "mapping_rejected_timestamp_recovered",
                "promotion_eligible": False,
                "next_action": next_action,
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).sort_values("local_code").reset_index(drop=True)


def _covering_g1_windows(
    windows: pd.DataFrame, mapping_day: str, event_date: str
) -> pd.DataFrame:
    if windows.empty:
        return windows
    mask = windows["listing_start"].ne("") & windows["listing_start"].le(mapping_day)
    right_open = windows.get("right_open", pd.Series(False, index=windows.index)).map(_truth)
    mask &= right_open | windows["listing_end"].eq("") | windows["listing_end"].ge(mapping_day)
    mask &= windows.get("pit_usable", pd.Series(False, index=windows.index)).map(_truth)
    mask &= windows["known_at"].ne("") & windows["known_at"].str[:10].le(event_date)
    return windows.loc[mask].copy()


def _post_event_candidate(event: dict[str, Any], security: pd.DataFrame) -> str:
    code = str(event["local_code"])
    if security.empty:
        return ""
    candidates = security.loc[
        security["local_code"].eq(code)
        & security["exchange_code"].eq("TYO")
        & security["retire_date"].eq(str(event["event_date"]))
        & security["candidate_ric"].str.match(
            rf"^{re.escape(code)}\.T(?:\^.*)?$", case=False, na=False
        )
    ]["candidate_ric"].drop_duplicates()
    return candidates.iloc[0] if len(candidates) == 1 else ""


def _build_completion_plan(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in results.to_dict("records"):
        if item["mapping_acceptance"] == "accepted_historical_pit":
            continue
        rows.append(
            {
                "local_code": item["local_code"],
                "event_id": item["event_id"],
                "gap_types": item["source_gap_types"],
                "mapping_status": item["mapping_acceptance"],
                "timestamp_status": item["timestamp_acceptance"],
                "blocking_evidence_class": item["candidate_evidence_timing"],
                "required_evidence": "unique admitted code-aligned Tokyo RIC window covering the day before delisting and known no later than the event date",
                "completion_status": "blocked_exact_pre_event_listing_identity_required",
                "promotion_eligible": False,
            }
        )
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def _acquire_sources(
    raw_dir: Path,
    *,
    timeout: float,
    fetcher: Fetcher,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    by_code: dict[str, dict[str, str]] = {}
    for spec in SOURCE_SPECS:
        _require_official_jpx_url(spec.url)
        payload = fetcher(spec.url, timeout)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ValueError(f"official source exceeds byte limit: {spec.url}")
        _verify_source_content(payload, spec)
        digest = hashlib.sha256(payload).hexdigest()
        source_id = "jpx-identity-" + hashlib.sha256(spec.url.encode("utf-8")).hexdigest()[:16]
        raw_path = raw_dir / f"{source_id}_{digest[:12]}.html"
        _atomic_bytes(raw_path, payload)
        row = {
            "source_id": source_id,
            "codes": "|".join(spec.codes),
            "evidence_kind": spec.evidence_kind,
            "source_url": spec.url,
            "publication_date": spec.publication_date,
            "raw_path": str(raw_path),
            "sha256": digest,
            "bytes": len(payload),
            "hash_verified": _sha256(raw_path) == digest,
            "content_verified": True,
        }
        rows.append(row)
        for code in spec.codes:
            by_code[code] = {key: str(value) for key, value in row.items()}
    if set(by_code) != set(TARGET_CODES):
        raise ValueError("official source registry does not cover the exact target set")
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS), by_code


def _verify_source_content(payload: bytes, spec: _SourceSpec) -> None:
    soup = BeautifulSoup(payload, "html.parser")
    normalized = _normalized_text(" ".join(soup.stripped_strings))
    missing = [token for token in spec.required_tokens if _normalized_text(token) not in normalized]
    publication = date.fromisoformat(spec.publication_date)
    date_tokens = {
        publication.strftime("%b. %d, %Y").lower(),
        publication.strftime("%b %d, %Y").lower(),
        publication.strftime("%B %d, %Y").lower(),
    }
    if missing:
        raise ValueError(f"official source content mismatch for {spec.url}: {missing}")
    if not any(_normalized_text(token) in normalized for token in date_tokens):
        raise ValueError(f"official source publication date missing: {spec.url}")


def _independent_checks(
    gaps: pd.DataFrame,
    results: pd.DataFrame,
    plan: pd.DataFrame,
    sources: pd.DataFrame,
    inputs: pd.DataFrame,
    raw_dir: Path,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    result_codes = set(results["local_code"])
    add("exact_target_scope", result_codes == set(TARGET_CODES) and len(results) == 8, "eight unique bounded codes")
    gap_groups = gaps.groupby("local_code")["gap_type"].agg(set).to_dict()
    gap_ok = all(
        TIMESTAMP_GAP_TYPE in gap_groups.get(code, set())
        and bool(MAPPING_GAP_TYPES & gap_groups.get(code, set()))
        for code in TARGET_CODES
    )
    add("source_gap_pair_present", gap_ok, "timestamp and mapping gaps recomputed per code")
    source_hash_ok = all(
        Path(row["raw_path"]).is_file()
        and _within(Path(row["raw_path"]).resolve(), raw_dir.resolve())
        and _sha256(Path(row["raw_path"])) == row["sha256"]
        for row in sources.to_dict("records")
    )
    add("official_source_hashes", source_hash_ok, f"{len(sources)} JPX pages re-hashed")
    add("input_hashes", bool(inputs["hash_verified"].all()), f"{len(inputs)} admitted inputs re-hashed")
    add(
        "official_source_allowlist",
        sources["source_url"].str.match(r"^https://(?:www\.)?jpx\.co\.jp/").all(),
        "all network evidence is under jpx.co.jp",
    )
    add(
        "historical_publication_order",
        all(
            date.fromisoformat(row["official_publication_date"])
            <= date.fromisoformat(row["event_date"])
            for row in results.to_dict("records")
        ),
        "publication date is no later than each event date",
    )
    accepted = results.loc[results["mapping_acceptance"].eq("accepted_historical_pit")]
    accepted_ok = all(
        int(row["g1_covering_windows"]) == 1
        and bool(re.match(rf"^{re.escape(row['local_code'])}\.T(?:\^.*)?$", row["accepted_refinitiv_ric"], re.I))
        for row in accepted.to_dict("records")
    )
    add("accepted_mapping_recompute", accepted_ok, f"{len(accepted)} accepted mappings meet the strict rule")
    rejected = results.loc[~results["mapping_acceptance"].eq("accepted_historical_pit")]
    add(
        "rejected_mapping_blank",
        rejected["accepted_refinitiv_ric"].eq("").all(),
        f"{len(rejected)} rejected rows carry no accepted RIC",
    )
    post_event = results.loc[results["local_code"].isin(POST_EVENT_ONLY_CODES)]
    add(
        "post_event_only_rejected",
        len(post_event) == 4
        and post_event["mapping_acceptance"].eq("rejected_post_event_only").all(),
        "four post-event-only mappings remain rejected",
    )
    successor = results.loc[results["local_code"].isin(TRANSITION_CODES)]
    add(
        "successor_only_rejected",
        len(successor) == 3
        and successor["mapping_acceptance"].eq("rejected_successor_only").all(),
        "three validated corporate transitions are not predecessor RIC evidence",
    )
    add(
        "completion_plan_exact",
        set(plan["local_code"]) == set(rejected["local_code"]) and len(plan) == len(rejected),
        "every rejected identity has one completion row",
    )
    add(
        "no_promotion",
        (~results["promotion_eligible"].map(_truth)).all()
        and (~plan["promotion_eligible"].map(_truth)).all(),
        "recovery lane cannot promote",
    )
    add(
        "no_refinitiv_network_source",
        not sources["source_url"].str.contains("refinitiv|lseg", case=False, regex=True).any(),
        "no Refinitiv or LSEG URL is called",
    )
    return pd.DataFrame(checks, columns=CHECK_COLUMNS)


def _input_inventory(
    paths: dict[str, Path],
    producer_manifest: dict[str, Any],
    transition_manifest: dict[str, Any],
) -> pd.DataFrame:
    expected = {
        "unresolved_gaps": producer_manifest.get("output_sha256", {}).get("unresolved_gaps.csv", ""),
        "halt_events": producer_manifest.get("output_sha256", {}).get("halt_suspension_events.parquet", ""),
        "g1_reference_symbol_map": producer_manifest.get("output_sha256", {}).get("inputs/g1_reference_symbol_map.csv", ""),
        "g1_security_master_candidates": producer_manifest.get("output_sha256", {}).get("inputs/g1_security_master_candidates.csv", ""),
        "g1_lineage": producer_manifest.get("output_sha256", {}).get("inputs/g1_lineage.json", ""),
        "validated_transition_ledger": transition_manifest.get("output_sha256", {}).get("validated_transition_ledger.csv", ""),
    }
    rows = []
    for input_id, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        expected_digest = expected.get(input_id, digest)
        rows.append(
            {
                "input_id": input_id,
                "path": str(path),
                "sha256": digest,
                "expected_sha256": expected_digest,
                "hash_verified": bool(expected_digest and digest == expected_digest),
            }
        )
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def _validate_gap_scope(gaps: pd.DataFrame) -> None:
    if set(gaps["local_code"]) != set(TARGET_CODES):
        raise ValueError("unresolved gaps do not contain the exact eight-code target set")
    for code in TARGET_CODES:
        kinds = set(gaps.loc[gaps["local_code"].eq(code), "gap_type"])
        if TIMESTAMP_GAP_TYPE not in kinds or not (kinds & MAPPING_GAP_TYPES):
            raise ValueError(f"target code lacks timestamp/mapping gap pair: {code}")


def _validate_event_scope(events: pd.DataFrame) -> None:
    if len(events) != len(TARGET_CODES) or set(events["local_code"]) != set(TARGET_CODES):
        raise ValueError("halt events do not contain exactly one event for each target code")
    if events["local_code"].duplicated().any():
        raise ValueError("target code has multiple halt events")
    if not events["event_type"].eq("delisting_non_trading_interval").all():
        raise ValueError("identity recovery accepts only delisting non-trading intervals")


def _verify_g1_lineage(path: Path) -> None:
    lineage = _json(path)
    for label in ("g1_producer_manifest", "g1_validation_manifest"):
        source = Path(str(lineage.get(label, "")))
        expected = str(lineage.get(f"{label}_sha256", ""))
        if not source.is_file() or not expected or _sha256(source) != expected:
            raise ValueError(f"G1 lineage hash mismatch: {label}")
    validation = _json(Path(lineage["g1_validation_manifest"]))
    if not (
        validation.get("validation_status") == "pass"
        and validation.get("scorecard_eligible") is True
        and validation.get("source_checksums_verified") is True
    ):
        raise ValueError("G1 identity windows are not independently admitted")


def _require_manifest_output(manifest_path: Path, output_path: Path, key: str) -> None:
    manifest = _json(manifest_path)
    expected = str(manifest.get("output_sha256", {}).get(key, ""))
    if not output_path.is_file() or not expected or _sha256(output_path) != expected:
        raise ValueError(f"manifest output hash mismatch: {key}")


def _require_official_jpx_url(url: str) -> None:
    if not re.fullmatch(r"https://www\.jpx\.co\.jp/[A-Za-z0-9_./-]+", url):
        raise ValueError(f"source URL is not allowlisted JPX HTTPS: {url}")


def _conservative_next_day(publication_date: str) -> str:
    day = date.fromisoformat(publication_date) + timedelta(days=1)
    return datetime.combine(day, time(), JST).astimezone(UTC).isoformat()


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _fetch(url: str, timeout: float) -> bytes:
    _require_official_jpx_url(url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "quant-lab-japan-g4-halt-identity-recovery/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(MAX_RESPONSE_BYTES + 1)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
