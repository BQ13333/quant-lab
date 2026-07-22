from __future__ import annotations

from datetime import date

import pandas as pd

from quant_lab.data.japan_g4_halt_identity_recovery import (
    POST_EVENT_ONLY_CODES,
    SOURCE_SPECS,
    TARGET_CODES,
    _build_completion_plan,
    _build_results,
    _conservative_next_day,
    _verify_source_content,
)


def test_bounded_fail_closed_classification_rejects_successor_and_post_event() -> None:
    event_dates = {
        "3313": "2018-09-26",
        "5289": "2018-09-26",
        "3654": "2019-02-26",
        "6697": "2025-05-29",
        "8648": "2017-05-01",
        "6534": "2018-10-26",
        "8685": "2018-12-03",
        "8686": "2019-10-03",
    }
    events = pd.DataFrame(
        [
            {
                "event_id": f"event-{code}",
                "event_type": "delisting_non_trading_interval",
                "local_code": code,
                "issue_name": f"Issue {code}",
                "event_date": event_dates[code],
                "mapping_status": (
                    "deterministic_reference_post_event"
                    if code in POST_EVENT_ONLY_CODES
                    else "missing_admitted_mapping"
                ),
                "refinitiv_ric": f"{code}.T^X" if code in POST_EVENT_ONLY_CODES else "",
            }
            for code in TARGET_CODES
        ]
    )
    gaps = pd.DataFrame(
        [
            {"local_code": code, "gap_type": gap_type}
            for code in TARGET_CODES
            for gap_type in (
                "historical_pit_timestamp_unresolved",
                "ric_mapping_not_historical_pit"
                if code in POST_EVENT_ONLY_CODES
                else "ric_mapping_unresolved",
            )
        ]
    )
    g1_map = pd.DataFrame(
        [
            {
                "local_code": code,
                "refinitiv_ric": successor,
                "listing_start": "2011-01-01",
                "listing_end": "2012-12-31",
                "right_open": False,
                "pit_usable": True,
                "known_at": "2011-02-01T00:00:00+00:00",
            }
            for code, successor in (("3313", "9278.T"), ("5289", "5290.T"), ("3654", "4433.T"))
        ]
    )
    security = pd.DataFrame(
        [
            {
                "local_code": code,
                "exchange_code": "TYO",
                "retire_date": event_dates[code],
                "candidate_ric": f"{code}.T^X",
            }
            for code in POST_EVENT_ONLY_CODES
        ]
    )
    transitions = pd.DataFrame(
        [
            {
                "old_code": code,
                "successor_code": successor,
                "successor_primary_rics": f"{successor}.T",
                "validation_pass": True,
            }
            for code, successor in (("3313", "9278"), ("5289", "5290"), ("3654", "4433"))
        ]
    )
    source_by_code = {}
    for spec in SOURCE_SPECS:
        for code in spec.codes:
            source_by_code[code] = {
                "evidence_kind": spec.evidence_kind,
                "publication_date": spec.publication_date,
                "source_url": spec.url,
                "sha256": "a" * 64,
            }

    results = _build_results(gaps, events, g1_map, security, transitions, source_by_code)
    plan = _build_completion_plan(results)

    assert len(results) == 8
    assert results["mapping_acceptance"].str.startswith("rejected_").all()
    assert set(results.loc[results["mapping_acceptance"].eq("rejected_successor_only"), "local_code"]) == {"3313", "5289", "3654"}
    assert set(results.loc[results["mapping_acceptance"].eq("rejected_post_event_only"), "local_code"]) == POST_EVENT_ONLY_CODES
    assert results.loc[results["local_code"].eq("6697"), "mapping_acceptance"].item() == "rejected_missing_identity"
    assert results["accepted_refinitiv_ric"].eq("").all()
    assert len(plan) == 8
    assert (~plan["promotion_eligible"]).all()


def test_official_source_contract_and_conservative_timestamp() -> None:
    spec = SOURCE_SPECS[3]
    publication = date.fromisoformat(spec.publication_date)
    body = (
        f"<html><body>{publication.strftime('%b %d, %Y')} "
        + " ".join(spec.required_tokens)
        + "</body></html>"
    ).encode()
    _verify_source_content(body, spec)
    assert _conservative_next_day("2017-03-30") == "2017-03-30T15:00:00+00:00"
