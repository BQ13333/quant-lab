from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.data.japan_g4_halt_mapping_recovery_v8 import _denominator_frame


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scoped = pd.DataFrame(
        [{"primary_event_id": "event-1", "report_event_id": "report-1"}]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "report_event_id": "report-1",
                "event_date": "2013-02-28",
                "local_code": "3521",
                "issue_name": "ECONACH HOLDINGS CO.,LTD.",
            }
        ]
    )
    audit = pd.DataFrame(
        [
            {
                "event_id": "event-1",
                "refinitiv_ric": "",
                "mapping_status": "missing_admitted_mapping",
                "mapping_route": "no_unique_admitted_candidate",
            }
        ]
    )
    return scoped, events, audit


def test_v8_denominator_requires_exact_unresolved_mapping_status() -> None:
    scoped, events, audit = _fixtures()
    result = _denominator_frame(scoped, events, audit)
    assert result["event_id"].tolist() == ["event-1"]

    audit.loc[0, "refinitiv_ric"] = "3521.T"
    with pytest.raises(ValueError, match="mapped or non-missing"):
        _denominator_frame(scoped, events, audit)


def test_v8_denominator_rejects_event_id_drift() -> None:
    scoped, events, audit = _fixtures()
    scoped.loc[0, "primary_event_id"] = "event-2"
    with pytest.raises(ValueError, match="event IDs do not reconcile"):
        _denominator_frame(scoped, events, audit)
