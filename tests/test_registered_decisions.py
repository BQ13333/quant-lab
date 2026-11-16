"""Ben's 16 Nov 2026 decisions are registered and do not loosen existing gates."""

from __future__ import annotations

from quant_lab.data.availability_evidence import Grade
from quant_lab.lifecycle import registered_decisions as d


def test_c17_is_registered_as_holm_report_only_not_blocking() -> None:
    assert d.C17_METHOD == "HOLM_STEP_DOWN"
    assert "yekutieli" not in d.C17_METHOD.lower()
    assert d.C17_MODE == "report-only"
    assert d.C17_BLOCKING is False


def test_charter_numbers_match_the_working_set() -> None:
    assert d.CHARTER_FLOORS_MYR == (1_485_000, 756_250, 312_500)
    assert d.ACTIVE_VOL == 0.0515
    assert d.INFORMATION_RATIO == 0.29
    assert d.BURSA_TICKET_COMMISSION_RATE == 0.0042
    assert d.BURSA_TICKET_COMMISSION_FLOOR_MYR == 12.0


def test_false_pause_target_is_per_sleeve_and_family_is_report_only() -> None:
    assert d.FALSE_PAUSE_SCOPE == "per_sleeve"
    assert d.FALSE_PAUSE_MAXIMUM_RATE == 0.20
    assert d.FALSE_PAUSE_HORIZON_MONTHS == 60
    assert d.FALSE_PAUSE_FAMILY_RATE_MODE == "report-only"
    assert d.FALSE_PAUSE_OBSERVED_MEAN_MAX == 0.1654041
    # A 20% cap is not a family FWER tightening and is not 5%.
    assert d.FALSE_PAUSE_MAXIMUM_RATE != 0.05


def test_c10_live_history_cost_excludes_vendor_pit_for_now() -> None:
    assert d.LIVE_HISTORY_COST_GRADES == frozenset({"PRIMARY", "CAPTURED"})
    assert d.VENDOR_PIT_STATUS == "later_option_not_admissible"
    assert "VENDOR_PIT" not in d.LIVE_HISTORY_COST_GRADES
    assert not hasattr(Grade, "VENDOR_PIT")
    admitted = {Grade.PRIMARY.value, Grade.CAPTURED.value}
    assert admitted == set(d.LIVE_HISTORY_COST_GRADES)


def test_no_spend_this_week_is_recorded() -> None:
    assert d.SPEND_THIS_WEEK == "none"
