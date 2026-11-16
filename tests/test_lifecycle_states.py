import pytest
from quant_lab.lifecycle.states import *


def test_json_typed_roundtrip():
    payload = dict(value=Value(.2), missing=Missing(), na=NotApplicable(), no=NoObservationExpected(),
                   permission=Permission.BLOCK, lifecycle=Lifecycle.RAMP,
                   schedule=Schedule((('ic', Expectation.DUE),)))
    assert from_json(to_json(payload)) == payload


@pytest.mark.parametrize('bad', [Missing(), NotApplicable(), NoObservationExpected(), Value(float('nan')), Value(float('inf'))])
def test_due_fails(bad):
    assert validate_inputs(Schedule((('ic', Expectation.DUE),)), {'ic': bad}) == Permission.BLOCK


def test_schedule_sentinels():
    schedule = Schedule((('ic', Expectation.NO_OBSERVATION_EXPECTED), ('cost', Expectation.NOT_APPLICABLE)))
    assert validate_inputs(schedule, dict(ic=NoObservationExpected(), cost=NotApplicable())) == Permission.OK
    assert validate_inputs(schedule, dict(ic=NotApplicable(), cost=NotApplicable())) == Permission.BLOCK
    assert validate_inputs(schedule, {}) == Permission.BLOCK
