from dataclasses import replace
import pytest
from quant_lab.research.strategy_card import StrategyCard
SPEC=dict(signals=['value'],weights=[1.],tilts=[],quotas={},stage_constructions=[.1,.25,1.])
CARD=StrategyCard(1,.05,.6,SPEC,'COMPOSITE')

def test_card_rejects_inconsistent_turnover():
    CARD.validate()
    for card in (replace(CARD,annual_turnover=1),):
        with pytest.raises(ValueError): card.validate()

def test_card_requires_full_composite_spec():
    for card in (replace(CARD,composite_spec={}),replace(CARD,evidence_grade='COMPONENT'),replace(CARD,composite_spec={**SPEC,'evidence_scope':'COMPONENT'})):
        with pytest.raises(ValueError): card.validate()


def test_turnover_equivalent_is_derived_not_input():
    assert CARD.turnover_implied_holding_months == 20
    with pytest.raises(TypeError): replace(CARD,avg_holding_months=20)

def test_core_plus_fringe_card_is_valid():
    from quant_lab.research.strategy_card import HoldingMeasurement
    replace(CARD, measured_holding=HoldingMeasurement('FIFO_LOTS','ENTRY_VALUE','RIGHT_KM',None,.8)).validate()

def test_measured_holding_requires_convention():
    from quant_lab.research.strategy_card import HoldingMeasurement
    with pytest.raises(ValueError):
        replace(CARD,measured_holding=HoldingMeasurement('','','',1.,0.)).validate()

def test_open_lots_are_censored_not_dropped():
    from quant_lab.research.strategy_card import measure_holding
    m=measure_holding([(2,20,True),(24,80,False)])
    assert m.censored_share == .8 and m.median_months is None
