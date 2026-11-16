from dataclasses import replace
import pytest
from quant_lab.research.strategy_card import StrategyCard
SPEC=dict(signals=['value'],weights=[1.],tilts=[],quotas={},stage_constructions=[.1,.25,1.])
CARD=StrategyCard(1,.05,.6,20,SPEC,'COMPOSITE')

def test_card_rejects_inconsistent_holding_and_turnover():
    CARD.validate()
    for card in (replace(CARD,avg_holding_months=12),replace(CARD,annual_turnover=1)):
        with pytest.raises(ValueError): card.validate()

def test_card_requires_full_composite_spec():
    for card in (replace(CARD,composite_spec={}),replace(CARD,evidence_grade='COMPONENT'),replace(CARD,composite_spec={**SPEC,'evidence_scope':'COMPONENT'})):
        with pytest.raises(ValueError): card.validate()
