import pandas as pd
import pytest
from quant_lab.markets.rules import *
D='2026-11-16'
def entry(value,when,kind='PRIMARY'): return dict(value=value,effective_from=when,source_url='fixture://test',source_kind=kind)
def test_rule_lookup_is_point_in_time():
    r=MarketRules({'MY':{'tick':[entry(1,'2000-01-01'),entry(2,'2000-02-01')]}})
    assert r.for_date('MY','2000-01-15')['tick']==1
    assert r.for_date('MY','2000-02-01')['tick']==2

def test_secondary_sourced_rule_is_inactive_for_admission():
    r=MarketRules({'CN':{'x':[entry(1,'2000-01-01'),entry(2,'2000-02-01','SECONDARY')]}})
    assert 'x' not in r.for_date('CN',D)

def test_st_names_excluded_while_rule_unverified():
    for d in ('2020-01-01',D):
        assert not MarketRules().eligible_name('CN','*ST name',d)

def test_bursa_statutory_legs_6_50_at_5k():
    assert explicit_cost('MY','BUY',5000,D)==6.5

def test_hk_levies_0_0085_per_side():
    r=MarketRules().for_date('HK',D)
    assert sum(r[k] for k in ('trading_rate','sfc_rate','afrc_rate'))==pytest.approx(.000085)

def test_hk_settlement_minimum_applies_at_15k():
    assert explicit_cost('HK','BUY',15000,D)==pytest.approx(15+1.275+2)

def test_a_share_stamp_sell_only():
    # Unverified northbound legs block real schedules; isolate stamp with fixture legs.
    entries=MarketRules().entries
    for k in ('handling_rate','management_rate','transfer_rate'): entries['CN'][k]=[entry(0,'2000-01-01')]
    r=MarketRules(entries)
    assert explicit_cost('CN','SELL',15000,D,rules=r)-explicit_cost('CN','BUY',15000,D,rules=r)==7.5
    with pytest.raises(ValueError): explicit_cost('CN','BUY',15000,D)

def test_lab_contract_label_refused_by_trading_gate():
    c=LabCostContract(); c.require_stage('PAPER')
    with pytest.raises(ValueError): c.require_stage('LIVE')

def test_no_contract_without_commission_entry():
    with pytest.raises(ValueError): LabCostContract().cost('HK','BUY',15000,D,spread=0,impact=0)

def test_missing_asset_schedule_raises():
    from quant_lab.portfolio.costs import total_cost
    with pytest.raises(ValueError): total_cost(pd.DataFrame({'a':[1.]}),.001,country_schedule={})


def test_live_promote_rejects_lab_cost_label():
    from quant_lab.validation.gates import ValidationGate
    row=dict(sharpe=2,deflated_sharpe=.99,annual_excess_return=.1,information_ratio=1,max_drawdown=-.01,turnover=1,cost_label='LAB_SIMULATED_COST')
    assert not ValidationGate().promote(row).passed


def test_registered_bursa_lab_commission_is_explicit_assumption():
    c=LabCostContract(commissions={'MY':Commission(.0042,12,'D1/C3: assisted rate; RM12 minimum is a lab assumption')})
    assert c.cost('MY','BUY',5000,D,spread=.001,impact=.001)==pytest.approx(6.5+21+10)
    assert c.label=='LAB_SIMULATED_COST'
