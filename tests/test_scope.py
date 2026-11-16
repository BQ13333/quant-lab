import pytest
from quant_lab.models.signal_factory import *

@pytest.mark.parametrize('market',['US','JP','XX'])
def test_us_jp_rejected_at_admission(market):
    with pytest.raises(ValueError): build_signal_models(SignalFactoryConfig(admission_markets=(market,)))

def test_unknown_profile_raises():
    with pytest.raises(ValueError): build_signal_models(SignalFactoryConfig(profile='typo'))


def test_research_gate_rejects_out_of_scope_dataset(tmp_path):
    from dataclasses import replace
    from test_validation_firewall import sealed
    from quant_lab.research.trial_ledger import TrialLedger
    from quant_lab.validation.firewall import evaluate_research_gate
    ledger=TrialLedger(tmp_path/'l.db',tmp_path/'e')
    with pytest.raises(ValueError):
        evaluate_research_gate(replace(sealed(),markets=('JP',)),candidate_id='stable',ledger=ledger,observed_trial_sharpes=[])
