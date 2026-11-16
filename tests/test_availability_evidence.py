from dataclasses import replace
import pandas as pd
import pytest
from quant_lab.data.availability_evidence import *
CAL=pd.date_range('2026-01-01',periods=5,tz='UTC')
def record(grade=Grade.PRIMARY): return Record(3,grade,CAL[0],CAL[2],'v1')
def test_agreement_alone_is_not_admissible():
    assert pd.isna(available_at(record(Grade.CORROBORATED),CAL))

def test_snapshot_available_from_capture_time_only():
    assert available_at(record(Grade.CAPTURED),CAL)==CAL[3]

def test_restatement_does_not_overwrite_first_vintage():
    old=record(); new=replace(old,value=4,published_at=CAL[3],vintage_id='v2')
    assert append_vintage((old,),new)==(old,new)
    with pytest.raises(ValueError): append_vintage((old,),old)

def test_admission_masks_non_primary_cells():
    p=pd.DataFrame([[record(g) for g in Grade]],index=[CAL[4]])
    values,_=admission_view(p,CAL)
    assert values.iloc[0,:2].notna().all() and values.iloc[0,2:].isna().all()

def test_deadline_is_not_publication_date():
    assert pd.isna(available_at(replace(record(),published_at=None),CAL))

def test_exploration_run_is_labelled_and_charged(tmp_path):
    from quant_lab.research.trial_ledger import TrialLedger
    ledger=TrialLedger(tmp_path/'l.db',tmp_path/'e')
    reservation=dict(idempotency_key='explore',campaign='lab',family='explore',hypothesis_id='h',candidate_id='c',**{k:'a'*64 for k in ('code_hash','data_hash','config_hash','prompt_hash','model_hash')})
    p=pd.DataFrame([[record(Grade.CORROBORATED)]],index=[CAL[4]])
    view=exploration_view(p,ledger=ledger,reservation=reservation)
    assert view.attrs['exploration_only'] and ledger.snapshot().charged_count==1
    with pytest.raises(ValueError): admission_view(view,CAL)
