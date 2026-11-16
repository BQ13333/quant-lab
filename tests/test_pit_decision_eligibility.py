import pandas as pd
from quant_lab.data.pit_leakage_qa import decision_eligibility

def test_future_rows_do_not_change_past_eligibility():
    p=pd.DataFrame({'a':[1,2,None,4]},index=pd.date_range('2020-01-01',periods=4))
    future=pd.DataFrame({'a':[None]*10},index=pd.date_range('2020-01-05',periods=10))
    pd.testing.assert_frame_equal(decision_eligibility(p,2,.5),decision_eligibility(pd.concat([p,future]),2,.5).loc[p.index])
