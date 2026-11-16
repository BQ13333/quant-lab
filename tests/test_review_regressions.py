import numpy as np
import pandas as pd
import pytest
from quant_lab.validation.metrics import probabilistic_sharpe_ratio, deflated_sharpe_ratio
from quant_lab.validation.firewall import deflated_sharpe_with_lifetime
from quant_lab.portfolio.costs import capacity_score, country_cost_schedule, linear_transaction_cost
from quant_lab.portfolio import optimizers
from quant_lab.models.ensembles import WeightedEnsembleModel
from quant_lab.contracts import SignalFrame
from quant_lab.regimes import VolTrendRegimeDetector
from quant_lab.research.trial_ledger import TrialLedger


def test_f01_more_charged_trials_never_improve_dsr():
    r = np.random.default_rng(42).normal(.001, .01, 504)
    values = [deflated_sharpe_with_lifetime(r, observed_trial_sharpes=[-1, 1], lifetime_attempts=n) for n in (2, 10, 10000)]
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_f01_nonfinite_trial_fails(bad):
    with pytest.raises(ValueError):
        deflated_sharpe_ratio([-.01, .01, .02, .03], [-1, 1, bad])


@pytest.mark.parametrize('r', [[-.01,.01,.02], [.01]*4, [-.01,.01,.02,np.nan], [-.01,.01,.02,np.inf]])
def test_f04_invalid_moments_raise(r):
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(r)


@pytest.mark.parametrize('field', ['candidate_id','campaign','family','hypothesis_id','parent_trial_id','code_hash','data_hash','config_hash','prompt_hash','model_hash'])
def test_f03_retry_identity_drift(tmp_path, field):
    ledger = TrialLedger(tmp_path/'ledger.sqlite', tmp_path/'evidence')
    args = dict(idempotency_key='k', campaign='my', family='f', hypothesis_id='h', candidate_id='c', parent_trial_id=None)
    args.update({k: 'a'*64 for k in ['code_hash','data_hash','config_hash','prompt_hash','model_hash']})
    original = ledger.reserve_trial(**args)
    assert ledger.reserve_trial(**args).trial_id == original.trial_id
    args[field] = 'b'*64 if field.endswith('hash') else 'different'
    with pytest.raises(RuntimeError, match='drift'):
        ledger.reserve_trial(**args)


@pytest.mark.parametrize('weights', [[], [np.nan], [np.inf]])
def test_f12_invalid_weights(weights):
    model = WeightedEnsembleModel(name='bad', models=[object()], weights=weights)
    with pytest.raises(ValueError):
        model.predict(pd.DataFrame({'a':[1.,2.]}), 0, 2)


def test_f12_nonfinite_score():
    p = pd.DataFrame({'a':[1.]})
    with pytest.raises(ValueError):
        SignalFrame(pd.DataFrame({'a':[np.nan]}), 'm', 'daily').validate_against(p)


@pytest.mark.parametrize('adv', [pd.Series(dtype=float), pd.Series({'a':0.}), pd.Series({'a':-1.}), pd.Series({'a':np.nan})])
def test_f14_invalid_liquidity(adv):
    with pytest.raises(ValueError):
        capacity_score(pd.DataFrame({'a':[0.,1.]}), adv, 100.)


@pytest.mark.parametrize('country', ['China','Hong Kong','Unknown'])
def test_f14_unsupported_market(country):
    with pytest.raises(ValueError):
        country_cost_schedule(country)


def test_f15_opening_cost_and_explicit_holdings():
    w = pd.DataFrame({'a':[1.,1.]})
    assert linear_transaction_cost(w, 10).tolist() == [.001,0.]
    assert linear_transaction_cost(w, 10, starting_holdings=pd.Series({'a':1.})).tolist() == [0.,0.]


def test_f11_rescaling_invariant():
    rng = np.random.default_rng(23)
    prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(0,.02,(160,3)),axis=0)), index=pd.date_range('2020-01-01',periods=160))
    detector = VolTrendRegimeDetector(10,10,10)
    scaled = prices.copy(); scaled[0] *= 100
    pd.testing.assert_series_equal(detector.describe(prices), detector.describe(scaled))


def test_f11_warmup_unknown():
    from quant_lab.regimes import RegimeValue
    p = pd.DataFrame(np.exp(np.cumsum(np.random.default_rng(4).normal(0,.01,(15,2)),axis=0)))
    assert RegimeValue.UNKNOWN.value == VolTrendRegimeDetector(3,10,10).describe(p).iloc[-1]


def test_f16_infeasible_cap_turnover_raises():
    s = pd.DataFrame([[1.]*6]); prev = pd.DataFrame([[.4,.12,.12,.12,.12,.12]])
    with pytest.raises(ValueError, match='infeasible'):
        optimizers.turnover_constrained_optimizer(s, prev, .05, use_cvxpy=False)


def test_f16_annual_covariance(monkeypatch):
    r = pd.DataFrame(np.random.default_rng(8).normal(.001,.01,(10,6)))
    seen = []
    def capture(mu,cov):
        seen.append((mu,cov)); return np.ones(6)/6
    monkeypatch.setattr(optimizers, '_max_sharpe_cvxpy', capture)
    optimizers.max_sharpe_shrinkage(pd.DataFrame(1.,index=r.index,columns=r.columns),r,lookback=10,shrinkage=0.)
    mu,cov = seen[-1]
    np.testing.assert_allclose(mu, r.iloc[:9].mean().values*252)
    np.testing.assert_allclose(cov, r.iloc[:9].cov().values*252)
