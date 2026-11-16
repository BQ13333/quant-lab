import numpy as np
import pytest
from quant_lab.lifecycle.decay import *


def test_posterior_and_ar_inflation():
    p = normal_posterior([.1]*10, .02, .03, .1, .01, rho=.5)
    assert p.n_eff == pytest.approx(10/3)
    assert p.p_positive > p.p_net > .5
    assert normal_posterior([.1]*10, .02, .03, .1, rho=-.5).n_eff == 10
    assert normal_posterior([], .02, .03, .1).mean == pytest.approx(.02)


def test_cusum_gap():
    assert cusum(2., None, .02, .1) == (2., False)
    assert cusum(2., -.1, .02, .1)[0] == pytest.approx(3.05)


def test_bocpd_shift():
    rng = np.random.default_rng(12)
    xs = np.r_[rng.normal(0, .2, 80), rng.normal(-1.5, .2, 20)]
    assert bocpd(xs[:80], sigma=.2, window=8).probability_recent < .3
    assert max(bocpd(xs[:t], sigma=.2, window=8).probability_recent for t in range(81, 87)) > .9
    result = bocpd(xs, sigma=.2)
    assert sum(result.run_length) == pytest.approx(1.)
    assert result.run_length[0] == pytest.approx(1/60)


def test_half_life_and_slope():
    fit = exponential_half_life([1, 3.5, 8, 15.5, 30.5, 50.5], [2.224, 1.853, 1.657, 1.068, .589, .240])
    assert fit.half_life == pytest.approx(15.6, abs=.15)
    slope, se = rolling_decay_slope(np.arange(20)*-.01)
    assert slope == pytest.approx(-.01)
    assert se < 1e-8


@pytest.mark.parametrize('call', [lambda: normal_posterior([np.nan], .02, .01, .1),
    lambda: cusum(0, np.inf, 0, 1), lambda: bocpd([np.nan]),
    lambda: exponential_half_life([1,2,3], [1,np.inf,2]), lambda: rolling_decay_slope([np.nan]*12)])
def test_nonfinite(call):
    with pytest.raises(ValueError):
        call()
