import numpy as np
import pandas as pd
import pytest

from quant_lab.models.ensembles import (
    EqualWeightEnsembleModel,
    RegimeWeightedEnsembleModel,
    WeightedEnsembleModel,
)
from quant_lab.models.zoo import LowVolatilityModel, MomentumModel
from quant_lab.regimes import VolTrendRegimeDetector


def _make_prices(n_rows: int = 300, n_cols: int = 5) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    prices = pd.DataFrame(
        np.cumsum(np.random.randn(n_rows, n_cols) * 0.01, axis=0) + 100,
        index=dates,
        columns=[f"ASSET_{i}" for i in range(n_cols)],
    )
    return prices


def test_equal_weight_ensemble_averaging():
    prices = _make_prices()
    models = [
        MomentumModel(name="momentum"),
        LowVolatilityModel(name="lowvol"),
    ]
    ensemble = EqualWeightEnsembleModel(name="ew", models=models)
    sf = ensemble.predict(prices, start=250, end=300)
    assert sf.scores.shape == (50, prices.shape[1])
    assert sf.model_name == "ew"
    assert sf.metadata["family"] == "equal_weight_ensemble"
    assert not sf.scores.empty

    sf1 = models[0].predict(prices, start=250, end=300)
    sf2 = models[1].predict(prices, start=250, end=300)
    expected = (sf1.scores + sf2.scores) / 2.0
    pd.testing.assert_frame_equal(sf.scores, expected)


def test_equal_weight_ensemble_empty():
    prices = _make_prices()
    ensemble = EqualWeightEnsembleModel(name="empty")
    sf = ensemble.predict(prices, start=250, end=300)
    assert sf.scores.shape == (50, prices.shape[1])
    assert (sf.scores == 0.0).all().all()


def test_weighted_ensemble():
    prices = _make_prices()
    models = [
        MomentumModel(name="momentum"),
        LowVolatilityModel(name="lowvol"),
    ]
    weights = [0.7, 0.3]
    ensemble = WeightedEnsembleModel(name="ww", models=models, weights=weights)
    sf = ensemble.predict(prices, start=250, end=300)
    assert sf.scores.shape == (50, prices.shape[1])
    assert sf.model_name == "ww"
    assert sf.metadata["weights"] == weights

    sf1 = models[0].predict(prices, start=250, end=300)
    sf2 = models[1].predict(prices, start=250, end=300)
    expected = sf1.scores * 0.7 + sf2.scores * 0.3
    pd.testing.assert_frame_equal(sf.scores, expected)


def test_weighted_ensemble_mismatched_length():
    prices = _make_prices()
    models = [MomentumModel(name="momentum")]
    weights = [0.5, 0.5]
    ensemble = WeightedEnsembleModel(name="bad", models=models, weights=weights)
    with pytest.raises(ValueError):
        ensemble.predict(prices, start=250, end=300)


def test_regime_weighted_ensemble():
    prices = _make_prices()
    models = {
        "trend_up": MomentumModel(name="momentum"),
        "trend_down": LowVolatilityModel(name="lowvol"),
    }
    detector = VolTrendRegimeDetector()
    ensemble = RegimeWeightedEnsembleModel(
        name="regime_ew", models=models, detector=detector
    )
    sf = ensemble.predict(prices, start=250, end=300)
    assert sf.scores.shape == (50, prices.shape[1])
    assert sf.model_name == "regime_ew"
    assert sf.metadata["family"] == "regime_weighted_ensemble"
    assert not sf.scores.empty

    regimes = detector.describe(prices).iloc[250:300]
    sf_mom = models["trend_up"].predict(prices, start=250, end=300)
    sf_low = models["trend_down"].predict(prices, start=250, end=300)
    for idx in regimes.index:
        regime = regimes.loc[idx]
        if pd.isna(regime):
            continue
        if "trend_up" in regime:
            assert (sf.scores.loc[idx] == sf_mom.scores.loc[idx]).all()
        elif "trend_down" in regime:
            assert (sf.scores.loc[idx] == sf_low.scores.loc[idx]).all()
