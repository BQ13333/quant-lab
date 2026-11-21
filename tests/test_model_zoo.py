import numpy as np
import pandas as pd
import pytest

from quant_lab.models.zoo import (
    BreakoutModel,
    CountryRelativeMomentumModel,
    CrossCountryRankingModel,
    LowVolatilityModel,
    LowVolatilityModel126,
    LowVolatilityModel252,
    MeanReversionModel,
    MeanReversionModel21,
    MeanReversionModel63,
    MomentumModel,
    MomentumModel126,
    MomentumModel21,
    MomentumModel252,
    RegimeConditionedModel,
    TimeSeriesMomentumModel,
    TrendLowVolEnsembleModel,
    VolatilityAdjustedMomentumModel,
    VolatilityAdjustedMomentumModel126,
    VolatilityAdjustedMomentumModelVol21,
    VolatilityCompressionModel,
    VolumeConfirmedMomentumModel,
    WithinCountryRankingModel,
)
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


def _make_volume(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        np.random.randint(1000, 10000, size=prices.shape),
        index=prices.index,
        columns=prices.columns,
    )


def _make_groups(prices: pd.DataFrame) -> pd.Series:
    countries = ["SG", "TH", "MY", "ID", "PH"]
    return pd.Series(countries[: len(prices.columns)], index=prices.columns)


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        (LowVolatilityModel, {"name": "test"}),
        (LowVolatilityModel126, {"name": "test"}),
        (LowVolatilityModel252, {"name": "test"}),
        (MomentumModel, {"name": "test"}),
        (MomentumModel21, {"name": "test"}),
        (MomentumModel126, {"name": "test"}),
        (MomentumModel252, {"name": "test"}),
        (VolatilityAdjustedMomentumModel, {"name": "test"}),
        (VolatilityAdjustedMomentumModel126, {"name": "test"}),
        (VolatilityAdjustedMomentumModelVol21, {"name": "test"}),
        (MeanReversionModel, {"name": "test"}),
        (MeanReversionModel21, {"name": "test"}),
        (MeanReversionModel63, {"name": "test"}),
        (TimeSeriesMomentumModel, {"name": "test"}),
        (TimeSeriesMomentumModel, {"name": "test", "scale_by_vol": True}),
        (TrendLowVolEnsembleModel, {"name": "test"}),
        (BreakoutModel, {"name": "test"}),
        (VolatilityCompressionModel, {"name": "test"}),
    ],
)
def test_model_predict_shape_and_metadata(model_cls, kwargs):
    prices = _make_prices()
    model = model_cls(**kwargs)
    start, end = 250, 300
    sf = model.predict(prices, start=start, end=end)
    assert sf.scores.shape == (end - start, prices.shape[1])
    assert sf.model_name == "test"
    assert sf.horizon == "1d"
    assert not sf.scores.empty
    assert set(sf.scores.columns) == set(prices.columns)
    assert sf.scores.index.isin(prices.index).all()


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        (VolumeConfirmedMomentumModel, {"name": "test"}),
    ],
)
def test_volume_model_predict(model_cls, kwargs):
    prices = _make_prices()
    volume = _make_volume(prices)
    model = model_cls(volume=volume, **kwargs)
    start, end = 250, 300
    sf = model.predict(prices, start=start, end=end)
    assert sf.scores.shape == (end - start, prices.shape[1])
    assert sf.model_name == "test"
    assert not sf.scores.empty


@pytest.mark.parametrize(
    "model_cls, kwargs",
    [
        (CountryRelativeMomentumModel, {"name": "test"}),
        (WithinCountryRankingModel, {"name": "test"}),
        (CrossCountryRankingModel, {"name": "test"}),
    ],
)
def test_country_model_predict(model_cls, kwargs):
    prices = _make_prices()
    groups = _make_groups(prices)
    model = model_cls(groups=groups, **kwargs)
    start, end = 250, 300
    sf = model.predict(prices, start=start, end=end)
    assert sf.scores.shape == (end - start, prices.shape[1])
    assert sf.model_name == "test"
    assert not sf.scores.empty
    assert set(sf.scores.columns) == set(prices.columns)


def test_regime_conditioned_model():
    prices = _make_prices()
    inner = MomentumModel(name="inner")
    detector = VolTrendRegimeDetector()
    model = RegimeConditionedModel(
        name="regime_test", model=inner, regime="trend_up", detector=detector
    )
    start, end = 250, 300
    sf = model.predict(prices, start=start, end=end)
    assert sf.scores.shape == (end - start, prices.shape[1])
    assert sf.model_name == "regime_test"
    assert not sf.scores.empty
    regimes = detector.describe(prices).iloc[start:end]
    mask = regimes.apply(lambda r: "trend_up" in r if pd.notna(r) else False)
    if not mask.all():
        assert (sf.scores.loc[~mask] == 0.0).all().all()


def test_all_models_deterministic():
    prices = _make_prices()
    models = [
        LowVolatilityModel(name="lv"),
        MomentumModel(name="mom"),
        MeanReversionModel(name="mr"),
        TimeSeriesMomentumModel(name="tsm"),
        BreakoutModel(name="bo"),
        VolatilityCompressionModel(name="vc"),
    ]
    for model in models:
        sf1 = model.predict(prices, start=250, end=300)
        sf2 = model.predict(prices, start=250, end=300)
        pd.testing.assert_frame_equal(sf1.scores, sf2.scores)


def test_trend_low_vol_ensemble_default_models():
    prices = _make_prices()
    model = TrendLowVolEnsembleModel(name="ensemble")
    sf = model.predict(prices, start=250, end=300)
    assert sf.scores.shape == (50, prices.shape[1])
    assert sf.metadata["family"] == "trend_low_vol_ensemble"
