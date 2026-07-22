from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.optimization.search_spaces import (
    AllocatorSearchSpace,
    CostSearchSpace,
    ModelSearchSpace,
    PortfolioSearchSpace,
    RegimeSearchSpace,
)
from quant_lab.portfolio.costs import linear_transaction_cost
from quant_lab.portfolio.sizing import volatility_target
from quant_lab.validation.metrics import sharpe_ratio
from quant_lab.validation.holdout import HoldoutManager

try:
    import optuna

    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_from_space(trial: Any, space: dict[str, tuple[Any, ...]]) -> dict[str, Any]:
    """Sample a categorical value for each key in *space* using *trial*."""
    params: dict[str, Any] = {}
    for key, values in space.items():
        params[key] = trial.suggest_categorical(key, values)
    return params

def _build_model(model_params: dict[str, Any]) -> Any:
    """Instantiate a model from sampled hyperparameters."""
    from quant_lab.models.zoo import (
        MomentumModel,
        MeanReversionModel,
    )

    lookback = model_params.get("lookback", 63)
    # Choose a simple model family based on lookback for diversity
    if lookback <= 42:
        return MeanReversionModel(name=f"mean_reversion_{lookback}", lookback=lookback)
    return MomentumModel(name=f"momentum_{lookback}", lookback=lookback)


def _build_weights(
    allocator_params: dict[str, Any],
    portfolio_params: dict[str, Any],
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    group_map: pd.Series | None,
) -> pd.DataFrame:
    """Convert model scores to portfolio weights using sampled allocator params."""
    from quant_lab.portfolio import optimizers
    from quant_lab.portfolio.sizing import scores_to_group_neutral_long_only_weights, scores_to_long_only_weights

    weighting = allocator_params.get("weighting", "dollar_neutral")
    top_fraction = allocator_params.get("top_fraction", 0.33)
    freq = allocator_params.get("rebalance_frequency", "D")
    target_vol = portfolio_params.get("target_volatility", 0.12)

    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)

    if weighting == "dollar_neutral":
        weights = optimizers.dollar_neutral(scores)
    elif weighting == "long_only":
        weights = scores_to_long_only_weights(scores, top_fraction=top_fraction)
    elif weighting == "group_neutral_dollar_neutral":
        if group_map is None:
            raise ValueError("group_map required for group_neutral_dollar_neutral")
        weights = optimizers.country_neutral_dollar_neutral(scores, group_map)
    elif weighting == "group_neutral_long_only":
        if group_map is None:
            raise ValueError("group_map required for group_neutral_long_only")
        weights = scores_to_group_neutral_long_only_weights(scores, group_map, top_fraction=top_fraction)
    else:
        raise ValueError(f"unknown weighting: {weighting}")

    weights = volatility_target(
        weights=weights,
        asset_returns=asset_returns.reindex(weights.index),
        target_volatility=target_vol,
        periods_per_year=252,
    )

    if freq != "D":
        weights = optimizers.rebalance_schedule(weights, frequency=freq)

    return weights


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def objective(
    trial: Any,
    prices: pd.DataFrame,
    group_map: pd.Series | None,
    benchmark_returns: pd.Series | None,
    train_idx: pd.Index,
    test_idx: pd.Index,
) -> float:
    """Optuna objective for a single train/test split.

    Samples hyperparameters, builds a model + allocator on *train_idx*,
    evaluates net information ratio on *test_idx*, and never touches the
    final holdout.
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna is required. Install it with: pip install 'quant-lab[research]'")

    # Sample hyperparameters
    model_params = _sample_from_space(trial, ModelSearchSpace().to_dict())
    allocator_params = _sample_from_space(trial, AllocatorSearchSpace().to_dict())
    portfolio_params = _sample_from_space(trial, PortfolioSearchSpace().to_dict())
    cost_params = _sample_from_space(trial, CostSearchSpace().to_dict())
    regime_space = RegimeSearchSpace().to_dict()
    regime_space["vol_lookback_regime"] = regime_space.pop("vol_lookback")
    regime_params = _sample_from_space(trial, regime_space)
    if group_map is None and str(allocator_params.get("weighting", "")).startswith(
        "group_neutral_"
    ):
        return -1.0e12

    # Fit model on train, predict on test
    model = _build_model(model_params)
    train_start = prices.index.get_loc(train_idx[0])
    train_end = prices.index.get_loc(train_idx[-1]) + 1
    test_start = prices.index.get_loc(test_idx[0])
    test_end = prices.index.get_loc(test_idx[-1]) + 1

    model.fit(prices, train_start, train_end)
    signals = model.predict(prices, test_start, test_end)
    signals.validate_against(prices)

    # Build allocator weights
    weights = _build_weights(
        allocator_params=allocator_params,
        portfolio_params=portfolio_params,
        scores=signals.scores,
        prices=prices,
        group_map=group_map,
    )

    # Net returns on test
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    gross_returns = (weights.shift(1).fillna(0.0) * asset_returns.reindex(weights.index)).sum(axis=1)
    cost_bps = cost_params.get("cost_bps", 5)
    costs = linear_transaction_cost(weights, cost_bps)
    net_returns = gross_returns - costs

    # Align to test period only
    net_returns = net_returns.reindex(test_idx).dropna()

    if net_returns.empty:
        return 0.0

    # Net information ratio vs benchmark, or Sharpe if no benchmark
    if benchmark_returns is not None:
        aligned = pd.concat(
            [net_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
            axis=1,
            join="inner",
        ).dropna()
        if aligned.empty:
            return 0.0
        active = aligned["strategy"] - aligned["benchmark"]
        tracking_error = active.std(ddof=1) * np.sqrt(252)
        if tracking_error == 0 or pd.isna(tracking_error):
            return 0.0
        ir = (active.mean() * 252) / tracking_error
        return float(ir)

    return float(sharpe_ratio(net_returns, periods_per_year=252))


# ---------------------------------------------------------------------------
# Study creation
# ---------------------------------------------------------------------------


def create_optuna_study(
    prices: pd.DataFrame,
    group_map: pd.Series | None,
    benchmark_returns: pd.Series | None,
    holdout_manager: HoldoutManager,
    n_trials: int = 100,
) -> Any:
    """Create an Optuna study and optimize on the non-holdout data.

    The study is configured to maximize the net information ratio. The inner
    train/test split is drawn from *holdout_manager.train_index* so that the
    holdout set is never used for parameter selection.
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna is required. Install it with: pip install 'quant-lab[research]'")

    import optuna

    train_index = holdout_manager.train_index
    # Deterministic 70/30 split of the training data for inner CV
    split_point = int(len(train_index) * 0.7)
    inner_train = train_index[:split_point]
    inner_test = train_index[split_point:]

    def _objective(trial: Any) -> float:
        return objective(
            trial,
            prices,
            group_map,
            benchmark_returns,
            inner_train,
            inner_test,
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(_objective, n_trials=n_trials)
    return study


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_optuna_search(
    prices: pd.DataFrame,
    group_map: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    holdout_manager: HoldoutManager | None = None,
    n_trials: int = 100,
) -> tuple[dict[str, Any], float, Any]:
    """Run a full Optuna search with holdout isolation.

    Returns
    -------
    best_params : dict
        Best hyperparameters found on the inner train/test split.
    best_value : float
        Best objective value (net information ratio).
    study : optuna.study.Study
        The completed Optuna study object.
    """
    if not _HAS_OPTUNA:
        raise ImportError("optuna is required. Install it with: pip install 'quant-lab[research]'")

    if holdout_manager is None:
        holdout_manager = HoldoutManager(prices.index, holdout_fraction=0.2)

    # Verify holdout is isolated: objective only uses train_index internally
    holdout_dates = set(holdout_manager.holdout_index)
    train_dates = set(holdout_manager.train_index)
    if holdout_dates & train_dates:
        raise RuntimeError("Holdout and train indices overlap")

    study = create_optuna_study(
        prices=prices,
        group_map=group_map,
        benchmark_returns=benchmark_returns,
        holdout_manager=holdout_manager,
        n_trials=n_trials,
    )

    return study.best_params, study.best_value, study
