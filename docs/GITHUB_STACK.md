# GitHub And Open-Source Stack

## Recommendation

Use GitHub for vetted open-source libraries and reproducible project templates. Do not install random "trading agents" or opaque auto-traders from GitHub.

The correct pattern is:

1. Keep this repository as the control plane.
2. Add open-source packages through pinned dependencies.
3. Wrap each external tool behind an adapter.
4. Never let an external framework own the research process end to end.

## Phase 1 Installs

Core scientific stack:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

Research operations:

- `optuna` for hyperparameter search.
- `mlflow` for experiment tracking.
- `duckdb` and `pyarrow` for local research data storage.

Modeling:

- `lightgbm` or `catboost` for tabular alpha models.
- `statsmodels` for time-series diagnostics.

Portfolio and risk:

- `skfolio` for scikit-learn-style portfolio optimization and cross-validation.
- `PyPortfolioOpt` for quick mean-variance, Black-Litterman, shrinkage, and HRP prototypes.
- `riskfolio-lib` for broader risk measures and constraints.

Backtesting:

- `vectorbt` for fast parameter sweeps and vectorized signal tests.

## Phase 2 Evaluation

Potentially useful, but not first:

- `microsoft/qlib`: strong AI quant research platform, useful as an adapter or benchmark.
- `backtrader`: event-driven tests, useful when order mechanics matter.
- `zipline-reloaded`: useful if you want a Zipline-style workflow.
- `mlfinlab`: conceptually aligned with financial ML, but check licensing and maintenance before relying on it.

## Avoid At First

- Unverified GitHub trading bots.
- Repositories promising live profitability.
- Any dependency that requires secrets before it can be evaluated.
- Any agent that can place trades or send orders without a separate risk gate.

## Security Rules

- Pin versions in a lockfile.
- Read licenses before commercial use.
- Prefer PyPI or conda releases over installing from arbitrary branches.
- Run unknown code in a throwaway virtual environment.
- Keep broker credentials out of source control.
- Require tests before integrating an external adapter.
