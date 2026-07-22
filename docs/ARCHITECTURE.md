# Quant Lab Architecture

## Mission

Build a model tournament and routing system that behaves like a serious quant research desk:

- Many independent alpha models are allowed to compete.
- Validation is time-aware and hostile to overfitting.
- Portfolio construction accounts for volatility, correlation, turnover, cost, and capacity.
- A regime router decides when a model is eligible, reduced, or shut off.
- Every decision is reproducible from data snapshot, code version, config, and metrics.

## Layers

### 1. Data Layer

Responsibilities:

- Load market, fundamental, macro, alternative, and execution data.
- Enforce point-in-time availability.
- Normalize calendars and corporate actions.
- Store raw, cleaned, feature, and prediction artifacts separately.

Core contract:

- `MarketPanel`: aligned prices, returns, volume, and metadata.

Future adapters:

- Local parquet/duckdb.
- Vendor APIs.
- Qlib dataset format.
- Broker or execution logs for TCA.

Current free-data adapter:

- Yahoo Finance via `yfinance`, used for first-pass ASEAN index, stock, FX, and intraday-sample extraction.
- Free Yahoo coverage is not a replacement for point-in-time constituent, corporate-action, and exchange-quality data.

### 2. Feature Layer

Responsibilities:

- Compute cross-sectional and time-series features.
- Keep feature timestamps aligned to when data was knowable.
- Track feature lineage and lookback windows.

Initial feature families:

- Momentum and reversal.
- Realized volatility.
- Liquidity and turnover.
- Cross-sectional rank signals.
- Regime descriptors.

### 3. Model Zoo

Responsibilities:

- Provide one interface for all alpha models.
- Keep simple baselines beside complex models.
- Output scores, not direct trades.
- Build repeatable model suites through a signal factory so agent loops can
  track exactly which hypotheses were tested.

Initial candidates:

- Momentum.
- Mean reversion.
- Volatility-adjusted momentum.
- Linear or ridge model.
- Gradient boosted trees.

Later candidates:

- Qlib model adapters.
- Meta-labeling models.
- Deep sequence models.
- Statistical arbitrage models.

Current signal factory profiles:

- `lite`: fast smoke-test suite for global ETF proxy research.
- `slow`: lower-turnover momentum, trend, low-volatility, and ensemble suite.
- `full`: broader exploratory suite that requires stricter multiple-testing
  penalties.

### 4. Validation Layer

Responsibilities:

- Walk-forward testing.
- Purge and embargo around labels.
- Score every model after transaction costs.
- Penalize multiple testing and non-normal returns.
- Detect model decay and regime dependence.

Key metrics:

- Sharpe, Sortino, Calmar.
- Max drawdown.
- Hit rate.
- Turnover.
- Cost drag.
- Probabilistic Sharpe.
- Deflated Sharpe.

### 5. Regime Router

Responsibilities:

- Describe current market state.
- Estimate which models are eligible in that state.
- Reduce risk when the environment is hostile.
- Prevent stale models from trading just because they once worked.

Initial regimes:

- Trend up, trend down, choppy.
- Low, medium, high volatility.
- Normal versus stressed correlation.

### 6. Portfolio Layer

Responsibilities:

- Convert model scores to target weights.
- Apply volatility targeting.
- Enforce exposure, concentration, leverage, liquidity, and turnover constraints.
- Combine models through allocation rules or meta-models.

### 7. Execution And TCA Layer

Responsibilities:

- Estimate slippage and market impact before trading.
- Compare expected versus realized implementation shortfall.
- Feed cost observations back into model selection.

This is intentionally second-phase unless intraday/tick data is available.

### 8. Experiment Ledger

Responsibilities:

- Log configs, code version, data version, parameters, metrics, plots, and model artifacts.
- Make every leaderboard entry reproducible.
- Persist research-loop manifests, promotion decisions, signal manifests, paper
  order intents, and paper fills.

Recommended adapter:

- MLflow for experiment tracking and model registry.

Current local ledgers:

- `loop_manifest.json` records one research-loop run.
- `promotion_decisions.csv` records gate pass/fail reasons.
- `signal_manifest.csv` records tested model candidates.
- `orders.csv` and `fills.csv` record paper-trading activity.

## First-Phase Build

The first phase should stay deliberately boring:

- Establish clean interfaces.
- Prove the tournament loop works on synthetic or simple market data.
- Make validation hard enough that weak strategies fail quickly.
- Add heavy libraries only through adapters.

The platform should become impressive because it rejects bad models, not because it imports fashionable ones.
