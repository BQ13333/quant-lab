# Data And Math Agent Factory

## Objective

Build a permanent multi-agent quant research shop around a deterministic Python
engine. Agents coordinate data discovery, coverage audits, feature proposals,
math upgrades, validation review, and reporting. They do not bypass the
backtester, promotion gates, or human approval.

## Operating Principle

The platform must be able to answer three questions for every strategy:

- What exact data did it know at the time?
- What exact math selected it?
- What exact constraints would have stopped it from trading?

If any answer is missing, the strategy stays in research.

## Agent Lanes

### 1. Security Master Agents

Scope:

- Exchange listings.
- Delistings.
- Ticker changes.
- ISIN/local code mappings.
- Board/market segments.
- Suspensions and listing status.

Required output:

- `listings.csv`
- `corporate_actions.csv`
- `delistings.csv`
- `symbol_map.csv`
- `manifest.csv`

Definition of done:

- Every row has source, fetch timestamp, exchange, local symbol, normalized
  symbol, security type, currency, and status.
- Current-listed and historical-listed universes are kept separate.
- Survivorship limitations are explicit in the manifest.

### 2. Market Data Agents

Scope:

- Daily OHLCV.
- Adjusted prices.
- Corporate-action-adjusted returns.
- FX rates.
- Index and ETF benchmarks.
- Intraday bars when execution research requires them.

Required output:

- Raw exchange/vendor files under `artifacts/data_lake/raw/`.
- Partitioned Parquet under `artifacts/data_lake/prices/`.
- Coverage reports with first date, last date, missing ratio, stale rows, and
  adjustment flags.

Definition of done:

- No silent ticker drops.
- Price source and adjustment method are recorded.
- Failed downloads appear in a failure manifest.
- Batches are bounded so a laptop run cannot exhaust memory.

### 3. Data QA Agents

Scope:

- Missing data.
- Calendar alignment.
- Split/dividend sanity checks.
- Stale prices.
- Outlier returns.
- Duplicate symbols.
- Look-ahead and point-in-time violations.

Required output:

- `coverage.csv`
- `quality_flags.csv`
- `qa_summary.md`
- pass/fail status for backtest eligibility.

Definition of done:

- A research run cannot consume new data without a QA manifest.
- QA failures are visible in Obsidian and CSV artifacts.

### 4. Feature Agents

Scope:

- Price/volume features.
- Fundamental features.
- Earnings/event features.
- Macro and FX features.
- Microstructure features.

Required output:

- Feature spec with timestamp availability rules.
- Feature lineage manifest.
- Unit tests for leakage and shape stability.

Definition of done:

- Every feature declares its lookback, release lag, and known-at timestamp.
- No feature can access future returns, future constituents, or revised data
  without an explicit lag.

### 5. Math Engine Agents

Scope:

- Model families.
- Hyperparameter search.
- Ensemble construction.
- Portfolio optimization.
- Overfitting controls.
- Regime routing.
- Risk/capacity constraints.

Required output:

- Candidate model code.
- Search-space manifest.
- Validation report.
- Killed-strategy report.
- Champion model card.

Definition of done:

- Every new model has a baseline comparison.
- Every search has a multiple-testing correction.
- Every promoted model survives transaction costs, drawdown, turnover, capacity,
  and final holdout checks.

### 6. Execution And TCA Agents

Scope:

- Paper order generation.
- IBKR integration readiness.
- Slippage and impact model.
- Trade cost analysis.
- Broker fill ledger.
- Kill switches.

Required output:

- Paper orders.
- Paper fills.
- TCA report.
- Execution safety checklist.

Definition of done:

- No live order path is enabled by default.
- Human approval is required before any broker-routed trade.
- Paper trading must reconcile expected versus realized fills.

## Regional Data Agent Mandates

### ASEAN Agent

Markets:

- Malaysia: Bursa Malaysia.
- Singapore: SGX.
- Indonesia: IDX.
- Thailand: parked outside the current three-country universe.
- Philippines: removed because we cannot trade it.
- Vietnam: removed from the active project universe.

Immediate work:

- Upgrade current listing adapters.
- Attach reliable price vendors where Yahoo coverage is weak.
- Add Bursa/IDX source adapters or vendor mapping files.
- Build country-level coverage dashboards.

### North Asia Agent

Markets:

- Hong Kong: HKEX.
- Japan: JPX.
- Korea: KRX/KOSPI/KOSDAQ/KONEX.

Excluded from the current quant-project mandate:

- China: SSE, SZSE, BSE.
- Taiwan: TWSE, TPEx.

Existing adapters and artifacts are retained only for reproducibility and must not enter active extraction, backtesting, or optimization.

Immediate work:

- Keep official listing adapters current.
- Improve price coverage and corporate-action handling.
- Separate A-share, H-share, ADR, ETF, and REIT universes.

### US And Global Agent

Markets:

- US common stocks and ADRs.
- Global ETFs.
- Regional benchmark ETFs.
- Index proxies.

Immediate work:

- Add delisted and corporate-action-aware vendor path.
- Maintain a current free listing snapshot for discovery.
- Build benchmark and hedge proxy panels.

## Python Core

Python remains the research engine because it is best for:

- Data ingestion.
- Backtesting.
- ML research.
- Optimization.
- Agent orchestration.
- IBKR integration.

R can be used later for specialist statistical research. C++ is only justified
for low-latency execution or large-scale numerical kernels after the strategy is
already proven.

## Elite Maths Engine Roadmap

### Current Core

- Walk-forward validation.
- Purged and embargoed splits.
- Deflated Sharpe.
- Cost stress.
- Turnover and drawdown gates.
- Volatility targeting.
- Net IR constrained optimization.
- Regime-conditioned models.
- Optuna search scaffolding.
- Paper execution ledger.

### Next Maths Upgrades

1. Nested model selection.
   - Inner loop selects parameters.
   - Outer loop estimates true out-of-sample performance.
   - Final untouched holdout remains sealed until champion selection.

2. Probability of Backtest Overfitting.
   - Use CPCV paths to estimate how often a strategy wins by luck.
   - Store PBO beside deflated Sharpe.

3. Feature-neutral and sector-neutral attribution.
   - Explain whether returns come from intended alpha or hidden exposures.

4. Robust portfolio construction.
   - Hierarchical risk parity.
   - Shrinkage covariance.
   - CVaR/min-tail-risk optimization.
   - Turnover-aware optimization.

5. Transaction cost and capacity model.
   - ADV participation.
   - Spread/slippage curves.
   - Market-impact stress by exchange.
   - Capacity decay at different AUM levels.

6. ML meta-layer.
   - Meta-labeling for when to trade a signal.
   - Regime gating.
   - Model decay classifier.
   - Ensemble weight learner with strict holdout control.

7. Live research monitor.
   - Paper trading only.
   - Model drift alerts.
   - Data freshness checks.
   - Broker reconciliation.

## Memory And Storage Discipline

- Mine by market and by batch.
- Store raw, cleaned, feature, signal, and result data separately.
- Use partitioned Parquet for local hot storage.
- Use Google Drive as cold artifact storage.
- Never load the whole global universe unless the task explicitly requires it.
- Every large run must write progress manifests so interrupted jobs can resume.

## Promotion Rule

No model can move toward paper trading unless it passes:

- Data QA.
- Walk-forward validation.
- Multiple-testing correction.
- Final holdout.
- Cost/slippage stress.
- Liquidity/capacity stress.
- Risk manager review.
- Human approval.

The correct culture is not "find a high Sharpe." The correct culture is
"kill weak strategies quickly and preserve the few that survive."
