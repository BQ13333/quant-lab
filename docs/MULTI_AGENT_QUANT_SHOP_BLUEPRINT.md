# Multi-Agent Quant Shop Blueprint

## Purpose

Build a permanent ASEAN/Asia public-equity quant research shop where agents coordinate source mining, data acquisition, strategy implementation, PhD-level validation, portfolio construction, reporting, and paper-trading readiness.

Agents do not decide what trades. Deterministic Python, data manifests, validation gates, and human approval decide whether anything can be promoted.

## Current Source Coverage Baseline

The official external book-corpus inventory is:

- Artifact: `artifacts/source_coverage/SOURCE-COVERAGE-BOOKS-20260707`
- External markdown books scanned: `69`
- Headings inventoried: `32,471`
- Strategy/model-like headings: `4,884`
- Data-dependent headings: `893`
- Current source strategy specs: `15`
- Implemented specs: `11`
- Blocked specs: `4`

There is also a broader repo-inclusive inventory at `artifacts/source_coverage/SOURCE-COVERAGE-08Q-20260707`, but source coverage decisions should use the book-corpus ledger unless the task explicitly concerns repo documentation.

Core files:

- `markdown_files.csv`
- `heading_inventory.csv`
- `strategy_heading_inventory.csv`
- `data_dependency_heading_inventory.csv`
- `source_strategy_spec_coverage.csv`
- `source_file_coverage.csv`
- `source_coverage_manifest.json`

Rule: every strategy implementation must link back to a source coverage row, a manually curated source spec, or a deliberate "original research" note. No orphan strategy code.

## Command Center

```text
Research Director
  -> Source Mining Cell
  -> Data Factory Cell
  -> Feature Factory Cell
  -> Strategy Coding Cell
  -> PhD Math Validation Cell
  -> Portfolio Construction Cell
  -> Risk, Capacity, TCA Cell
  -> Reporting and Obsidian Cell
  -> Paper-Trading Readiness Cell
  -> Human Approval Gate
```

Each cell has agents, artifact contracts, pass/fail definitions, and escalation rules. Coordination should eventually run in LangGraph, but every cell must first be executable as deterministic Python scripts and tests.

## Closed-Loop Alpha Factory

The core research loop is not "find the highest Sharpe." The loop is "find alpha that survives adversarial evidence." A high Sharpe is useful only if it survives out-of-sample testing, multiple-testing correction, cost stress, capacity checks, and source/data provenance.

### Active Universe

Default tradable research universe:

- Malaysia.
- Singapore.
- Indonesia.

Removed from active project universe:

- Philippines, because we cannot trade it.
- Vietnam.

Parked outside current scope:

- Thailand.

### Prime Objective

The `chief_quant_pm` owns the alpha objective and decides whether a research loop continues, mutates, pauses for data, or kills a family.

The standing PM mandate is:

- Net out-of-sample Sharpe promotion line: `3.0`.
- Minimum deflated Sharpe probability: `0.95`.
- Sharpe 3 is a promotion gate, not permission to overfit.

Promotion is based on a composite score:

- Out-of-sample net Sharpe and net information ratio.
- Deflated Sharpe and probabilistic Sharpe.
- PBO / CPCV stability.
- Final sealed holdout behavior.
- Cost, turnover, drawdown, liquidity, and capacity survival.
- Incremental value versus simpler baselines.
- PnL and position correlation versus existing candidates.
- Source originality and explainability.

### Loop Topology

```text
0. Chief Quant PM defines alpha mission
   -> 1. Markdown Source Extraction Cell mines books/notes
   -> 2. Formula Registry Cell converts ideas into candidate specs
   -> 3. Data QA Cell checks whether required data exists point-in-time
   -> 4. Feature Factory Cell builds lagged, lineage-tagged features
   -> 5. Model Factory Cell runs baseline, ML, ensemble, and original variants
   -> 6. Backtest Engine Cell runs U2/U3 universes with frozen splits
   -> 7. Olympiad Math Cell tries to disprove the result
   -> 8. Portfolio PM Cell builds constrained mixtures
   -> 9. Risk/TCA Cell applies cost, capacity, and stress
   -> 10. Research Memo Cell writes model cards and kill reports
   -> 11. PM Decision Gate: promote, mutate, park for data, or kill
   -> if gate not met, return to the narrowest failing upstream node
```

### Agent And Sub-Agent Design

`chief_quant_pm`:

- Owns the alpha book.
- Rejects high-Sharpe artifacts that are overfit, too concentrated, untradeable, or not original enough.
- Decides when the loop should continue versus kill a strategy family.

`markdown_librarian` sub-agents:

- `source_inventory_agent`: scans a user-configured, authorized research-note directory.
- `formula_extractor_agent`: extracts formulas, model families, assumptions, required data, and implementation hints.
- `strategy_taxonomist_agent`: classifies ideas into trend, mean reversion, value, quality, revisions, event, liquidity, volatility, microstructure, portfolio construction, and risk.
- `source_provenance_agent`: assigns every strategy a source ID or explicit original-research ID.
- `copyright_safety_agent`: stores paraphrased summaries and references, not long copied book text.

`model_factory` sub-agents:

- `baseline_model_agent`: price/volume rules, source-backed strategies, simple factor baselines.
- `ml_model_agent`: regularized linear models, tree models, gradient boosting, random forests, meta-labeling.
- `deep_learning_agent`: only after simple baselines are strong; uses strict purged CV and small parameter budgets.
- `genetic_formula_agent`: mutates formulas under grammar constraints; every mutation counts in multiple-testing.
- `regime_router_agent`: routes model weights by volatility/liquidity/regime states.
- `ensemble_agent`: blends weak alphas using correlation, breadth, and transfer-coefficient penalties.

`olympiad_math_cell` sub-agents:

- `proof_obligation_agent`: writes the null hypothesis, leakage risks, and minimum proof needed.
- `cpcv_pbo_agent`: estimates probability of backtest overfitting.
- `deflated_sharpe_agent`: adjusts Sharpe for non-normality and many trials.
- `reality_check_agent`: runs White Reality Check / SPA style tests.
- `bootstrap_agent`: performs block bootstrap and stress resampling.
- `parameter_surface_agent`: rejects boundary optima and fragile narrow peaks.
- `adversarial_validation_agent`: tests whether winners are data artifacts, country artifacts, or time-period artifacts.

The canonical successor is the exactly 20-role PhD-level mathematics panel in
`docs/PHD_MATH_SPECIALIST_PANEL.md` and `configs/phd_math_specialists.json`.
The older olympiad names remain capability aliases so they do not create an
additional, overlapping set of mathematical agents.

`portfolio_pm_cell` sub-agents:

- `alpha_book_pm`: ranks surviving alpha families, not individual overfit variants.
- `allocator_agent`: constrained long-only, group-neutral, HRP, robust covariance, CVaR, and turnover-aware optimizers.
- `risk_budget_agent`: assigns risk budget by country, model family, and liquidity.
- `capacity_agent`: checks ADV participation and USD 5m / larger AUM scaling.
- `kill_switch_agent`: defines drawdown, decay, and live-monitoring tripwires.

### Loop-Back Rules

The loop does not blindly keep optimizing. It routes the failure to the right owner:

- Data failure: return to Data Factory or park the strategy as `blocked_by_data`.
- Leakage failure: return to Feature Factory and Validation Cell.
- Weak net return: return to Model Factory for feature/model mutation.
- High turnover/cost failure: return to Portfolio Construction and TCA.
- Concentration failure: return to allocator and universe breadth review.
- Multiple-testing failure: kill or demote, unless a simpler source-backed explanation survives.
- Holdout failure: kill the exact variant; optionally preserve the family for a future independent test.
- Strong but correlated candidate: ensemble only if it adds marginal risk-adjusted return.

### Candidate Ledger

Every attempted model, including failures, must write:

- `candidate_id`
- `parent_candidate_id`
- `source_strategy_id`
- `formula_family`
- `feature_set_id`
- `model_class`
- `hyperparameter_hash`
- `train_window`
- `validation_window`
- `holdout_window`
- `trial_number`
- `net_sharpe`
- `net_ir`
- `max_drawdown`
- `turnover`
- `cost_assumption_bps`
- `pbo`
- `deflated_sharpe`
- `psr`
- `capacity_usd`
- `promotion_state`
- `kill_reason`

This ledger is the protection against self-deception. Killed ideas stay in the denominator.

### Minimum Gate For A Research-Grade Candidate

Before anything is called research-grade:

- U2 and U3 backtests both run.
- Final holdout from `2024-01-02` to `2026-07-06` remains sealed until selection.
- Net out-of-sample Sharpe is at least `3.0` for PM promotion.
- Deflated Sharpe probability is at least `0.95`.
- Net Sharpe is positive after realistic and stressed costs.
- Deflated Sharpe is acceptable after counting all trials.
- CPCV/PBO does not scream overfit.
- Parameter surface is not a one-cell spike.
- Portfolio passes max weight, liquidity, turnover, drawdown, and country exposure constraints.
- The model has a source-backed or clearly documented original-research rationale.

### What The Loop Optimizes

Primary target:

- Best robust out-of-sample net information ratio / Sharpe, not in-sample Sharpe.

Secondary targets:

- Lower drawdown.
- Lower turnover.
- More breadth.
- More independent alpha contribution.
- Higher transfer coefficient from signal to portfolio.
- Better capacity at USD 5m and above.
- Cleaner explanation and source lineage.

### What The Loop Must Not Do

- Do not select by raw Sharpe alone.
- Do not tune on the final holdout.
- Do not ignore killed trials.
- Do not let ML agents create untraceable features.
- Do not promote a model whose edge disappears at realistic costs.
- Do not accept a strategy whose alpha is one stock, one month, or one country event.
- Do not let agents edit promotion results manually.

## Agent Cells

### 1. Research Director Cell

Agents:

- `chief_quant_pm`
- `research_queue_manager`
- `experiment_controller`
- `promotion_committee_secretary`

Owns:

- Strategy backlog.
- Experiment priority.
- Source-to-code coverage.
- Promotion decisions.

Artifacts:

- `artifacts/research_queue/strategy_backlog.csv`
- `artifacts/research_queue/experiment_queue.csv`
- `artifacts/research_queue/promotion_minutes.md`
- `vault/10_maps/Quant Shop Map.md`

Done criteria:

- Every active task has owner, source, data requirement, expected artifact, and gate.
- No strategy can move to validation without a source ID and data lineage.

### 2. Source Mining Cell

Agents:

- `markdown_librarian`
- `strategy_taxonomist`
- `formula_extractor`
- `source_gap_auditor`
- `copyright_safety_editor`

Owns:

- All `.md` inventory.
- Strategy heading coverage.
- Formula and method taxonomy.
- Mapping source ideas to implementable or data-blocked specs.

Inputs:

- `${QUANT_LAB_PRIVATE_WORKSPACE}/**/*.md` (user-supplied and authorized)
- `artifacts/source_coverage/*`
- `src/quant_lab/models/source_strategies.py`

Artifacts:

- `source_coverage_manifest.json`
- `strategy_heading_inventory.csv`
- `source_strategy_manifest.csv`
- `source_to_model_gap.csv`
- `blocked_by_data.csv`

Done criteria:

- Every high-priority source heading is classified as `implemented`, `requires_data`, `not_applicable`, or `needs_human_review`.
- No long copyrighted excerpts are copied into artifacts.
- Every implemented source idea has test coverage and provenance columns in the scoreboard.

### 3. Data Factory Cell

Agents:

- `asean_security_master_agent`
- `north_asia_security_master_agent`
- `refinitiv_symbol_map_agent`
- `corporate_actions_agent`
- `delisting_survivorship_agent`
- `fx_benchmark_agent`
- `fundamentals_pit_agent`
- `analyst_revisions_agent`
- `earnings_events_agent`
- `liquidity_adv_agent`

Owns:

- Security master.
- Symbol maps.
- PIT fundamentals.
- Analyst estimates/revisions.
- Earnings timestamps.
- Corporate actions and delistings.
- OHLCV, FX, ADV, benchmarks.

Artifacts:

- `listings.csv`
- `symbol_map.csv`
- `security_master.parquet`
- `corporate_actions.csv`
- `delistings.csv`
- `earnings_events.parquet`
- `analyst_revisions.parquet`
- `fundamentals_pit.parquet`
- `ohlcv/`
- `coverage.csv`
- `quality_flags.csv`
- `data_manifest.json`

Done criteria:

- Every row has vendor, fetch timestamp, source timestamp, currency, exchange, security type, and survivorship status.
- Historical listed universe and current listed universe are separate.
- PIT fields include known-at/release timestamps and revision policy.
- Failed downloads are in failure manifests; no silent ticker drops.

### 4. Data QA Cell

Agents:

- `coverage_auditor`
- `calendar_alignment_agent`
- `split_dividend_sanity_agent`
- `outlier_return_agent`
- `stale_price_agent`
- `lookahead_leakage_agent`

Owns:

- Data eligibility gates before backtesting.

Artifacts:

- `qa_summary.md`
- `coverage.csv`
- `quality_flags.csv`
- `leakage_checks.csv`
- `eligible_universe.csv`

Done criteria:

- Backtests cannot consume data without QA artifacts.
- Missingness, stale rows, outliers, suspicious split returns, and calendar gaps are flagged.
- PIT leakage checks pass before fundamentals/revisions enter models.

### 5. Feature Factory Cell

Agents:

- `price_volume_feature_agent`
- `fundamental_factor_agent`
- `earnings_surprise_agent`
- `analyst_revision_agent`
- `macro_fx_agent`
- `microstructure_feature_agent`
- `feature_lineage_agent`

Owns:

- Feature creation with known-at timestamps.
- Feature caching.
- Feature lineage.

Artifacts:

- `feature_store/<feature_family>/`
- `feature_spec.yaml`
- `feature_lineage.csv`
- `feature_qa.csv`
- `feature_cache_manifest.json`

Done criteria:

- Every feature declares input columns, lag policy, lookback, rebalance cadence, and known-at time.
- Feature caches are reusable by model sweeps.
- Unit tests prove shape stability and no future returns/constituents are used.

### 6. Strategy Coding Cell

Agents:

- `source_strategy_porter`
- `model_api_guardian`
- `strategy_unit_test_agent`
- `source_provenance_agent`
- `feature_cache_adapter`

Owns:

- Turning source ideas into `AlphaModel` implementations.
- Keeping strategy implementations small, testable, and provenance-tagged.

Artifacts:

- `src/quant_lab/models/source_strategies.py`
- `tests/test_source_strategies.py`
- `source_strategy_manifest.csv`
- `strategy_model_card.md`

Done criteria:

- Every model emits finite scores.
- Every model exposes `meta_source_strategy_id`, `meta_source_file`, `meta_family`, and data requirements.
- Data-blocked models are registered as blocked, not faked with price-only proxies.

### 7. PhD Math Validation Cell

Agents:

- `nested_cv_agent`
- `cpcv_pbo_agent`
- `deflated_sharpe_agent`
- `white_reality_check_agent`
- `spa_test_agent`
- `fdr_multiple_testing_agent`
- `ic_decay_breadth_agent`
- `regime_stability_agent`
- `bayesian_model_risk_agent`
- `bootstrap_stress_agent`

Owns:

- Anti-overfit proof.
- Statistical significance.
- Robustness under costs, regimes, universes, and time.

Current math already present:

- Walk-forward validation.
- Purged/embargoed split scaffolding.
- Deflated Sharpe.
- Cost stress.
- Drawdown, turnover, concentration gates.
- PBO proxy and effective independent trials.
- Candidate return correlation.

Next required math:

- Full CPCV path-based PBO.
- White's Reality Check.
- Hansen SPA test.
- Benjamini-Hochberg FDR for large candidate pools.
- Nested model selection with sealed final holdout.
- Block bootstrap confidence intervals.
- IC decay and breadth estimation.
- Bayesian shrinkage for noisy alpha estimates.
- Regime-conditioned stability tests.
- Turnover-adjusted information coefficient.

Artifacts:

- `overfit_audit.json`
- `cpcv_pbo.csv`
- `reality_check.csv`
- `spa_test.csv`
- `fdr_adjustment.csv`
- `ic_decay.csv`
- `holdout_report.md`
- `promotion_gate_report.json`

Done criteria:

- A high Sharpe alone never promotes a model.
- Champion selection uses nested CV or a sealed holdout.
- Multiple testing count includes all attempted candidates, including killed ones.
- Audit posture must be `research_grade` or stronger before paper trading.

### 8. Portfolio Construction Cell

Agents:

- `mean_variance_agent`
- `robust_covariance_agent`
- `black_litterman_agent`
- `risk_parity_agent`
- `hrp_agent`
- `mean_cvar_agent`
- `turnover_constrained_optimizer_agent`
- `country_sector_neutrality_agent`
- `ensemble_weight_agent`

Owns:

- Converting alpha scores into portfolios.
- Strategy mixture optimization.
- Constraints and implementation efficiency.

Math roadmap:

- Ledoit-Wolf/OAS covariance shrinkage.
- Robust mean-variance optimization.
- Mean-CVaR and drawdown-aware optimization.
- Hierarchical risk parity.
- Black-Litterman-style blending of priors and alpha forecasts.
- Ensemble weights with correlation penalty.
- Transfer coefficient estimation from ideal to constrained portfolio.

Artifacts:

- `portfolio_optimizer_manifest.json`
- `weights.parquet`
- `exposure_report.csv`
- `constraint_report.csv`
- `transfer_coefficient.csv`
- `ensemble_weight_report.csv`

Done criteria:

- Portfolio is evaluated net of costs and turnover.
- Concentration, country, liquidity, and drawdown limits pass.
- Ensemble candidates show incremental value over best single strategy after correlation penalty.

### 9. Risk, Capacity, And TCA Cell

Agents:

- `liquidity_capacity_agent`
- `transaction_cost_model_agent`
- `slippage_agent`
- `stress_scenario_agent`
- `drawdown_kill_switch_agent`
- `paper_fill_reconciliation_agent`

Owns:

- ADV/capacity.
- Cost curves.
- Slippage/TCA.
- Tradeability.
- Kill switches.

Artifacts:

- `capacity_report.csv`
- `slippage_model.json`
- `tca_report.csv`
- `stress_report.md`
- `kill_switch_rules.yaml`

Done criteria:

- No production order path unless paper fills reconcile.
- Capacity uses ADV, spread, volatility, market-specific cost schedules, and position size.
- Cost stress at 1x/2x/3x is reported.

### 10. Reporting And Obsidian Cell

Agents:

- `model_card_writer`
- `tear_sheet_agent`
- `experiment_ledger_agent`
- `weekly_research_digest_agent`
- `audit_pack_agent`

Owns:

- Human-readable evidence.
- Permanent research memory.

Artifacts:

- `vault/30_experiments/<run_id>/index.md`
- `model_card.md`
- `backtest_tearsheet.md`
- `killed_strategy_report.csv`
- `weekly_digest.md`

Done criteria:

- Every experiment has a machine artifact and a human-readable note.
- Notes link to exact CSV/Parquet/JSON files.
- Killed strategies are retained for multiple-testing accounting.

### 11. Execution Readiness Cell

Agents:

- `paper_order_agent`
- `ibkr_gateway_agent`
- `broker_safety_agent`
- `reconciliation_agent`
- `human_approval_agent`

Owns:

- Paper trading only until explicit approval.
- IBKR path safety.

Artifacts:

- `paper_orders.csv`
- `paper_fills.csv`
- `broker_reconciliation.csv`
- `execution_safety_checklist.md`

Done criteria:

- Live routing is disabled by default.
- Every order file has pre-trade risk checks.
- Human approval is required before production.

## Coordination Protocol

### Daily Loop

1. Source Mining Cell updates source coverage and strategy backlog.
2. Data Factory Cell updates data manifests and coverage.
3. Data QA Cell marks eligible universes.
4. Feature Factory Cell builds only approved PIT-safe features.
5. Strategy Coding Cell implements source-backed models.
6. PhD Math Cell runs validation and audit.
7. Portfolio Cell constructs mixtures.
8. Risk/TCA Cell checks tradeability.
9. Reporting Cell writes model cards and killed-strategy reports.
10. Research Director decides promote, rework, or kill.

### Handoff Contract

Every handoff must include:

- `run_id`
- `agent_id`
- `input_artifacts`
- `output_artifacts`
- `data_asof_utc`
- `source_asof_utc`
- `known_limitations`
- `pass_fail`
- `next_owner`

### Promotion States

```text
idea
  -> source_mapped
  -> data_requirements_known
  -> data_available
  -> feature_built
  -> backtested
  -> audit_passed
  -> risk_passed
  -> paper_trading_ready
  -> production_candidate
```

Production candidate is not production. Production still requires human approval, broker safety, paper-trading evidence, and compliance review.

## Immediate Work Packages

### WP1: Source Coverage Completion

- Use `examples/run_source_coverage.py`.
- Rank top source files by strategy headings.
- Build `source_to_model_gap.csv`.
- Move high-priority headings into `source_strategy_manifest.csv`.

### WP2: Feature Cache

- Cache rolling returns, volatility, skew, moving averages, regression slopes, residuals, ranks, and beta estimates.
- Reuse feature matrices across source strategy sweeps.
- This is mandatory before longer Refinitiv history.

### WP3: Full Anti-Overfit Upgrade

- Implement full CPCV path-based PBO.
- Add White Reality Check and SPA test.
- Add FDR-adjusted candidate significance.
- Keep killed candidates in the multiple-testing denominator.

### WP4: PIT Data Expansion

- Refinitiv earnings dates.
- Analyst estimate revisions.
- PIT fundamentals.
- Corporate actions and delistings.
- ADV/spread/capacity.

### WP5: Strategy Mixture Engine

- Optimize ensembles across surviving source families.
- Penalize correlated PnL and correlated positions.
- Require incremental contribution over simpler baselines.

## Non-Negotiable Rules

- No orphan strategies.
- No silent ticker drops.
- No future data.
- No survivorship-only promotion.
- No high-Sharpe promotion without multiple-testing correction.
- No live trading path before paper trading.
- No agent may bypass deterministic validation.
