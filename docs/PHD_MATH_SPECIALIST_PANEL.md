# PhD-Level Mathematics Specialist Panel

## Purpose

This panel turns the quantitative library into exactly 20 evidence-bound specialist agent roles. “PhD-level” describes the required depth of reasoning and review contract; it is not a claim that a software agent holds a real-world degree.

The panel does not replace deterministic research code. It defines who must challenge a model, what mathematical evidence they must request, and which defects block promotion.

## Canonical roster

| # | Specialist | Primary ownership |
|---:|---|---|
| 1 | Measure, Probability and Martingale | Filtrations, adaptedness, integrability, measure change, no-arbitrage |
| 2 | Stochastic Calculus, Rough Paths and Volterra | Ito/functional calculus, Levy and Volterra dynamics, signatures, rough volatility |
| 3 | Stochastic Control, HJB and BSDE | Control admissibility, dynamic programming, execution, hedging, market making |
| 4 | Numerical PDE, SDE and Monte Carlo | Convergence, stability, discretization and sampling error, numerical Greeks |
| 5 | Time-Series Dependence and Dynamical Systems | Stationarity, mixing, cointegration, spectral structure, regimes, filtering |
| 6 | High-Frequency Econometrics | Latent prices, noisy/asynchronous observations, realized variation, jumps |
| 7 | High-Dimensional Statistics and Random Matrices | Covariance spectra, factors, sparsity, shrinkage, p/n instability |
| 8 | Robust Statistics, Extremes and Rare Events | Heavy tails, contamination, EVT, tail dependence, crisis simulation |
| 9 | Bayesian Inference and Latent Variables | Posterior inference, filtering, hierarchical models, approximation diagnostics |
| 10 | Causal Identification and Invariant Learning | Estimands, interventions, orthogonal scores, confounding, regime invariance |
| 11 | Sequential Inference and Online Adaptation | Confidence sequences, e-processes, stopping, change detection, adaptive regret |
| 12 | Selection Bias and Backtest Validation | Nested purged CV, CPCV/PBO, DSR/PSR, Reality Check, SPA, FDR, holdout isolation |
| 13 | Dependent-Data Learning Theory | Generalization, stability, complexity, effective sample size, distribution shift |
| 14 | Neural Dynamics and Structured Sequences | Neural CDE/SDE/ODE, HiPPO, S4, Mamba, irregular sampling, long memory |
| 15 | Neural Operators and Scientific ML | Function-space learning, FNO/PINO, deep BSDE/PDE, physics-informed surrogates |
| 16 | Generative Diffusion and Schrodinger Bridges | Score SDEs, path laws, conditional scenarios, synthetic-data validation |
| 17 | Graph, Network and Systemic Risk | Dynamic graphs, spillovers, contagion, learned adjacency, relational identifiability |
| 18 | Optimization, Duality and Information Geometry | Convexity, KKT, solver correctness, sensitivity, natural gradients |
| 19 | Portfolio, DRO and Optimal Transport | Robust allocation, coherent risk, Wasserstein ambiguity, martingale transport |
| 20 | Microstructure, Hawkes and Execution | Point-process stability, queues, impact, fills, costs, venue realism |

The full mandates, routing tags, proof obligations, kill criteria, and corpus anchors live in `configs/phd_math_specialists.json`. Domain-specific AI/ML, neural-model, genetic/evolutionary optimization, and quant-fund missions live in `PhD_Math_Agent_Library/indexes/agent_capabilities.json` and each specialist's `SPECIALIST_PROFILE.md`.

## Review protocol

1. Route each material model claim to a three-agent panel: one lead and two non-overlapping challengers.
2. Each specialist writes an independent opinion before reading the other opinions.
3. Every claim is labeled theorem, assumption, empirical estimate, numerical approximation, or conjecture.
4. Challengers must provide a counterexample, violated assumption, alternative explanation, numerical failure, or an explicit statement that no fatal objection was found.
5. Deterministic code runs leakage checks, null simulations, toy cases, convergence tests, baseline comparisons, and ledger reconciliation.
6. The claim ledger records `accepted`, `conditional`, `rejected`, or `unresolved` with evidence.
7. One valid counterexample, leakage finding, identification failure, or reproducibility failure blocks promotion. Majority vote cannot overrule mathematics.

## Required verdicts

- `pass`: all material mathematical obligations are evidenced and deterministic gates pass.
- `conditional`: the result is valid only under named assumptions or unresolved but bounded limitations.
- `fail`: a material assumption, identification, proof, numerical, leakage, or reproducibility obligation fails.
- `abstain`: the question is outside the specialist's mandate or evidence is insufficient to evaluate it.

## Anti-handwaving standard

Terms such as robust, optimal, causal, stable, significant, calibrated, convergent, and arbitrage-free require a mathematical definition and evidence. A citation must map the cited theorem's assumptions to diagnostics on the current data and model. A high Sharpe ratio, agent confidence, or agent consensus is never mathematical evidence.

Neural models additionally require simple baselines, ablations, trial counts, seed distributions, null-label tests, leakage-safe normalization, and uncertainty under regime shift. Numerical models require tolerances and convergence evidence. Killed trials stay in the multiple-testing denominator.

## Python interface

```python
from quant_lab.agents.phd_math_specialists import build_review_panel

reviews = build_review_panel(
    "Audit a neural SDE trained on irregular order-book events",
    tags=["neural_sde", "microstructure"],
)

for review in reviews:
    print(review["specialist_id"], review["debate_role"])
    send_to_model(review["prompt"])
```

The router is deliberately transparent and deterministic. It prepares specialist prompts; it does not grant trading or promotion authority.
