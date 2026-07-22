# QForge-20

## Canonical name

**QForge-20 — the Bounded Adversarial PhD Quant Research Workflow**

Use `QForge-20` as the short name in plans, code reviews, campaign records, and
operator instructions. The name means:

- `QForge`: a repeatable quant-research process that turns hypotheses into
  falsifiable, reproducible evidence;
- `20`: the canonical registry of twenty PhD-level mathematical specialist
  roles;
- `Bounded`: iteration stops on predeclared trial, generation, token, cost,
  wall-time, and failure limits;
- `Adversarial`: counterexamples, leakage checks, simple baselines, and
  falsification have priority over model consensus.

## Status

The direct DeepSeek V4-Pro adapter, hybrid router, cloud-dossier sanitizer,
redacted token/cost receipts, resumable infrastructure failures, deterministic
fixtures, and provider tests are implemented. The closed ten-strategy Malaysia
cohort is registered under `QFORGE20-MALAYSIA-RESEARCH-V1` without falsely
relabeling its earlier Ollama reviews as V4-Pro work.

The first live focused post-backtest wave completed on 2026-07-13: ten frozen
Malaysia strategies received three isolated V4-Pro reviews each. The verified
index reports `review_complete`; the panel produced 28 conditional and two fail
verdicts, vetoed two strategies, and promoted none. This is adversarial
interpretation of already-contaminated research results, not new backtesting,
terminal evidence, or dry-run authorization.

The successful-call ledger contains 30 redacted receipts. Seventeen malformed
responses were safely deferred and retried, but this campaign version did not
persist their provider token/cost receipts. Its recorded USD subtotal is
therefore explicitly a lower bound. The machine report and human diagnosis are:

- `artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/v4_review_report.json`
- `docs/QFORGE_20_MALAYSIA_V4_FINDINGS.md`

Never place a DeepSeek API key in this repository, a prompt, a ledger, an
artifact, or a chat transcript. The runtime must obtain it from
`DEEPSEEK_API_KEY` or an approved local secret store.

## Alignment statement

QForge-20 assigns responsibilities as follows:

| Plane | Owner | Responsibility | Cannot do |
|---|---|---|---|
| Control plane | Codex | Plan campaigns, validate tool requests, invoke deterministic local code, enforce state and budgets, inspect evidence | Manufacture evidence, relax a failed gate, or self-promote a candidate |
| Planning plane | Local Qwen MLX | Intake, deduplication support, decomposition, specialist routing, bounded mutation planning, routine summaries | Perform final mathematical certification or tune against a holdout |
| Research plane | DeepSeek V4-Pro | Heavy mathematical research, coding proposals, specialist analysis, counterexamples, falsification design, and evidence-bound revision | Access secrets, raw restricted data, the terminal holdout, a broker, or unrestricted tools |
| Specialist plane | Twenty canonical PhD roles | Supply role-specific mandates, proof obligations, kill criteria, and corpus anchors | Treat a vote or shared model confidence as proof |
| Evidence plane | Deterministic Python services | Data validation, backtests, statistics, stress tests, factor/regime audits, risk checks, and replay | Change a hypothesis after observing protected results |
| Authority plane | Predeclared gates and explicit human approval | Decide research progression, freeze, terminal evaluation, and any later operational approval | Delegate promotion authority to an LLM |

The canonical execution chain is:

```text
Codex control plan
  -> local Qwen intake, decomposition, and specialist routing
  -> independent DeepSeek V4-Pro specialist research
  -> Codex-validated deterministic tool execution
  -> DeepSeek V4-Pro adversarial interpretation and revision proposal
  -> deterministic validation and institutional gates
  -> bounded child trial or frozen finalist
  -> one-time terminal evaluation on genuinely future data
```

Codex and Qwen plan and coordinate. DeepSeek V4-Pro performs the heavy
research. Deterministic evidence adjudicates. No model decides that its own
work is true.

## The twenty-specialist operating model

The source of truth for specialist identity is
`configs/phd_math_specialists.json`. The roles are evidence-bound analytical
specifications, not claims that twenty distinct human researchers or twenty
independent foundation models are present.

DeepSeek V4-Pro may instantiate any of the twenty roles in an isolated request
using that role's:

- mandate;
- routing tags;
- proof obligations;
- kill criteria;
- corpus anchors;
- required structured dossier fields.

One foundation model serving twenty contexts still has correlated errors.
Therefore:

- first opinions are generated in isolated contexts before debate;
- the proposer cannot review or approve its own exact output context;
- one concrete, reproducible counterexample can block progression;
- majority vote never establishes mathematical truth;
- deterministic tests, baselines, and independent data partitions remain the
  decisive evidence.

### Research waves

QForge-20 uses staged specialist coverage rather than spending twenty V4-Pro
calls on every weak idea.

1. **Focused research wave.** Qwen routes the frozen research ticket to three
   to five relevant specialists. The selection-bias specialist is mandatory.
2. **Independent challenge wave.** Relevant challengers test identification,
   dependence, robustness, numerical validity, learning/generalization,
   portfolio construction, cost, capacity, and execution assumptions.
3. **Evidence wave.** V4-Pro requests only allowlisted deterministic tools.
   Tool results are immutable evidence linked by hashes.
4. **Finalist coverage wave.** A candidate that survives preliminary gates is
   checked for coverage across all twenty mandates. Only in-scope specialists
   render a substantive verdict; routing gaps and mandatory abstentions are
   resolved before freeze.

All twenty roles participate in the research system. They do not all need to
execute on every trial.

## Quant research lifecycle

### 1. Institutional hypothesis intake

Before any expensive model call:

- state a falsifiable economic claim;
- declare the market, universe, horizon, features, labels, costs, and required
  point-in-time data;
- declare the simple baseline and failure region;
- assign a hypothesis identity, exact content hash, and semantic fingerprint;
- reject exact duplicates and reordered near-duplicates;
- freeze the research cutoff and terminal cutoff;
- reserve the trial in the lifetime ledger before generation or backtesting.

### 2. Qwen planning and routing

Qwen converts the admitted ticket into a bounded research plan. It may select
specialists, order proof obligations, identify required corpus anchors, and
prepare a sanitized dossier. It may not introduce hidden parameter searches,
change the frozen market scope, or see a terminal result.

### 3. Independent V4-Pro research

Each selected specialist receives the same frozen candidate identity but only
the context needed for its mandate. The default direct-API model is
`deepseek-v4-pro` with thinking enabled at `high` effort. `max` effort is
reserved for explicit escalation of a difficult proof, conflict, or finalist
audit.

Each response must distinguish:

- theorem;
- assumption;
- empirical estimate;
- numerical approximation;
- conjecture.

Each response must produce a strict contract containing claims, assumptions,
counterexamples, falsification tests, verdict, confidence, and any requested
allowlisted tool calls.

### 4. Controlled tool execution

V4-Pro does not receive unrestricted shell or filesystem access. Codex or the
durable controller validates and executes narrow functions such as:

- `search_math_corpus`;
- `read_trial_evidence`;
- `run_data_availability_check`;
- `run_candidate_backtest`;
- `run_null_test`;
- `run_ablation`;
- `run_falsification_test`;
- `run_regime_audit`;
- `run_factor_audit`;
- `run_cost_capacity_stress`;
- `run_unit_tests`;
- `submit_staged_patch`.

Tools may not expose credentials, broker functions, live-order functions,
arbitrary network access, terminal-holdout values, or undeclared datasets.
Generated code lands in a staging area and must pass review and tests before it
can enter the research engine.

### 5. Evidence adjudication

LLM prose never substitutes for a calculation. Candidate progression remains
subject to the existing deterministic controls, including:

- point-in-time and causal-availability checks;
- purged or embargoed out-of-sample validation;
- lifetime multiple-testing accounting;
- probabilistic and deflated-Sharpe evidence;
- probability-of-backtest-overfitting controls;
- drawdown, turnover, concentration, cost, capacity, and stress limits;
- robustness across universes and predeclared regimes;
- factor-adjusted performance and factor-repackaging checks;
- parameter, seed, sampling, and numerical-stability tests;
- rolling decay and genuinely future paper evidence when it exists.

Raw Sharpe is diagnostic evidence, not a promotion rule.

### 6. Bounded reiteration

V4-Pro is allowed to reiterate, challenge, write staged research code, request
tests, and propose a revision. It is not allowed to reiterate until something
passes.

Every material revision must:

- create a separately reserved child trial;
- preserve parent trial and candidate identity;
- consume blinded, enumerated failure or mutation codes;
- count in the lifetime trial denominator;
- retain the original evidence and never overwrite a failed result;
- remain within predeclared family, generation, token, cost, and wall-time
  budgets.

The current safety baseline remains at most three mutation generations per
lineage, thirty new trials per campaign, and five consecutive failures before
stopping. Continuous research means scheduling completed finite campaigns. It
does not mean an immortal loop or a lineage that keeps searching until it
passes.

### 7. Finalist and conflict handling

A finalist receives broader specialist coverage and an adversarial synthesis.
Conflicting claims are resolved by an additional falsification test, a proof,
an independently computed baseline, or explicit human escalation. They are not
resolved by majority vote or by asking the same model repeatedly until its
answer changes.

Infrastructure failures have their own state:

- timeout, rate limit, quota exhaustion, provider outage, or malformed
  transport response -> `deferred`;
- mathematical counterexample or failed deterministic gate -> research
  failure;
- missing required evidence -> incomplete, not pass;
- protected-data or policy violation -> blocked.

A deferred V4-Pro review can resume idempotently. It must never become an
automatic research failure or an accidental pass.

### 8. Research freeze and terminal separation

The research loop may use research-period evidence only. Once a candidate is
frozen:

- its code, parameters, features, universe, costs, and decision rule are
  content-addressed;
- no specialist may use terminal results to revise, rank, or mutate it;
- the terminal service uses observations strictly after the frozen cutoff;
- a terminal failure is terminal for that frozen candidate;
- any later idea begins a new lineage and cannot reclaim the consumed holdout.

Historical data already used for candidate generation, ranking, mutation, or
review is contaminated research evidence, not a pristine holdout.

### 9. Operational boundary

QForge-20 ends at research evidence, freeze, terminal assessment, and an
explicitly authorized deterministic dry run. It provides no broker authority,
account authority, live-order function, or automatic capital allocation.

## Direct DeepSeek API boundary

The target V4-Pro research plane uses the direct DeepSeek OpenAI-compatible
endpoint, not Ollama Cloud. Direct access provides explicit token accounting
and avoids subscription-session limits, but it changes the privacy boundary.

Requirements:

- model: `deepseek-v4-pro`;
- endpoint: `https://api.deepseek.com`;
- secret source: `DEEPSEEK_API_KEY` or approved local secret store;
- default reasoning: thinking enabled, `reasoning_effort=high`;
- escalation reasoning: `max` only through an explicit policy decision;
- no hosted fallback to another model;
- sanitized structured dossiers only;
- per-campaign token and USD ceilings;
- recorded provider, model, timestamps, token counts, cost, stop reason, and
  prompt/schema/config/response hashes;
- no API key, raw prompt body, raw response body, or environment secret in the
  ledger.

DeepSeek context caching is automatic. No raw licensed dataset, proprietary
position, credential, broker information, or terminal-holdout content may be
placed in a cloud dossier.

## Durable state machine

The target campaign states are:

```text
admitted
  -> planned
  -> specialist_research
  -> awaiting_tools
  -> evidence_ready
  -> specialist_challenge
  -> reviewed
  -> child_trial_reserved | institutional_audit
  -> finalist_coverage
  -> frozen
  -> terminal_pending
  -> terminal_pass | terminal_fail
```

`deferred` is a resumable infrastructure substate available to cloud-facing
steps. Every transition must be ledger-backed and idempotent.

## Promotion invariants

QForge-20 is aligned only when all of the following remain true:

1. Trial identity is reserved before model generation or backtesting.
2. The lifetime denominator never resets between campaigns.
3. Qwen planning cannot bypass institutional intake.
4. V4-Pro cannot self-approve or change a candidate without a child trial.
5. Twenty role contexts are not treated as twenty independent models.
6. One reproducible counterexample can block progression.
7. Infrastructure failure is deferred, never converted into evidence.
8. Deterministic tests adjudicate model claims.
9. Research-period and terminal evidence never flow backward across the
   firewall.
10. Promotion gates cannot be relaxed after seeing a failure.
11. API spend, tokens, attempts, generations, and wall time are bounded.
12. No agent has broker or live-trading authority.

## Implementation acceptance criteria

QForge-20 must not be described as active until all of these are verified:

- a direct `DeepSeekProvider` reads its key only from an approved secret
  source;
- provider routing assigns V4-Pro to heavy specialist research without
  changing local Qwen planning routes;
- cloud requests require an explicit `cloud_sanitized` policy classification;
- sanitizer tests prove restricted fields cannot enter a request;
- high/max reasoning and structured-response parsing are tested;
- rate limits and outages enter a resumable deferred state;
- token and monetary usage stop the campaign at configured ceilings;
- the twenty-role registry and staged routing remain deterministic;
- proposer and reviewer contexts are isolated and content-addressed;
- child mutations preserve lineage and consume the lifetime denominator;
- offline replay produces identical transitions without API access;
- terminal-holdout and broker boundaries remain enforced;
- operator documentation explains key provisioning, recovery, cost monitoring,
  and emergency shutdown.

## Operator commands

Register or refresh all frozen strategy specifications and research-period
results under the QForge-20 name without making a hosted call:

```bash
uv run python examples/register_qforge20_malaysia_results.py \
  --v4-review-status pending_rotated_key
```

After rotating the exposed credential, store it in the operator's macOS
Keychain using the interactive prompt below. `-w` is deliberately last so the
secret is prompted for instead of appearing in shell history:

```bash
security add-generic-password -U \
  -a "$USER" \
  -s qforge20-deepseek \
  -w
```

The provider checks `DEEPSEEK_API_KEY` first and then this Keychain service.
Never pass the key as a command argument or paste it into chat. Run at most two
strategies per bounded campaign:

```bash
uv run python examples/run_qforge20_malaysia_reviews.py \
  --campaign-id QFORGE20-MYS-B01 \
  --batch-start 1 \
  --batch-size 2 \
  --max-campaign-cost-usd 0.60
```

Repeat with starts 3, 5, 7, and 9. Each strategy receives three isolated,
role-specific V4-Pro reviews, including the mandatory selection-bias role.
Weak strategies do not consume all twenty calls; all twenty mandates are
reserved for finalist coverage as defined above.

Implementation locations:

- Direct provider: `src/quant_lab/research/providers/deepseek.py`
- Cloud sanitizer: `src/quant_lab/research/providers/cloud_sanitizer.py`
- Hybrid router: `src/quant_lab/research/providers/qforge20.py`
- Result index: `src/quant_lab/research/qforge20_workflow.py`
- Review runner: `examples/run_qforge20_malaysia_reviews.py`
- Verified report builder: `examples/build_qforge20_malaysia_review_report.py`
- Routing policy: `configs/qforge20_routing.json`

Build or refresh the evidence-verified campaign report after all batches finish:

```bash
uv run python examples/build_qforge20_malaysia_review_report.py
```

Local Ollama remains the no-cost planning and routine-research baseline;
DeepSeek is an optional bounded heavy-research plane. QForge-20 must continue
to fail closed if authentication, quota, privacy, cost, or provider checks
fail. A credential pasted into chat is exposed even if it is also held in the
Keychain and must be revoked and replaced before any later hosted campaign.
