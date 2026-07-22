# Malaysia Loop Recovery Runbook

This runbook covers only the finite, local-Ollama Malaysia research loop. Never work around a gate by editing the SQLite database, deleting failed trials, changing the fixed universe/cost policy, or moving terminal evidence into research inputs.

## First checks

```bash
ollama ps
ollama list
curl -fsS --max-time 10 http://127.0.0.1:11434/api/tags
uv run python -m compileall -q src/quant_lab
uv run pytest -q tests/test_model_providers.py tests/test_specialist_runtime.py tests/test_bounded_research_loop.py
```

No API key is required. If the endpoint is not loopback, stop and correct the configuration.

## Normal resume after interruption

Run the same campaign command with the same campaign ID, budgets, initial queue, model routing, and universe source. `campaign_state.json` is written atomically. The controller verifies its configuration hash before resuming.

Completed candidate calls and each completed specialist role are replayed from immutable ledger evidence. They are not called again. An in-progress network request with no durable response may be attempted again; it has not produced an accepted contract or research result.

The shared lifetime vault is separate from each campaign state directory. Import only completed source ledgers. An active source, changed source fingerprint, conflicting trial identity, broken correction target, or evidence hash mismatch must fail the import.

If resume reports configuration drift, do not edit the state file. Restore the exact prior configuration or start a new campaign ID with a parent/reference note explaining the change.

An interruption after `backtest` but before `institutional_audit` resumes at the deterministic audit. An interruption after the audit resumes from its persisted evidence and must not recalculate a different regime or factor specification. The `institutional_audit` ledger event must precede every `dry_run` event.

## Hypothesis intake conflict

Do not rename and resubmit a rejected duplicate. Inspect the persistent intake reason (`EXACT_DUPLICATE` or `SEMANTIC_DUPLICATE`). A child may enter only with an admitted parent and blinded enum mutation codes. Never add performance numbers, holdout results, or a looser threshold to make a duplicate look new.

If a hypothesis ID conflicts with a different definition, stop the campaign and create a genuinely new reviewed ticket. Do not edit the SQLite row.

## Provider unavailable or timed out

- Confirm the model name is installed exactly.
- Confirm Ollama listens on `127.0.0.1:11434`.
- Check `ollama ps` for memory pressure.
- Structured calls use an 8k context. Do not increase it merely to avoid a schema failure.
- A timeout, rate limit, unavailable model, retry exhaustion, malformed JSON, or schema mismatch is a typed failure and cannot approve a candidate.

The 70B model is intentionally used only for the final adversarial review in the standard three-role panel. Routine and selection-bias reviews use Qwen. This keeps the local process operational while retaining an independent adversarial model.

## Malformed local output

Do not strip unknown fields or coerce arbitrary values. The contract reports the failing field name without logging its value. Improve the local schema instruction or task payload, then start/resume under the bounded policy.

A recorded malformed specialist attempt remains a failure on resume. Majority agreement cannot override it. A later fix requires a new child candidate or a new explicitly scoped campaign, not mutation of the old event.

A conditional panel may run the backtest required by its falsification tests, but it cannot reach institutional audit or dry run. Do not rewrite `conditional` as `pass`; preserve the review receipt and its blocking failure code.

## Ledger contention

SQLite uses WAL, foreign keys, a 30-second busy timeout, immediate reservation transactions, and unique idempotency keys. If contention persists:

1. Stop duplicate supervisors for the same campaign.
2. Confirm no manual SQLite shell has an open write transaction.
3. Rerun the same campaign command.
4. Verify that the reservation returns the existing trial ID.

Never copy only the database without its content-addressed evidence directory.

## Evidence corruption or missing artifact

Call `ledger.verify_evidence(<sha256>)`. A missing file, size mismatch, or hash mismatch is fatal. Restore the exact bytes from a trusted backup; do not regenerate a different file under the same name. Content addressing prevents replacement evidence from silently passing.

An empty/malformed universe file, fewer than 34 U3 or 62 U2 names, non-Malaysia country, non-`.KL` symbol, or broken subset relation blocks campaign creation.

An empty or insufficient historical panel blocks the backtester. It does not fall back to another market, current quotes, or synthetic performance.

## Budget stop

The state file records typed stop reasons:

- `MAX_TRIALS`
- `MAX_GENERATIONS`
- `FAMILY_TRIAL_BUDGET`
- `TOKEN_BUDGET`
- `EXTERNAL_COST_BUDGET`
- `WALL_TIME_BUDGET`
- `CONSECUTIVE_FAILURES`
- `QUEUE_EMPTY`

A stop is expected control flow. Continuous research means scheduling another reviewed finite batch, not removing the cap or creating `while true`.

## Research-gate failure

Mutation prompts receive enum codes only. They never receive holdout returns, holdout Sharpe, rank, or the numerical distance to a threshold. Full metrics remain evidence for audit and human diagnosis.

If a candidate fails U2 robustness, costs, turnover, drawdown, concentration, DSR, or PBO, create only the bounded child mutation produced by the queue. Do not switch between U3 and U2, lower costs, or optimize the threshold after seeing the result.

## Institutional audit rejection

`REGIME_INSTABILITY`, `FACTOR_RESIDUAL_FAILURE`, and `ALPHA_DECAY_FAILURE` are independent research failures. They may create one bounded child mutation, but cannot be overridden by an attractive raw Sharpe or by specialist prose. Audit evidence is content-addressed; verify its hash with the ledger before diagnosis.

Do not refit regime boundaries, swap the factor set, shorten the decay window, or loosen a threshold inside the same campaign. Such a change is a new methodology version and requires a new predeclared campaign, while the rejected attempt remains in the lifetime denominator.

## Scheduler or supervisor interruption

Each scheduled item is a finite campaign. Before starting the next item, verify that the previous campaign state contains a terminal stop reason and has no active trial. If it is still active, resume it instead of starting an overlapping campaign. Never replace the finite schedule with an unbounded loop.

## Dry-run review

Dry-run evidence exists only for research-gate-passing candidates. Verify:

- every target is in U3;
- each row is `SIMULATED_FILL` or contains a deterministic rejection reason;
- slippage, fee, and configuration hash are present;
- no broker/account/network field is present.

If any execution/broker import or submission method appears in `bounded_loop.py`, treat it as a security regression and block the change.

## Safe rollback

Source rollback must not delete ledger/evidence history. Revert source through normal Git operations, keep the campaign artifact directory immutable, and use a new campaign ID after the code/config hash changes. Existing outcomes remain part of the lifetime multiple-testing denominator.
