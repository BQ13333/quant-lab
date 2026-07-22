# Codex And Claude Co-Work Setup

## Decision

Use a clean, layered setup:

- Codex: primary implementation, data engineering, tests, and local verification.
- Claude Code: second implementation/review loop for large phases and independent
  critique.
- Supergoal: long-run planning wrapper for Codex and Claude Code.
- LangGraph: permanent Python workflow layer inside `quant_lab`.
- Automaton: sandbox-only research, not connected to this repo, IBKR, wallets,
  data lake credentials, or production workflows.

## Installed Skills

Supergoal is installed locally in:

```text
~/.codex/skills/supergoal
~/.claude/skills/supergoal
```

Restart Codex and Claude Code after install so both hosts load the skill.

## When To Use Supergoal

Use Supergoal for work that needs a multi-phase plan and one long autonomous run:

- Build a new data vendor adapter end-to-end.
- Implement a maths-engine module plus tests and docs.
- Refactor a large research workflow.
- Add an execution safety subsystem.
- Run a multi-agent research milestone with final audit.

Do not use Supergoal for tiny edits, quick bug fixes, or commands that can be
done in a normal Codex turn.

## Standard Supergoal Prompt Shape

```text
/supergoal Build <specific module> for quant_lab.

Constraints:
- Preserve deterministic Python engine behavior.
- Do not enable live trading.
- Add tests.
- Update docs/configs.
- Run relevant tests and final audit.
- Write artifacts under the existing repo structure.
```

After Supergoal prepares the run, paste the generated `/goal` command in the
same host. For true parallel execution, use a separate worktree or separate
thread; do not run two autonomous edits in the same working tree.

## Codex And Claude Division Of Labor

### Codex Primary

- Data adapters.
- Backtesting/research engine code.
- Validation and portfolio math.
- Test execution.
- Google Drive artifact packaging.
- IBKR/paper-trading safety layer.

### Claude Code Secondary

- Independent code review.
- Alternative implementation sketch.
- Documentation polish.
- Architecture critique.
- Verification of hidden assumptions.

## Automaton Boundary

Do not install or run Automaton inside this quant repo.

Reasons:

- It is designed for continuous autonomous operation.
- It includes self-modification and self-replication concepts.
- It can connect to wallets, infrastructure, domains, and real-world actions.
- That is too much autonomy for a financial research/trading codebase.

Allowed use:

- Separate sandbox directory.
- No broker credentials.
- No Google Drive data lake credentials.
- No wallet funding.
- No access to this project except read-only exported docs.

## Promotion Rule

No agent framework can promote a strategy. Promotion remains deterministic:

1. Data tier is serious-research or better.
2. Validation gates pass.
3. Multiple-testing correction passes.
4. Final holdout passes.
5. TCA/capacity checks pass.
6. Human approval is explicit.
