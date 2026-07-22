# Local Research Baseline

Recorded on 2026-07-12 before the Malaysia local-Ollama research-loop implementation.

## Runtime

- Python package manager: `uv`
- Test runner: `pytest`
- Local inference runtime: Ollama 0.31.2
- Installed local models: `deepseek-r1:70b`, `qwen3.5:35b-mlx`, `qwen3-coder:30b`
- Repository state at dispatch: no Git repository (`Baseline ref: no-git`)

## Scope

- Market: Malaysia only
- Primary universe: predeclared Malaysia U3 liquid-proxy set
- Robustness universe: Malaysia U2 backtest-eligible set
- Japan work is explicitly out of scope and must remain unmodified.
- Live trading is out of scope; only deterministic dry-run execution is permitted.

## Test baseline

The Malaysia-scoped regression command excludes exactly two known, unrelated Japan
Wayback lifecycle tests that were already failing before implementation:

- `tests/test_japan_g4_halt_history.py::test_tse_wayback_lifecycle_clean_packet_is_admitted_and_revalidated`
- `tests/test_japan_g4_tse_wayback_lifecycle_validator.py::test_validator_reconciles_bytes_cdx_urls_and_event_sets`

Every other existing test and every new Malaysia/safety test must pass. The exact
passing count and command exit status are recorded in the Phase 1 transcript.
