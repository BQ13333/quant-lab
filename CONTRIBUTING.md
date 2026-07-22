# Contributing

Thank you for helping improve Quant Lab.

## Development setup

```bash
uv sync --extra agents
uv run pytest
```

Keep changes small and evidence-oriented. New research features should include
tests, document their data and timing assumptions, and state what would falsify
the result. Model-assisted output must not bypass deterministic validation or
human-approval boundaries.

## Data and source policy

Do not commit credentials, private model conversations, licensed market data,
commercial books, OCR-derived copyrighted text, or any source you do not have
permission to redistribute. Use synthetic fixtures for tests and document how
users can supply their own authorized data locally.

## Pull requests

Before opening a pull request:

1. Run the relevant tests and then the full default suite.
2. Update documentation for public interfaces or governance changes.
3. Preserve point-in-time, holdout, and fail-closed behavior.
4. Explain any new network, provider, data, or credential boundary.
5. Avoid performance claims that are not backed by reproducible evidence.
