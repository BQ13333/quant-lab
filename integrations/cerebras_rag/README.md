# Cerebras RAG reference integration

This integration provides an allowlist-first retrieval pipeline for an
authorized Markdown research library. Retrieval and embeddings stay local;
Cerebras is used only for explicitly eligible labeling, reranking, synthesis,
and grounding verification.

## Capabilities

- read-only Markdown discovery constrained to a configured source root;
- separate local-search and cloud-eligibility glob policies;
- deny terms that override every allowlist;
- local BGE embeddings, SQLite FTS5, vector similarity, and rank fusion;
- deterministic document/chunk identifiers and atomic index replacement;
- structured Cerebras labels and evidence reranking;
- cited answer synthesis with a separate grounding verifier;
- a local findings and run ledger; and
- macOS Keychain credential loading—no API key in the repository or `.env`.

The code does not include a corpus, index, model cache, API key, or prior model
requests and responses.

## Setup

The `ask` command's current `langchain-cerebras` adapter requires Python 3.11
or 3.12. Local audit/build/search and the direct SDK commands do not depend on
that adapter.

```bash
uv sync --extra cerebras-rag
cp integrations/cerebras_rag/config.example.json \
  integrations/cerebras_rag/config.json
```

Edit `config.json` so `source_root` points to Markdown you have the right to
process and transmit. Treat `cloud_allowed_globs` as a narrow allowlist. A deny
term always forces a document to remain local-only.

Store the API key in macOS Keychain using the account and service names from
the configuration:

```bash
security add-generic-password \
  -a quant-lab-cerebras \
  -s ai.cerebras.api-key \
  -w 'YOUR_CEREBRAS_API_KEY'
```

## Commands

```bash
uv run --extra cerebras-rag python integrations/cerebras_rag/rag.py audit
uv run --extra cerebras-rag python integrations/cerebras_rag/rag.py build
uv run --extra cerebras-rag python integrations/cerebras_rag/rag.py label
uv run --extra cerebras-rag python integrations/cerebras_rag/rag.py search \
  "selection bias and backtest overfitting"
uv run --extra cerebras-rag python integrations/cerebras_rag/rag.py ask \
  "Which specialists should review a backtest-overfitting claim?"
```

Run `audit` before `build`, `label`, or `ask` and inspect which files are
cloud-eligible. The integration is a research reference implementation, not a
claim that its retrieval quality is state of the art.
