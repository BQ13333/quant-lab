# Findings and Knowledge Hygiene Completion

This runbook is the authoritative local implementation guide for registry and
knowledge-workflow steps 9 through 11. It extends, but does not replace, the
immutable run-control rules in `RUN_CONTROL_PLANE.md`.

## Custody boundary

The 853 historical directories in `vault/30_experiments` are frozen evidence.
The legacy `findings.sqlite3` is also frozen evidence and is not an operational
truth source. Historical operational classifications come only from snapshot
`findings-freeze-v2-ff19f1116e2d2fbab9ee` and its immutable
`findings-hygiene-v2` overlay.

The completion builder writes only to versioned, content-addressed sidecars
under the local knowledge-layer output root. It never writes a manifest into a
historical directory and never presents a migration projection as a replayable
control-plane manifest.

## Step 9: canonical findings registry

The derived registry must reconcile exactly:

- 853 historical runs and 2,557 evidence files;
- 847 migration lineages;
- 841 quarantined test/synthetic runs;
- six rejected runs with evidence-bound kill reasons;
- three superseded runs linked to their replacement in the same lineage;
- three parked runs with structured LSEG missing-data requirements;
- zero approval-required and zero accepted historical runs.

The action queue is a view derived from state. It can contain only
`ASEAN-20260704-232302`, `ASEAN-CONSTRAINED-OPT`, and
`MALAYSIA-20260705-082423`. Markdown prose cannot change registry state.

The canonical registry is an ongoing union, not a one-time migration dump.
The `experiments` table is the immutable 853-run historical projection;
`control_plane_runs` replays every later governed run and its append-only event
head; `control_plane_lineage_edges` gives every live run either an enforced
live parent or one of the three frozen legacy anchors; `operational_runs` is
their union. A new folder that is neither one of the
853 frozen IDs nor a replayable control-plane run makes the build fail closed.
Historical rows cannot be changed to `approval_required` or `approved` by any
standalone SQL update. A future approved live run must carry passing validation
and a verified signed-decision hash. Every planned live root must also carry one
replayable `data_admission_verified` event whose plan hash is the frozen
readiness-package hash. Validation and result-approval replay accepts hash-keyed
historical trust registries so key rotation does not erase old governed runs.
The root event archives the original signed entitlement, data manifest,
execution contract, human release and bundle bytes. Registry discovery
cryptographically replays them with the hash-matched historical admission and
release trust policies; a locally invented hash receipt is insufficient.

## Step 10: canonical research-artifact index

Every current Markdown file receives exactly one content-bound policy decision.
The historical checklist counted 10,142 files; the builder records that baseline
and separately records all later growth. New files make status fail until a new
content-addressed policy generation is built.

The default dense index includes current model cards, validation evidence, kill
evidence, canonical research memos, promotion minutes when they exist, and real
named-experiment evidence. It excludes quarantined tests, synthetic output,
superseded evidence, ZAI OCR intermediates, formula retries, duplicate
conversions, caches, temporary reports, and generated artifacts that have not
entered the registry. Exact duplicates become aliases and receive no duplicate
vectors. All index content is local-only (`cloud_allowed=false`).
The supported search API rejects every destination except `local` before it
opens the index. Search also verifies the current pointer, completion manifest,
source inventory, FTS/content bijection, vector dimensions/finiteness/content
hashes and package status; it will not serve a stale generation.

## Step 11: LSEG rerun readiness

Only the three parked lineages receive readiness packages. Each package binds:

- its immutable legacy anchor, overlay record, classification basis, and evidence;
- one reserved future root run ID and the legacy-to-governed lineage boundary;
- a falsifiable candidate, market scope, replication window, target and horizon;
- LSEG PIT membership, listing/delisting, corporate-action, tradability,
  liquidity, calendar, FX, publication/effective timestamp and vintage fields;
- frozen feature/split/validation source hashes and the exact 12 validation gates.

Readiness is not execution. Until a human-administered entitlement and data
admission bundle passes every offline activation check, `execution_authorized`
remains false and no LSEG, AWS, model, backtest, or workflow callback is invoked.
The activation API accepts file paths, never a caller-created receipt mapping.
It requires actual read-only, single-link artifacts under an administrator-owned
admission root; recomputes every size, SHA-256, partition and inventory root;
opens every partition as physical Parquet; reconciles Arrow schema, row count,
MIC, PIT timestamp extrema, every admitted market and the requested date window;
requires a signed weekday exchange-session calendar with at least 15 sessions
per full month; recomputes stable-ID, membership, corporate-action and timestamp
coverage from physical rows; measures price coverage against the exact
calendar-by-instrument-life denominator on disk; rejects mirrored coverage claims
that differ from measurement;
retains distinct revision/vintage rows by instrument-session-vintage identity
while counting each instrument session once for coverage, and rejects only an
exact duplicate vintage or impossible known-at ordering;
parses strict UTC timestamps and a full Git commit; verifies distinct
root-anchored Ed25519 authorities for entitlement/data/execution and human data
release; and cross-binds all components to the plan, legacy anchor and reserved
root run ID. A failed or crashed callback is durably failed and is never retried
automatically.

The execution contract is grounded in a checked-out Git commit/tree and exact
source inventory, including the complete control-plane implementation and
package import facades. Admission rejects any tracked modification or untracked
file across the entire repository, not only the named source subset. Its
validation policy and research configuration are separate immutable artifacts
whose hashes are frozen in the readiness plan.

After admission, `lseg_pit_adapter.py` opens only the verified local LSEG
partitions and exposes a sealed descriptor bounded to at most 65,536 rows per
batch; it never concatenates the lake in memory. Each partition receives one
full SHA-256 verification when the descriptor is opened. Repeated agent scans
then perform one Arrow read, guarded before and after by the verified file's
device, inode, size, mode, owner, link-count, modification-time and change-time
identity, rather than rereading every byte once for hashing and again for Arrow.
`lseg_rerun.py` rechecks that descriptor and creates exactly one
`created/not_run/ineligible` control-plane run with the reserved root ID. It has
no Yahoo or remote-model fallback. The callback result is replayed from the
control-plane event chain before activation is marked complete, and the complete
signed admission receipt is stored as its second immutable event. A later retry
replays the current manifest and event chain again; it never trusts only the
stored completion receipt.
The first governed rerun has `parent_run_id=null` plus a separate
`legacy_migration_anchor`; downstream governed rework uses normal parent-child
control-plane lineage.

## Commands

Use the quant-lab environment. The builder reuses the already provisioned local
RAG embedding runtime and cached BGE model when `fastembed` is not installed in
quant-lab itself:

```sh
export QUANT_LAB_WORKSPACE=/path/to/authorized-research-workspace
export QUANT_LAB_RAG_HOME=/path/to/local-rag-runtime
uv run python scripts/complete_knowledge_hygiene.py build
uv run python scripts/complete_knowledge_hygiene.py status
uv run python scripts/complete_knowledge_hygiene.py search \
  "Malaysia point in time survivor bias"
```

Completion requires the builder verification suite, the full repository tests,
and independent architecture, reliability, and governance scores of 5/5.

The search command uses a content-bound verification receipt after a successful
full build/status. Each query compares path, size, inode, modification time and
change time across the Markdown and implementation inventory and across every
packaged database/checksum artifact, and binds the exact configured trust-policy
hash sets. Any source, package or trust-context drift triggers full rehash
verification before search. `status` always performs the full audit.

## Data-admission trust provisioning

Production activation remains intentionally disabled until a human
administrator provisions both root-owned registries:

- `/Library/Application Support/QuantLab/data-admission-trust.sha256` for the
  entitlement verifier, admitted-data signer and execution-contract signer;
- `/Library/Application Support/QuantLab/data-release-trust.sha256` for the
  human data-release approver.

The referenced trust stores must live outside the agent-controlled vault, be
owned by the signed-in human account, and be read-only. Private Ed25519 keys must
not be stored in this repository, the experiment vault, an Obsidian note or a
Codex conversation. Provisioning these anchors authorizes only data availability
for one frozen rerun; it does not approve any research result.

Before real activation, the frozen execution sources must be committed in a
clean tracked checkout. The admission gate deliberately rejects any untracked or
modified repository file even when selected working-copy hashes match a readiness
plan.
