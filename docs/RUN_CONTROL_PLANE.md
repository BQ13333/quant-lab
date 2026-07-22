# Quant research run control: steps 3–8

This control plane governs future experiments. It does not rewrite the frozen 853-folder historical evidence set or the findings-hygiene migration overlay.

The stable public API is `quant_lab.research.control_plane`. Its implementation is split into `_control_plane_contracts` (schemas and policy contracts), `_control_plane_trust` (runner/human trust and signatures), and `_control_plane_store` (atomic storage, replay, and state transitions). Markdown notes are supporting displays only; they cannot change a run's identity, validation, promotion, or approval state.

## 1. Create one root run

The Research Director/controller must supply the root `run_id`. The library never generates one. Research runs also require explicit point-in-time data, feature, code, target, label, and split identities.

```python
from quant_lab.agents.workflow import default_research_state

state = default_research_state(
    "Lagged medium-horizon momentum persists after Malaysia trading costs.",
    "Malaysia point-in-time ordinary equities",
    run_id="MYS-MOMENTUM-20260720-001",
    parent_run_id=None,
    vault_root="/absolute/path/to/quant_lab/vault",
    run_kind="research",
    data_manifest_id="LSEG-MYS-PIT-20260719",
    data_asof_utc="2026-07-19T00:00:00+00:00",
    feature_manifest_hash="<64-character sha256>",
    code_commit="<Git commit hash>",
    target="next_21d_excess_return",
    label_horizon="21_trading_days",
    split_manifest_id="MYS-PURGED-CV-001",
)
```

Before starting the machine workflow, preregister the root-approved validation policy:

```python
from quant_lab.research.control_plane import (
    ExperimentRunStore,
    ValidationRunnerTrustStore,
)

runner_policy = ValidationRunnerTrustStore.from_file(
    "/external/read-only/validation-runners-and-policy.json"
)
ExperimentRunStore(state["vault_root"], state["run_id"]).register_validation_policy(
    runner_policy,
    expected_manifest_hash=state["manifest_hash"],
)
```

Registration is allowed only while the run is still `created/not_run/ineligible`. The immutable policy snapshot fixes the gate thresholds and calculator ID/version/source-hash allowlists before modelling begins. A later policy—even another root-authorized policy—cannot replace it for that run.

All downstream stages reuse that `run_id`. Per-task or validator execution IDs are child identifiers; they never replace the root run ID. A rework or revalidation must be a new child run whose `parent_run_id` names an existing run of the same kind.

The authoritative projection is:

```text
<vault>/30_experiments/<run_id>/run_manifest.json
```

Immutable manifest versions and hash-chained events are under `.control/`. Supporting notes use deterministic paths under `notes/`.

## 2. Independent state machines

| State machine | Legal progression |
|---|---|
| Workflow | `created → running → complete` or `created → running → failed` |
| Validation | `not_run → pass`, `fail`, or `inconclusive` |
| Promotion | `ineligible → parked_for_data`, `rejected`, or `approval_required`; `approval_required → approved` or `rejected`; `approved → superseded` |

Important consequences:

- Completing the eight machine stages does not validate a candidate.
- Passing validation requests human approval; it does not approve the candidate.
- Missing LSEG entitlement, unavailable PIT history, incomplete timestamp lineage, or inconclusive PIT/lineage evidence produces `parked_for_data`.
- Failed deterministic evidence produces `rejected`.
- `test`, `synthetic`, and `smoke` runs remain non-promotable even if their sample evidence passes.
- Parked, rejected, failed, and already validated runs are not edited backwards. Start an explicitly linked child run.

## 3. Test and synthetic isolation

Automated tests must use `tmp_path` or another temporary root and set `run_kind="test"` or `run_kind="smoke"`. Synthetic demonstrations use `run_kind="synthetic"`.

The controller rejects non-research writes to the repository's production `vault`, path traversal, symlink escapes, placeholder identities, temporary/test tokens, eight-character hexadecimal IDs, and UUID-like ambiguous IDs. It also rejects attempts to inject pre-completed or pre-approved initial manifests.

## 4. Deterministic validation evidence

Validation admission requires exactly 12 immutable JSON artifacts:

1. `point_in_time_data_audit`
2. `purged_embargoed_walk_forward_splits`
3. `cpcv_pbo`
4. `dsr_psr`
5. `multiple_testing_fdr`
6. `cost_sensitivity`
7. `drawdown_turnover`
8. `liquidity_capacity`
9. `regime_stability`
10. `sealed_holdout`
11. `data_feature_split_lineage`
12. `code_reproducibility`

Every artifact must:

- be produced by deterministic Python running under an allowlisted external validation-runner key;
- contain the gate-specific metric schema enforced by the controller;
- bind the root run, candidate ID and content hash, data manifest, data cutoff, feature manifest, code commit, split manifest, calculator version/source hash, and exact input hashes;
- carry a detached Ed25519 runner attestation verified against a read-only policy whose SHA-256 is held in a root-owned system anchor;
- use a calculator and thresholds present in the run's preregistered root-approved validation policy;
- be a unique regular file inside the run root;
- have a recomputed SHA-256 and byte size in the admitted bundle;
- be generated after the run manifest and before admission.

The sealed-holdout artifact also binds the holdout manifest and one-time release ledger hashes. The controller derives the validation result from the 12 attested artifact statuses. Models—including Cerebras, Kimi, DeepSeek, Codex, and the mathematics specialists—may interpret or challenge these artifacts, but their prose cannot set a gate status or generate an accepted runner attestation.

## 5. Human approval

The codebase contains verification only; it contains no private key and no approval-signing function.

An authorised human administrator must provision an Ed25519 public-key allowlist outside the agent-controlled vault and make it read-only. Its SHA-256 must be listed by an administrator in the root-owned, non-group-writable registry at `/Library/Application Support/QuantLab/approval-trust.sha256`. If that registry is absent, approval remains disabled. The controller rechecks the policy file, root registry, ownership, mode, path, and content before every decision and replay.

The independent deterministic-runner and validation policy uses the same pattern with `/Library/Application Support/QuantLab/validation-runner-trust.sha256`. Each registry accepts one SHA-256 per line so a root administrator can retain historical policy hashes during key rotation and old decisions remain replayable. Neither root registry is created automatically by Codex.

The detached signature binds:

- approval ID and root run ID;
- immutable manifest identity hash;
- exact candidate and admitted evidence-bundle hash;
- `approved`, `rejected`, or `rework` decision;
- approver identity, approval time, key ID, and acknowledged limitations.

Only a verified `approved` decision changes `promotion_status` to `approved`. `rejected` and `rework` both close the current run as `rejected`; rework proceeds in a child run. Supersession requires a second signed human record and an existing replacement run whose `parent_run_id` points to the approved run. Approval and supersession events preserve the exact trust-policy hash used at decision time.

## 6. Recovery and verification

Each writer holds a per-run file lock. Immutable manifest versions and an event are fsynced before the current manifest projection is atomically replaced. If a process stops after the event but before the projection update, the next load reconstructs the current manifest from the event head and matching immutable version.

Exclusive immutable files are first fsynced under a temporary name and atomically linked into place, so an interruption cannot expose partial final content. A manifest version written before its event is preserved under `.control/orphans/manifests/` and excluded from authoritative replay before the operation is retried.

Use `ExperimentRunStore.verify_integrity()` for an unvalidated run. Once validation exists, pass the root-anchored `ValidationRunnerTrustStore`. Replay derives validation and promotion outcomes again from the signed gate statuses and requires the result to equal both the admission event and its manifest version.

For a run containing human decisions, also pass the root-anchored `ApprovalTrustStore`. If approval and later supersession used different retained policies, pass a mapping from each historical policy SHA-256 to its corresponding `ApprovalTrustStore`. Replay selects the exact policy recorded by each event, re-verifies every signature and binding, and repeats the approval-after-admission and supersession-after-approval time checks.

No current run is implicitly approved by this implementation. The local machine has not been provisioned with either root-owned trust registry by this code, so policy registration, validation and approval remain fail-closed until the human administrator completes that separate setup. The first LSEG research run should begin at `created/not_run/ineligible` and advance only as real point-in-time data and independently attested deterministic artifacts become available.
