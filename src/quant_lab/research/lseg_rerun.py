"""Executable, local-data-only entrypoint for an admitted LSEG rerun."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from quant_lab.data.lseg_pit_adapter import open_admitted_lseg_dataset
from quant_lab.research.control_plane import ExperimentRunStore, RunKind, RunManifest
from quant_lab.research.data_admission import (
    VerifiedDataAdmission,
    archive_verified_data_admission,
)


def _create_admitted_lseg_run(
    plan: Mapping[str, Any],
    admission: VerifiedDataAdmission,
    *,
    workspace_root: Path,
    admission_root: Path,
    admission_bundle_path: Path,
) -> dict[str, Any]:
    """Bind the already verified admission to the run and expose a batch scanner."""
    candidate_input = open_admitted_lseg_dataset(
        admission_root=admission_root,
        admission_bundle_path=admission_bundle_path,
        verified=admission,
    )
    if candidate_input.row_count < 1:
        raise ValueError("verified LSEG admission contains no partitions")
    manifest = RunManifest.new(
        run_id=admission.root_run_id,
        parent_run_id=None,
        run_kind=RunKind.RESEARCH,
        hypothesis=plan["research_contract"]["hypothesis"],
        universe=json.dumps(
            plan["research_contract"]["universe"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        data_manifest_id=admission.data_manifest_id,
        data_asof_utc=admission.data_asof_utc,
        feature_manifest_hash=admission.feature_manifest_hash,
        code_commit=admission.code_commit,
        target=plan["research_contract"]["target"],
        label_horizon=plan["research_contract"]["label_horizon"],
        split_manifest_id=admission.split_manifest_id,
    )
    vault = workspace_root.resolve() / "quant_lab" / "vault"
    store = ExperimentRunStore(vault, admission.root_run_id)
    created = store.create(manifest)
    store._record_archived_data_admission(
        archive_verified_data_admission(
            admission_root=admission_root,
            bundle_path=admission_bundle_path,
            verified=admission,
        ),
        expected_manifest_hash=created.content_hash(),
    )
    replay = store.verify_integrity()
    return {
        "run_id": created.run_id,
        "manifest_path": store.manifest_path.relative_to(workspace_root.resolve()).as_posix(),
        "manifest_sha256": replay["manifest_hash"],
        "manifest_identity_sha256": replay["manifest_identity_hash"],
        "event_count": replay["event_count"],
        "workflow_status": created.workflow_status.value,
        "validation_status": created.validation_status.value,
        "promotion_status": created.promotion_status.value,
        "dataset_descriptor": candidate_input.descriptor(),
    }


__all__: list[str] = []
