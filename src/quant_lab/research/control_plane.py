"""Stable public facade for the quant-research run control plane.

The implementation is separated into contracts/policy, trust/signature, and
storage/state-machine layers so each layer can be reviewed independently.
"""

from quant_lab.research._control_plane_contracts import (
    ApprovalDecision,
    ApprovalError,
    ConcurrentUpdateError,
    ControlPlaneError,
    EvidenceArtifactReference,
    GateStatus,
    HumanApprovalRecord,
    HumanSupersessionRecord,
    IllegalTransitionError,
    IntegrityError,
    PromotionStatus,
    REQUIRED_VALIDATION_GATES,
    RunKind,
    RunManifest,
    ValidationArtifact,
    ValidationEvidenceBundle,
    ValidationStatus,
    WORKFLOW_STAGES,
    WorkflowStatus,
)
from quant_lab.research._control_plane_store import ExperimentRunStore, _atomic_write
from quant_lab.research._control_plane_trust import (
    ApprovalTrustStore,
    AuthorizedHumanSigner,
    AuthorizedValidationRunner,
    ValidationRunnerTrustStore,
)

__all__ = [
    "ApprovalDecision",
    "ApprovalError",
    "ApprovalTrustStore",
    "AuthorizedHumanSigner",
    "AuthorizedValidationRunner",
    "ConcurrentUpdateError",
    "ControlPlaneError",
    "EvidenceArtifactReference",
    "ExperimentRunStore",
    "GateStatus",
    "HumanApprovalRecord",
    "HumanSupersessionRecord",
    "IllegalTransitionError",
    "IntegrityError",
    "PromotionStatus",
    "REQUIRED_VALIDATION_GATES",
    "RunKind",
    "RunManifest",
    "ValidationArtifact",
    "ValidationEvidenceBundle",
    "ValidationRunnerTrustStore",
    "ValidationStatus",
    "WorkflowStatus",
    "WORKFLOW_STAGES",
]
