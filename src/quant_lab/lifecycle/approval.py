"""Lifecycle approvals reuse the externally anchored human signing authority."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
from .states import to_json
from quant_lab.research._control_plane_contracts import HumanApprovalRecord, ApprovalDecision
from quant_lab.research._control_plane_trust import ApprovalTrustStore

@dataclass(frozen=True)
class Authorization:
    record: HumanApprovalRecord
    trust_store: ApprovalTrustStore


def digest(value):
    return hashlib.sha256(to_json(value).encode()).hexdigest()


def require_human_approval(approval_id, authorization, state, evidence):
    if not isinstance(authorization, Authorization):
        raise ValueError('signed human approval required')
    record, store = authorization.record, authorization.trust_store
    if type(record) is not HumanApprovalRecord or type(store) is not ApprovalTrustStore:
        raise ValueError('invalid human approval authority')
    try:
        store.assert_external_and_unchanged(Path(__file__).resolve().parents[3])
        store.verify(record)
        if record.approval_id != approval_id or record.decision != ApprovalDecision.APPROVED or record.manifest_identity_hash != digest(state) or record.evidence_bundle_hash != digest(evidence):
            raise ValueError('approval does not bind this state and evidence')
    except Exception as exc:
        raise ValueError('human approval verification failed') from exc


def monitor_evidence(inputs, calendar, policy, reentry, retirement, ramp_plan):
    return dict(action='monitor',inputs=dict(inputs),calendar_hash=calendar.sha256(),policy=policy,
                reentry=reentry,retirement=retirement,stage_reports=ramp_plan.stage_reports if ramp_plan else (),first_ramp=ramp_plan.first_feasible if ramp_plan else policy.first_ramp)


def ramp_evidence(plan, evidence):
    return dict(action='advance_ramp',stages=plan.stages,first_feasible=plan.first_feasible,
                stage_reports=plan.stage_reports,required_rebalances=plan.required_rebalances,weeks=evidence.weeks,
                rebalances=evidence.rebalances,fills=evidence.fills,fill_days=evidence.fill_days,
                cost_test_passed=evidence.cost_test_passed)
