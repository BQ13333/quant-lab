from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from quant_lab.agents.phd_math_specialists import PhDMathSpecialist, SpecialistPanel, load_specialist_panel, route_specialists
from quant_lab.research.contracts import ModelCallRecord, SpecialistReview, SpecialistReviewDraft
from quant_lab.research.providers import LocalModelRequest, LocalModelRouter, ModelCapability
from quant_lab.research.providers.base import LocalProviderError
from quant_lab.research.trial_ledger import TrialLedger


SELECTION_BIAS_ROLE = "selection_bias_validation_specialist"
_RESUMABLE_INFRASTRUCTURE_FAILURES = {
    "MODEL_AUTHENTICATION_FAILED",
    "MALFORMED_MODEL_OUTPUT",
    "MODEL_QUOTA_EXCEEDED",
    "MODEL_RATE_LIMITED",
    "MODEL_TIMED_OUT",
    "MODEL_UNAVAILABLE",
    "RETRY_EXHAUSTED",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_DOSSIER_KEYS = (
    "holdout",
    "terminal",
    "forward_outcome",
    "broker",
    "order_capability",
    "author",
    "provider",
    "model_identity",
)
_FORBIDDEN_REVIEW_PHRASES = ("terminal holdout", "broker credential", "live order")


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported dossier value: {type(value).__name__}")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _check_keys(value: Any, path: str = "research_context") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(blocked in normalized for blocked in _FORBIDDEN_DOSSIER_KEYS):
                raise ValueError(f"blinded dossier field is forbidden: {path}.{key}")
            _check_keys(item, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_keys(item, path)
    else:
        _canonical(value)


@dataclass(frozen=True)
class MalaysiaReviewDossier:
    candidate_id: str
    hypothesis_hash: str
    candidate_hash: str
    problem: str
    tags: tuple[str, ...]
    evidence_hashes: Mapping[str, str]
    research_context: Mapping[str, Any] = field(default_factory=dict)
    market: str = "Malaysia"
    exchange: str = "Bursa Malaysia"
    parent_candidate_id: str | None = None
    review_phase: str = "pre_backtest_design"

    def validate(self) -> None:
        if self.market != "Malaysia" or self.exchange != "Bursa Malaysia":
            raise ValueError("specialist runtime is locked to Malaysia / Bursa Malaysia")
        if self.review_phase not in {"pre_backtest_design", "post_backtest_research"}:
            raise ValueError("unsupported specialist review phase")
        if not self.candidate_id.strip() or not self.problem.strip() or len(self.tags) == 0:
            raise ValueError("candidate identity, problem, and routing tags are required")
        for name, value in {
            "hypothesis_hash": self.hypothesis_hash,
            "candidate_hash": self.candidate_hash,
            **dict(self.evidence_hashes),
        }.items():
            if not str(name).strip() or not _SHA256.fullmatch(value):
                raise ValueError("dossier provenance must contain named lowercase SHA-256 hashes")
        _check_keys(self.research_context)
        lowered = self.problem.lower()
        if "japan" in lowered or "jpx" in lowered:
            raise ValueError("non-Malaysia market content is forbidden in this pilot")
        if "broker" in lowered or "live order" in lowered:
            raise ValueError("broker and live-order content is forbidden in research review")

    def content_hash(self) -> str:
        self.validate()
        return _digest(json.dumps(_canonical(self.__dict__), sort_keys=True, separators=(",", ":")))

    def blinded_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "blind_candidate_id": _digest(self.candidate_id)[:20],
            "market": self.market,
            "exchange": self.exchange,
            "problem": self.problem,
            "tags": list(self.tags),
            "evidence_hashes": dict(self.evidence_hashes),
            "research_context": _canonical(self.research_context),
            "review_phase": self.review_phase,
        }


@dataclass(frozen=True)
class ReviewAttempt:
    specialist_id: str
    task_id: str
    debate_role: str
    status: str
    review: SpecialistReview | None = None
    model_call: ModelCallRecord | None = None
    failure_code: str | None = None


@dataclass(frozen=True)
class PanelDisposition:
    candidate_id: str
    dossier_hash: str
    selected_specialists: tuple[str, ...]
    attempts: tuple[ReviewAttempt, ...]
    backtest_allowed: bool
    progression_allowed: bool
    vetoed: bool
    reasons: tuple[str, ...]
    conflicts: tuple[str, ...]
    disposition_hash: str


class ReviewContractViolation(LocalProviderError):
    code = "REVIEW_CONTRACT_VIOLATION"


class SpecialistReviewRuntime:
    """Deterministic, blinded, Malaysia-only mathematical review controller."""

    def __init__(
        self,
        router: LocalModelRouter,
        *,
        ledger: TrialLedger | None = None,
        panel: SpecialistPanel | None = None,
        retry_infrastructure_failures: bool = False,
    ) -> None:
        self.router = router
        self.ledger = ledger
        self.panel = panel or load_specialist_panel()
        self.retry_infrastructure_failures = retry_infrastructure_failures

    def assign(self, dossier: MalaysiaReviewDossier) -> tuple[PhDMathSpecialist, ...]:
        dossier.validate()
        routed = route_specialists(dossier.problem, dossier.tags, limit=len(self.panel.specialists), panel=self.panel)
        selection = self.panel.by_id(SELECTION_BIAS_ROLE)
        relevant = [item for item in routed if item.id != SELECTION_BIAS_ROLE]
        selected = (selection, *relevant[:2])
        if len({item.id for item in selected}) != 3:
            raise RuntimeError("three independent specialist perspectives could not be assigned")
        return selected

    def review_candidate(
        self,
        dossier: MalaysiaReviewDossier,
        *,
        trial_id: str | None = None,
    ) -> PanelDisposition:
        return self.review_roles(dossier, self.assign(dossier), trial_id=trial_id, require_progression_panel=True)

    def review_roles(
        self,
        dossier: MalaysiaReviewDossier,
        specialists: Sequence[PhDMathSpecialist | str],
        *,
        trial_id: str | None = None,
        require_progression_panel: bool = False,
    ) -> PanelDisposition:
        dossier.validate()
        if self.ledger is not None and trial_id is None:
            raise ValueError("trial_id is required when specialist reviews are recorded")
        resolved = tuple(self.panel.by_id(item) if isinstance(item, str) else item for item in specialists)
        if not resolved or len({item.id for item in resolved}) != len(resolved):
            raise ValueError("specialist selection must be non-empty and independent")
        attempts = tuple(
            self._run_one(dossier, specialist, index, trial_id=trial_id)
            for index, specialist in enumerate(resolved)
        )
        disposition = self._dispose(dossier, attempts, require_progression_panel=require_progression_panel)
        if self.ledger is not None and trial_id is not None:
            self.ledger.record_panel_disposition(
                trial_id,
                {
                    "candidate_id": dossier.candidate_id,
                    "selected_specialists": list(disposition.selected_specialists),
                    "backtest_allowed": disposition.backtest_allowed,
                    "progression_allowed": disposition.progression_allowed,
                    "vetoed": disposition.vetoed,
                    "reasons": list(disposition.reasons),
                    "conflicts": list(disposition.conflicts),
                    "disposition_hash": disposition.disposition_hash,
                },
            )
        return disposition

    def _run_one(
        self,
        dossier: MalaysiaReviewDossier,
        specialist: PhDMathSpecialist,
        index: int,
        *,
        trial_id: str | None,
    ) -> ReviewAttempt:
        dossier_hash = dossier.content_hash()
        task_id = f"MYS-TASK-{_digest(dossier_hash + specialist.id)[:20]}"
        debate_role = ("selection_bias_guard", "independent_reviewer", "adversarial_reviewer")[min(index, 2)]
        recorded = self._recorded_attempt(
            trial_id=trial_id,
            task_id=task_id,
            specialist_id=specialist.id,
            debate_role=debate_role,
        )
        if recorded is not None:
            return recorded
        capability = (
            ModelCapability.ADVERSARIAL_MATH
            if debate_role == "adversarial_reviewer"
            else ModelCapability.ROUTINE_REVIEW
        )
        system_prompt = self._system_prompt(specialist, debate_role, dossier.review_phase)
        payload = {
            **dossier.blinded_payload(),
            "task_id": task_id,
            "specialist_id": specialist.id,
            "specialist_lane": specialist.lane,
            "specialist_mandate": specialist.mandate,
            "routing_tags": list(specialist.routing_tags),
            "debate_role": debate_role,
            "proof_obligations": list(specialist.proof_obligations),
            "kill_criteria": list(specialist.kill_criteria),
            "corpus_anchors": list(specialist.corpus_anchors),
            "field_contract": {
                "verdict": "one of pass, conditional, fail, abstain",
                "claims": "JSON array of plain strings only",
                "assumptions": "non-empty JSON array of plain strings only",
                "counterexamples": "JSON array of plain strings only; use [] when none",
                "falsification_tests": "non-empty JSON array of plain strings only",
                "veto": "JSON boolean; true only with a concrete string counterexample",
                "confidence": "JSON number from 0 through 1",
            },
        }
        request = LocalModelRequest(
            request_id=f"MYS-CALL-{_digest(task_id)[:20]}",
            capability=capability,
            expected_contract=SpecialistReviewDraft,
            system_prompt=system_prompt,
            payload=payload,
            max_tokens=1024,
            max_prompt_chars=64_000,
            timeout_seconds=240,
            max_attempts=1,
            temperature=0.0,
            seed=42,
            privacy_class="local_only",
            config={"market_scope": "malaysia", "review_schema": "specialist_review_draft_v1"},
        )
        model = self.router.policy.models[capability]
        try:
            result = self.router.route(request)
            if not isinstance(result.contract, SpecialistReviewDraft):
                raise ReviewContractViolation("provider returned the wrong contract type")
            draft = result.contract
            review = SpecialistReview(
                review_id=f"MYS-REVIEW-{_digest(task_id + specialist.id + draft.content_hash())[:20]}",
                task_id=task_id,
                specialist_id=specialist.id,
                verdict=draft.verdict,
                claims=draft.claims,
                assumptions=draft.assumptions,
                counterexamples=draft.counterexamples,
                falsification_tests=draft.falsification_tests,
                veto=draft.veto,
                confidence=draft.confidence,
                evidence=(),
                provenance_hashes={"candidate": dossier.candidate_hash},
            )
            self._validate_review(review, task_id=task_id, specialist_id=specialist.id)
            if self.ledger is not None and trial_id is not None:
                self.ledger.record_specialist_review(trial_id, review, result.call_record)
            return ReviewAttempt(specialist.id, task_id, debate_role, "success", review, result.call_record)
        except (LocalProviderError, TypeError, ValueError) as exc:
            code = getattr(exc, "code", "REVIEW_CONTRACT_VIOLATION")
            failure_record = getattr(exc, "call_record", None)
            if not isinstance(failure_record, ModelCallRecord):
                failure_record = None
            if self.ledger is not None and trial_id is not None:
                self.ledger.record_specialist_failure(
                    trial_id,
                    task_id=task_id,
                    specialist_id=specialist.id,
                    failure_code=code,
                    request_fingerprint=request.fingerprint(model),
                    record=failure_record,
                )
            return ReviewAttempt(
                specialist.id,
                task_id,
                debate_role,
                "failure",
                model_call=failure_record,
                failure_code=code,
            )

    def _recorded_attempt(
        self,
        *,
        trial_id: str | None,
        task_id: str,
        specialist_id: str,
        debate_role: str,
    ) -> ReviewAttempt | None:
        if self.ledger is None or trial_id is None:
            return None
        events = self.ledger.replay_manifest(trial_id)["events"]
        for event in reversed(events):
            if event["event_type"] != "specialist_review":
                continue
            payload = json.loads(event["payload_json"])
            if payload.get("task_id") != task_id or payload.get("specialist_id") != specialist_id:
                continue
            if event["status"] == "failure":
                failure_code = str(event["failure_class"] or "REVIEW_CONTRACT_VIOLATION")
                if self.retry_infrastructure_failures and failure_code in _RESUMABLE_INFRASTRUCTURE_FAILURES:
                    continue
                return ReviewAttempt(
                    specialist_id,
                    task_id,
                    debate_role,
                    "failure",
                    failure_code=failure_code,
                )
            evidence_sha = event.get("evidence_sha256")
            if not evidence_sha:
                raise RuntimeError("recorded specialist review is missing immutable evidence")
            bundle = json.loads(self.ledger.verify_evidence(evidence_sha).path.read_text(encoding="utf-8"))
            review = SpecialistReview.from_dict(bundle["review"])
            record = ModelCallRecord.from_dict(bundle["model_call"])
            self._validate_review(review, task_id=task_id, specialist_id=specialist_id)
            replay_record = replace(
                record,
                status="replay",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                stop_reason="ledger_replay",
            )
            replay_record.validate()
            return ReviewAttempt(
                specialist_id,
                task_id,
                debate_role,
                "success",
                review,
                replay_record,
            )
        return None

    @staticmethod
    def _system_prompt(specialist: PhDMathSpecialist, debate_role: str, review_phase: str) -> str:
        phase_instruction = (
            "This is a pre-backtest design review. A pass means no design-level blocker and authorizes the "
            "predeclared deterministic tests; it never claims alpha or performance is proven. Do not use a "
            "conditional verdict merely because future test results are not supplied."
            if review_phase == "pre_backtest_design"
            else "This is a post-backtest research interpretation review. Supplied metrics are contaminated "
            "research-period evidence, never terminal proof. Identify reproducible defects, alternative "
            "explanations, and predeclared next tests; do not tune parameters or promote the candidate."
        )
        return f"""You are the {specialist.title} in a QForge-20 Malaysia quantitative research panel.
Your debate role is {debate_role}. Review only the blinded Bursa Malaysia research dossier supplied.
Return exactly one SpecialistReviewDraft JSON object matching the supplied schema and no extra prose.
The controller adds review identity and provenance after validation. Follow field_contract literally:
claims, assumptions, counterexamples, and falsification_tests contain plain strings, never objects.
Explicitly state assumptions and falsification tests. Set veto=true only for a concrete reproducible
counterexample and then verdict must be fail or conditional. You receive research evidence only.
Do not discuss restricted evaluation data, identities, credentials, or execution capabilities.
{phase_instruction} Put required future checks
in falsification_tests. Use conditional or fail only for a concrete defect already present in the dossier.
"""

    @staticmethod
    def _validate_review(review: SpecialistReview, *, task_id: str, specialist_id: str) -> None:
        review.validate()
        if review.task_id != task_id or review.specialist_id != specialist_id:
            raise ReviewContractViolation("review identity does not match its assigned blinded task")
        text = review.canonical_json().lower()
        if any(phrase in text for phrase in _FORBIDDEN_REVIEW_PHRASES):
            raise ReviewContractViolation("review leaked or requested a restricted capability")

    @staticmethod
    def _dispose(
        dossier: MalaysiaReviewDossier,
        attempts: tuple[ReviewAttempt, ...],
        *,
        require_progression_panel: bool,
    ) -> PanelDisposition:
        selected = tuple(item.specialist_id for item in attempts)
        failures = [item for item in attempts if item.status != "success" or item.review is None]
        reviews = [item.review for item in attempts if item.review is not None]
        vetoed = any(review.veto for review in reviews)
        reasons: list[str] = []
        if failures:
            reasons.append("failed_review_attempt")
        if require_progression_panel and (len(selected) < 3 or SELECTION_BIAS_ROLE not in selected):
            reasons.append("incomplete_mandatory_panel")
        if vetoed:
            reasons.append("concrete_counterexample_veto")
        if any(review.verdict in {"fail", "abstain"} for review in reviews):
            reasons.append("non_passing_review")
        if any(review.verdict == "conditional" for review in reviews):
            reasons.append("conditional_review")
        verdicts = tuple(review.verdict for review in reviews)
        conflicts = ("verdict_disagreement",) if len(set(verdicts)) > 1 else ()
        mandatory_complete = not (
            require_progression_panel and (len(selected) < 3 or SELECTION_BIAS_ROLE not in selected)
        )
        backtest_allowed = (
            not failures
            and mandatory_complete
            and not vetoed
            and len(reviews) == len(attempts)
            and all(review.verdict in {"pass", "conditional"} for review in reviews)
        )
        progression_allowed = backtest_allowed and all(review.verdict == "pass" for review in reviews)
        stable = {
            "candidate_id": dossier.candidate_id,
            "dossier_hash": dossier.content_hash(),
            "selected": selected,
            "reviews": [review.content_hash() for review in reviews],
            "failures": [item.failure_code for item in failures],
            "backtest_allowed": backtest_allowed,
            "progression_allowed": progression_allowed,
            "vetoed": vetoed,
            "reasons": reasons,
            "conflicts": conflicts,
        }
        disposition_hash = _digest(json.dumps(stable, sort_keys=True, separators=(",", ":")))
        return PanelDisposition(
            dossier.candidate_id,
            dossier.content_hash(),
            selected,
            attempts,
            backtest_allowed,
            progression_allowed,
            vetoed,
            tuple(reasons),
            conflicts,
            disposition_hash,
        )
