from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from quant_lab.validation.metrics import sharpe_ratio


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FrozenCandidateBundle:
    candidate_id: str
    candidate_hash: str
    code_hash: str
    data_hash: str
    config_hash: str
    research_end: str
    frozen_at: str

    def validate(self) -> None:
        if not self.candidate_id.strip() or not self.research_end.strip() or not self.frozen_at.strip():
            raise ValueError("frozen candidate identity and timestamps are required")
        for value in (self.candidate_hash, self.code_hash, self.data_hash, self.config_hash):
            if not _SHA256.fullmatch(value):
                raise ValueError("frozen candidate hashes must be lowercase SHA-256")


@dataclass(frozen=True)
class TerminalPromotionEvidence:
    candidate_id: str
    passed: bool
    reason_codes: tuple[str, ...]
    observation_count: int
    evidence_hash: str
    evaluated_at: str


class TerminalHoldoutService:
    """Forward-only terminal evaluator that emits no scores or mutation payload."""

    def __init__(
        self,
        *,
        contaminated_through: str | pd.Timestamp = "2026-07-06",
        min_forward_observations: int = 63,
        min_forward_sharpe: float = 0.0,
    ) -> None:
        self.contaminated_through = pd.Timestamp(contaminated_through)
        self.min_forward_observations = min_forward_observations
        self.min_forward_sharpe = min_forward_sharpe

    def evaluate(
        self,
        bundle: FrozenCandidateBundle,
        *,
        forward_returns: pd.Series,
    ) -> TerminalPromotionEvidence:
        bundle.validate()
        if not isinstance(forward_returns.index, pd.DatetimeIndex):
            raise TypeError("forward holdout requires a DatetimeIndex")
        if pd.Timestamp(bundle.research_end) > self.contaminated_through:
            raise ValueError("frozen research_end exceeds the declared contaminated boundary")
        forward = forward_returns.loc[forward_returns.index > self.contaminated_through].dropna()
        digest = hashlib.sha256()
        digest.update(bundle.candidate_hash.encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(forward, index=True).values.tobytes())
        if len(forward) < self.min_forward_observations:
            passed = False
            reasons = ("INSUFFICIENT_FORWARD_EVIDENCE",)
        else:
            passed = sharpe_ratio(forward) > self.min_forward_sharpe
            reasons = ("FORWARD_GATE_PASSED",) if passed else ("FORWARD_GATE_FAILED",)
        return TerminalPromotionEvidence(
            candidate_id=bundle.candidate_id,
            passed=passed,
            reason_codes=reasons,
            observation_count=len(forward),
            evidence_hash=digest.hexdigest(),
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )
