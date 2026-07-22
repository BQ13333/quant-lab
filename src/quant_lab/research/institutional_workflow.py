from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from quant_lab.research.bounded_loop import (
    ALLOWED_FEATURES,
    BacktestEvidence,
    CampaignConfig,
    InstitutionalAuditEvidence,
    MalaysiaUniverseContract,
    MutationDirective,
)
from quant_lab.research.contracts import CandidateSpec
from quant_lab.validation.metrics import sharpe_ratio
from quant_lab.validation.overfit import sharpe_decay_analysis


ALLOWED_FAMILIES = {
    "momentum",
    "mean_reversion",
    "low_volatility",
    "volatility_adjusted_momentum",
    "lagged_liquidity",
    "time_series_momentum",
    "ewmac",
    "breakout",
    "channel_position",
    "residual_momentum",
    "bollinger_reversion",
}
_TOKEN = re.compile(r"[a-z0-9]+")
_FORBIDDEN_SCOPE = ("japan", "jpx", "openrouter", "broker", "order", "holdout result")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _normalized_terms(value: str) -> tuple[str, ...]:
    return tuple(sorted(set(_TOKEN.findall(value.lower()))))


@dataclass(frozen=True)
class HypothesisTicket:
    hypothesis_id: str
    economic_claim: str
    family: str
    features: tuple[str, ...]
    horizon_days: int
    data_requirements: tuple[str, ...]
    universe: str = "MALAYSIA-U3-U2"
    parent_hypothesis_id: str | None = None
    mutation_codes: tuple[str, ...] = ()

    def validate(self) -> None:
        if not re.fullmatch(r"MYS-H-[A-Z0-9_-]{3,80}", self.hypothesis_id):
            raise ValueError("hypothesis_id must be a Malaysia hypothesis identifier")
        if len(self.economic_claim.strip()) < 20:
            raise ValueError("economic claim must be explicit and falsifiable")
        if self.family not in ALLOWED_FAMILIES:
            raise ValueError("hypothesis family is outside the Malaysia allowlist")
        if not self.features or not set(self.features).issubset(ALLOWED_FEATURES):
            raise ValueError("hypothesis features are outside the frozen Malaysia set")
        if not 1 <= self.horizon_days <= 252:
            raise ValueError("hypothesis horizon must be between one and 252 trading days")
        if self.universe != "MALAYSIA-U3-U2":
            raise ValueError("hypothesis escaped the Malaysia U3/U2 universe")
        if not self.data_requirements:
            raise ValueError("hypothesis data requirements are required")
        scope = _canonical(asdict(self)).lower()
        if any(term in scope for term in _FORBIDDEN_SCOPE):
            raise ValueError("hypothesis contains a forbidden market or execution scope")
        if self.parent_hypothesis_id is None and self.mutation_codes:
            raise ValueError("root hypotheses cannot contain mutation feedback")
        if self.parent_hypothesis_id is not None and not self.mutation_codes:
            raise ValueError("child hypotheses require explicit blinded mutation codes")

    def exact_hash(self) -> str:
        payload = asdict(self)
        payload.pop("hypothesis_id")
        return _digest(payload)

    def semantic_fingerprint(self) -> str:
        return _digest(
            {
                "claim_terms": _normalized_terms(self.economic_claim),
                "family": self.family,
                "features": sorted(set(self.features)),
                "horizon_days": self.horizon_days,
                "data_requirements": sorted(_normalized_terms(" ".join(self.data_requirements))),
                "universe": self.universe,
            }
        )


@dataclass(frozen=True)
class HypothesisAdmission:
    hypothesis_id: str
    accepted: bool
    created: bool
    reason_code: str
    exact_hash: str
    semantic_fingerprint: str


class HypothesisIntakeRegistry:
    """Persistent, deterministic intake and duplicate firewall for Malaysia ideas."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path, timeout=30.0) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS hypothesis_intake (
                    hypothesis_id TEXT PRIMARY KEY,
                    exact_hash TEXT NOT NULL UNIQUE,
                    semantic_fingerprint TEXT NOT NULL,
                    ticket_json TEXT NOT NULL,
                    parent_hypothesis_id TEXT REFERENCES hypothesis_intake(hypothesis_id),
                    admitted_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_hypothesis_semantic
                    ON hypothesis_intake(semantic_fingerprint);
                """
            )

    def admit(self, ticket: HypothesisTicket) -> HypothesisAdmission:
        ticket.validate()
        exact = ticket.exact_hash()
        semantic = ticket.semantic_fingerprint()
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            same_id = connection.execute(
                "SELECT exact_hash, semantic_fingerprint FROM hypothesis_intake WHERE hypothesis_id = ?",
                (ticket.hypothesis_id,),
            ).fetchone()
            if same_id is not None:
                if same_id["exact_hash"] != exact:
                    raise RuntimeError("hypothesis identity conflicts with its admitted definition")
                connection.commit()
                return HypothesisAdmission(ticket.hypothesis_id, True, False, "ALREADY_ADMITTED", exact, semantic)
            exact_row = connection.execute(
                "SELECT hypothesis_id FROM hypothesis_intake WHERE exact_hash = ?", (exact,)
            ).fetchone()
            if exact_row is not None:
                connection.commit()
                return HypothesisAdmission(ticket.hypothesis_id, False, False, "EXACT_DUPLICATE", exact, semantic)
            semantic_row = connection.execute(
                "SELECT hypothesis_id FROM hypothesis_intake WHERE semantic_fingerprint = ? ORDER BY admitted_at LIMIT 1",
                (semantic,),
            ).fetchone()
            is_explicit_child = ticket.parent_hypothesis_id is not None and bool(ticket.mutation_codes)
            if ticket.parent_hypothesis_id is not None:
                parent = connection.execute(
                    "SELECT 1 FROM hypothesis_intake WHERE hypothesis_id = ?", (ticket.parent_hypothesis_id,)
                ).fetchone()
                if parent is None:
                    raise ValueError("parent hypothesis must be admitted before its child")
            if semantic_row is not None and not is_explicit_child:
                connection.commit()
                return HypothesisAdmission(ticket.hypothesis_id, False, False, "SEMANTIC_DUPLICATE", exact, semantic)
            if (
                semantic_row is not None
                and is_explicit_child
                and semantic_row["hypothesis_id"] != ticket.parent_hypothesis_id
            ):
                connection.commit()
                return HypothesisAdmission(ticket.hypothesis_id, False, False, "SEMANTIC_DUPLICATE", exact, semantic)
            connection.execute(
                """
                INSERT INTO hypothesis_intake(
                    hypothesis_id, exact_hash, semantic_fingerprint, ticket_json,
                    parent_hypothesis_id, admitted_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket.hypothesis_id,
                    exact,
                    semantic,
                    _canonical(asdict(ticket)),
                    ticket.parent_hypothesis_id,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()
            return HypothesisAdmission(ticket.hypothesis_id, True, True, "ADMITTED", exact, semantic)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def assert_admitted(self, hypothesis_id: str) -> None:
        with sqlite3.connect(self.db_path, timeout=30.0) as connection:
            row = connection.execute(
                "SELECT 1 FROM hypothesis_intake WHERE hypothesis_id = ?", (hypothesis_id,)
            ).fetchone()
        if row is None:
            raise ValueError(f"hypothesis was not admitted through institutional intake: {hypothesis_id}")

    def directive(self, hypothesis_id: str, *, directive_id: str, variant: int = 1) -> MutationDirective:
        with sqlite3.connect(self.db_path, timeout=30.0) as connection:
            row = connection.execute(
                "SELECT ticket_json FROM hypothesis_intake WHERE hypothesis_id = ?", (hypothesis_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown admitted hypothesis: {hypothesis_id}")
        payload = json.loads(row[0])
        return MutationDirective(
            directive_id=directive_id,
            hypothesis_id=hypothesis_id,
            family=payload["family"],
            variant=variant,
            mutation_codes=tuple(payload.get("mutation_codes", ())),
        )


@dataclass(frozen=True)
class MalaysiaAuditInputs:
    strategy_returns: pd.Series
    market_returns: pd.Series
    factor_returns: pd.DataFrame
    regime_labels: pd.Series


class MalaysiaAuditInputSource(Protocol):
    def institutional_audit_inputs(self, candidate_id: str) -> MalaysiaAuditInputs:
        ...


class DeterministicMalaysiaInstitutionalAuditor:
    """Non-generative regime, factor-residual, and recent-decay checker."""

    def __init__(self, source: MalaysiaAuditInputSource, *, min_regime_observations: int = 63) -> None:
        self.source = source
        self.min_regime_observations = min_regime_observations

    def audit(
        self,
        candidate: CandidateSpec,
        evidence: BacktestEvidence,
        *,
        trial_id: str,
        universe: MalaysiaUniverseContract,
        config: CampaignConfig,
    ) -> InstitutionalAuditEvidence:
        inputs = self.source.institutional_audit_inputs(candidate.candidate_id)
        numeric = pd.concat(
            [
                inputs.strategy_returns.rename("strategy"),
                inputs.market_returns.rename("market"),
                inputs.factor_returns,
            ],
            axis=1,
            join="inner",
        ).dropna()
        if len(numeric) < 504 or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("institutional audit requires 504 finite aligned Malaysia observations")

        regime_frame = pd.concat(
            [inputs.strategy_returns.rename("strategy"), inputs.regime_labels.rename("regime")],
            axis=1,
            join="inner",
        ).dropna()

        counts = regime_frame.groupby("regime", observed=True)["strategy"].size()
        eligible_labels = counts[counts >= self.min_regime_observations].index
        eligible = regime_frame.loc[regime_frame["regime"].isin(eligible_labels)]
        regime_ir = eligible.groupby("regime", observed=True)["strategy"].apply(sharpe_ratio)
        regime_pnl = eligible.groupby("regime", observed=True)["strategy"].sum()
        positive = regime_pnl.clip(lower=0.0)
        positive_total = float(positive.sum())
        max_positive_share = float(positive.max() / positive_total) if positive_total > 0.0 else 1.0
        positive_fraction = float((regime_ir > 0.0).mean()) if len(regime_ir) else 0.0

        factor_columns = list(inputs.factor_returns.columns)
        regression = numeric[["strategy", *factor_columns]].dropna()
        y = regression["strategy"].to_numpy(dtype=float)
        factors = regression[factor_columns].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(regression)), factors])
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        residual = y - fitted
        adjusted = y - factors @ beta[1:]
        total_ss = float(np.sum((y - y.mean()) ** 2))
        r_squared = 0.0 if total_ss <= 0.0 else 1.0 - float(np.sum(residual**2)) / total_ss
        adjusted_series = pd.Series(adjusted, index=regression.index)
        residual_ir = float(sharpe_ratio(adjusted_series))
        alpha_t = float(_newey_west_mean_t(adjusted, max_lag=5))

        decay = sharpe_decay_analysis(numeric["strategy"], windows=(126, 252))
        decay_by_window = decay.set_index("window")["latest_sharpe"].to_dict()
        latest_126 = float(decay_by_window.get(126, float("-inf")))
        latest_252 = float(decay_by_window.get(252, float("-inf")))
        input_hash = _pandas_input_hash(numeric, inputs.regime_labels)
        methodology_hash = _digest(
            {
                "version": "malaysia_institutional_audit_v1",
                "regime": "lagged_126d_trend_x_lagged_63d_vol_expanding_33_67",
                "min_regime_observations": self.min_regime_observations,
                "factor_columns": factor_columns,
                "factor_gate": {"min_residual_ir": 0.25, "max_r_squared": 0.80, "min_alpha_t": 1.0},
                "decay_windows": [126, 252],
            }
        )
        metrics = {
            "candidate_id": candidate.candidate_id,
            "regime_count": int(len(regime_ir)),
            "min_regime_observations": int(counts.loc[eligible_labels].min()) if len(eligible_labels) else 0,
            "positive_regime_fraction": positive_fraction,
            "max_positive_pnl_share": max_positive_share,
            "worst_regime_ir": float(regime_ir.min()) if len(regime_ir) else float("-inf"),
            "factor_residual_ir": residual_ir,
            "factor_r_squared": float(max(0.0, min(1.0, r_squared))),
            "factor_alpha_t_stat": alpha_t,
            "decay_latest_126": latest_126,
            "decay_latest_252": latest_252,
            "input_hash": input_hash,
            "methodology_hash": methodology_hash,
        }
        return InstitutionalAuditEvidence(**metrics, audit_hash=_digest(metrics))


def _newey_west_mean_t(values: np.ndarray, *, max_lag: int) -> float:
    series = np.asarray(values, dtype=float)
    n = len(series)
    if n < 3:
        return 0.0
    centered = series - series.mean()
    long_run_variance = float(centered @ centered) / n
    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = float(centered[lag:] @ centered[:-lag]) / n
        long_run_variance += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    if long_run_variance <= 0.0:
        return 0.0
    return float(series.mean() / math.sqrt(long_run_variance / n))


def _pandas_input_hash(numeric: pd.DataFrame, regimes: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(numeric, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(regimes.reindex(numeric.index), index=True).values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FiniteCampaignSchedule:
    start_at: datetime
    campaign_count: int = 4
    interval_hours: int = 6
    max_trials_per_campaign: int = 3

    def validate(self) -> None:
        if self.start_at.tzinfo is None:
            raise ValueError("schedule start must be timezone-aware")
        if not 1 <= self.campaign_count <= 4:
            raise ValueError("at most four finite campaigns may be planned per day")
        if (self.campaign_count - 1) * self.interval_hours >= 24:
            raise ValueError("finite campaign plan must fit within one 24-hour window")
        if self.interval_hours < 1 or self.campaign_count * self.max_trials_per_campaign > 30:
            raise ValueError("schedule exceeds the bounded daily research budget")
        if not 1 <= self.max_trials_per_campaign <= 10:
            raise ValueError("per-campaign trial budget is invalid")

    def planned_runs(self) -> tuple[dict[str, object], ...]:
        self.validate()
        return tuple(
            {
                "sequence": index + 1,
                "run_at": (self.start_at + timedelta(hours=index * self.interval_hours)).isoformat(),
                "max_trials": self.max_trials_per_campaign,
                "requires_prior_campaign_terminal_state": index > 0,
            }
            for index in range(self.campaign_count)
        )
