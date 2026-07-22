from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "phd_math_specialists.json"
)
EXPECTED_SPECIALIST_COUNT = 20
ALLOWED_VERDICTS = {"pass", "conditional", "fail", "abstain"}


@dataclass(frozen=True)
class PhDMathSpecialist:
    id: str
    title: str
    lane: str
    routing_tags: tuple[str, ...]
    mandate: str
    proof_obligations: tuple[str, ...]
    kill_criteria: tuple[str, ...]
    corpus_anchors: tuple[str, ...]


@dataclass(frozen=True)
class SpecialistPanel:
    name: str
    principle: str
    allowed_verdicts: tuple[str, ...]
    required_dossier_fields: tuple[str, ...]
    debate_protocol: dict[str, object]
    specialists: tuple[PhDMathSpecialist, ...]

    def by_id(self, specialist_id: str) -> PhDMathSpecialist:
        for specialist in self.specialists:
            if specialist.id == specialist_id:
                return specialist
        raise KeyError(f"Unknown PhD math specialist: {specialist_id}")


def load_specialist_panel(path: str | Path | None = None) -> SpecialistPanel:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    specialists = tuple(
        PhDMathSpecialist(
            id=item["id"],
            title=item["title"],
            lane=item["lane"],
            routing_tags=tuple(item["routing_tags"]),
            mandate=item["mandate"],
            proof_obligations=tuple(item["proof_obligations"]),
            kill_criteria=tuple(item["kill_criteria"]),
            corpus_anchors=tuple(item["corpus_anchors"]),
        )
        for item in payload["specialists"]
    )
    panel = SpecialistPanel(
        name=payload["name"],
        principle=payload["principle"],
        allowed_verdicts=tuple(payload["allowed_verdicts"]),
        required_dossier_fields=tuple(payload["required_dossier_fields"]),
        debate_protocol=dict(payload["debate_protocol"]),
        specialists=specialists,
    )
    validate_specialist_panel(panel)
    return panel


def validate_specialist_panel(panel: SpecialistPanel) -> None:
    if len(panel.specialists) != EXPECTED_SPECIALIST_COUNT:
        raise ValueError(
            f"The canonical panel must contain exactly {EXPECTED_SPECIALIST_COUNT} "
            f"specialists; found {len(panel.specialists)}"
        )
    ids = [specialist.id for specialist in panel.specialists]
    if len(set(ids)) != len(ids):
        raise ValueError("Specialist IDs must be unique")
    if set(panel.allowed_verdicts) != ALLOWED_VERDICTS:
        raise ValueError(f"Verdicts must be exactly {sorted(ALLOWED_VERDICTS)}")
    for specialist in panel.specialists:
        if not re.fullmatch(r"[a-z][a-z0-9_]+", specialist.id):
            raise ValueError(f"Invalid specialist ID: {specialist.id}")
        if not specialist.routing_tags:
            raise ValueError(f"{specialist.id} has no routing tags")
        if len(specialist.proof_obligations) < 3:
            raise ValueError(f"{specialist.id} needs at least three proof obligations")
        if len(specialist.kill_criteria) < 3:
            raise ValueError(f"{specialist.id} needs at least three kill criteria")
        if not specialist.corpus_anchors:
            raise ValueError(f"{specialist.id} has no corpus anchors")


def route_specialists(
    problem: str,
    tags: Iterable[str] = (),
    *,
    limit: int = 3,
    panel: SpecialistPanel | None = None,
) -> list[PhDMathSpecialist]:
    """Route a problem to independent specialists using transparent tag matching."""

    if limit < 1:
        raise ValueError("limit must be positive")
    active_panel = panel or load_specialist_panel()
    text = " ".join([problem, *tags]).lower().replace("-", "_")
    phrase_text = re.sub(r"\s+", "_", text)
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    scored: list[tuple[int, int, PhDMathSpecialist]] = []
    for index, specialist in enumerate(active_panel.specialists):
        score = 0
        for tag in specialist.routing_tags:
            normalized = tag.lower().replace("-", "_")
            if normalized in tokens:
                score += 3
            elif normalized in text or normalized in phrase_text:
                score += 1
        scored.append((score, -index, specialist))
    ranked = [item[2] for item in sorted(scored, reverse=True)]
    selected = [specialist for score, _, specialist in sorted(scored, reverse=True) if score > 0]
    if not selected:
        selected = [
            active_panel.by_id("measure_probability_martingale_specialist"),
            active_panel.by_id("selection_bias_validation_specialist"),
            active_panel.by_id("optimization_information_geometry_specialist"),
        ]
    for default in ranked:
        if len(selected) >= limit:
            break
        if default not in selected:
            selected.append(default)
    return selected[:limit]


def build_specialist_prompt(specialist: PhDMathSpecialist, problem: str) -> str:
    obligations = "\n".join(f"- {item}" for item in specialist.proof_obligations)
    kill_criteria = "\n".join(f"- {item}" for item in specialist.kill_criteria)
    anchors = ", ".join(specialist.corpus_anchors)
    return f"""You are the {specialist.title}, a PhD-level mathematical review role.

Mandate: {specialist.mandate}

Problem under review:
{problem}

Required proof obligations:
{obligations}

Fatal or blocking criteria:
{kill_criteria}

Preferred corpus anchors: {anchors}

Return a structured review that separates theorem, assumption, empirical estimate,
numerical approximation, and conjecture. State the mathematical object, information
set or filtration, estimand or decision objective, identification status, assumption
diagnostics, error claims, falsification tests, counterexamples, and simple baseline.
End with exactly one verdict: pass, conditional, fail, or abstain. An unsupported
confidence score, citation, backtest Sharpe, or majority opinion is not evidence.
"""


def build_review_panel(
    problem: str,
    tags: Iterable[str] = (),
    *,
    panel_size: int = 3,
    panel: SpecialistPanel | None = None,
) -> list[dict[str, str]]:
    specialists = route_specialists(
        problem,
        tags,
        limit=panel_size,
        panel=panel,
    )
    roles = ("lead", "independent_challenger", "adversarial_challenger")
    return [
        {
            "specialist_id": specialist.id,
            "title": specialist.title,
            "debate_role": roles[index] if index < len(roles) else "challenger",
            "prompt": build_specialist_prompt(specialist, problem),
        }
        for index, specialist in enumerate(specialists)
    ]
