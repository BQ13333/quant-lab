from quant_lab.agents.phd_math_specialists import (
    EXPECTED_SPECIALIST_COUNT,
    build_review_panel,
    load_specialist_panel,
    route_specialists,
)


def test_canonical_panel_has_exactly_twenty_unique_specialists() -> None:
    panel = load_specialist_panel()
    assert len(panel.specialists) == EXPECTED_SPECIALIST_COUNT == 20
    assert len({specialist.id for specialist in panel.specialists}) == 20


def test_panel_has_four_allowed_verdicts() -> None:
    panel = load_specialist_panel()
    assert set(panel.allowed_verdicts) == {"pass", "conditional", "fail", "abstain"}


def test_neural_operator_problem_routes_to_correct_specialist() -> None:
    routed = route_specialists(
        "Review an FNO neural operator for a pricing PDE and its no-arbitrage error",
        limit=3,
    )
    assert routed[0].id == "neural_operator_scientific_ml_specialist"


def test_unknown_problem_uses_conservative_default_panel() -> None:
    routed = route_specialists("Review this entirely unspecified proposal", limit=3)
    assert [item.id for item in routed] == [
        "measure_probability_martingale_specialist",
        "selection_bias_validation_specialist",
        "optimization_information_geometry_specialist",
    ]


def test_review_panel_assigns_independent_roles_and_prompts() -> None:
    panel = build_review_panel(
        "Audit a neural SDE trained on irregular order-book events",
        tags=["neural_sde", "microstructure"],
    )
    assert len(panel) == 3
    assert [item["debate_role"] for item in panel] == [
        "lead",
        "independent_challenger",
        "adversarial_challenger",
    ]
    assert all("verdict: pass, conditional, fail, or abstain" in item["prompt"] for item in panel)
