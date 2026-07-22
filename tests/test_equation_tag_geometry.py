from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "equation_tag_geometry.py"
SPEC = importlib.util.spec_from_file_location("equation_tag_geometry", SCRIPT)
assert SPEC and SPEC.loader
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


def formula(index: int, bbox: tuple[int, int, int, int], body: str) -> dict[str, object]:
    return {
        "index": index,
        "label": "formula",
        "native_label": "display_formula",
        "content": f"$$\n{body}\n$$",
        "bbox_2d": list(bbox),
        "width": 1000,
        "height": 1000,
    }


def word(text: str, bbox: tuple[int, int, int, int]) -> dict[str, object]:
    return {"text": text, "bbox": list(bbox)}


def decision_for(plan: dict[str, object], label: str) -> dict[str, object]:
    decisions = plan["decisions"]
    assert isinstance(decisions, list)
    matches = [decision for decision in decisions if decision.get("label") == label]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("label", ["2024", "(2024)", "[2024]", "  ( 2024 )  "])
def test_year_like_labels_are_excluded(label: str) -> None:
    assert geometry.is_year_like_label(label)


@pytest.mark.parametrize("label", ["(24)", "(2.024)", "(2024a)", "(1789)", "(2101)"])
def test_equation_like_or_out_of_range_labels_are_not_years(label: str) -> None:
    assert not geometry.is_year_like_label(label)


def test_two_columns_are_matched_by_geometry_not_crossed_ocr_index_order() -> None:
    elements = [
        formula(90, (80, 180, 390, 240), "L=1"),
        formula(3, (550, 130, 850, 190), "R=2"),
    ]
    source_words = [
        word("(L.1)", (410, 190, 460, 220)),
        word("(R.1)", (870, 140, 920, 170)),
    ]

    plan = geometry.build_geometry_tag_plan(elements, source_words, 1000, 1000)

    assert plan["tags_by_element_index"] == {3: "(R.1)", 90: "(L.1)"}


def test_numbered_subset_maps_spatially_without_shifting_over_unnumbered_formula() -> None:
    elements = [
        formula(0, (100, 100, 780, 150), "a=1"),
        formula(1, (100, 240, 780, 290), "b=2"),
        formula(2, (100, 380, 780, 430), "c=3"),
    ]
    source_words = [
        word("(1)", (830, 105, 870, 145)),
        word("(2)", (830, 385, 870, 425)),
    ]

    plan = geometry.build_geometry_tag_plan(elements, source_words, 1000, 1000)

    assert plan["tags_by_element_index"] == {0: "(1)", 2: "(2)"}
    assert 1 not in plan["tags_by_element_index"]


def test_year_candidate_is_recorded_but_never_assigned() -> None:
    plan = geometry.build_geometry_tag_plan(
        [formula(4, (100, 200, 800, 260), "x=1")],
        [word("(2024)", (840, 205, 900, 250))],
        1000,
        1000,
    )

    assert plan["tags_by_element_index"] == {}
    assert decision_for(plan, "(2024)")["status"] == "year_excluded"


def test_table_value_is_excluded_even_when_aligned_with_a_formula() -> None:
    elements = [
        formula(4, (100, 200, 700, 260), "x=1"),
        {
            "index": 5,
            "label": "table",
            "native_label": "table",
            "content": "",
            "bbox_2d": [720, 180, 940, 300],
            "width": 1000,
            "height": 1000,
        },
    ]
    plan = geometry.build_geometry_tag_plan(
        elements,
        [word("(56.71)", (760, 205, 830, 250))],
        1000,
        1000,
    )

    assert plan["tags_by_element_index"] == {}
    assert decision_for(plan, "(56.71)")["status"] == "table_excluded"


@pytest.mark.parametrize(
    ("label", "status"),
    [("(0)", "zero_excluded"), ("(56.71)", "numeric_value_excluded")],
)
def test_value_like_labels_are_never_assigned(label: str, status: str) -> None:
    plan = geometry.build_geometry_tag_plan(
        [formula(4, (100, 200, 700, 260), "x=1")],
        [word(label, (760, 205, 830, 250))],
        1000,
        1000,
    )
    assert plan["tags_by_element_index"] == {}
    assert decision_for(plan, label)["status"] == status


def test_ambiguous_equidistant_candidate_is_skipped() -> None:
    elements = [
        formula(1, (100, 100, 700, 160), "a=1"),
        formula(2, (100, 200, 700, 260), "b=2"),
    ]
    plan = geometry.build_geometry_tag_plan(
        elements,
        [word("(7)", (760, 165, 800, 195))],
        1000,
        1000,
    )

    assert plan["tags_by_element_index"] == {}
    assert decision_for(plan, "(7)")["status"] == "ambiguous"


def test_duplicate_source_labels_are_not_silently_assigned_twice() -> None:
    elements = [
        formula(1, (100, 100, 700, 160), "a=1"),
        formula(2, (100, 300, 700, 360), "b=2"),
    ]
    source_words = [
        word("(8)", (760, 105, 800, 150)),
        word("(8)", (760, 305, 800, 350)),
    ]

    plan = geometry.build_geometry_tag_plan(elements, source_words, 1000, 1000)

    assert plan["tags_by_element_index"] == {}
    duplicate_decisions = [
        decision for decision in plan["decisions"] if decision.get("label") == "(8)"
    ]
    assert len(duplicate_decisions) == 2
    assert {decision["status"] for decision in duplicate_decisions} == {"duplicate_label"}


def test_shuffled_inputs_produce_identical_plan_and_decision_order() -> None:
    elements = [
        formula(50, (100, 100, 760, 160), "a=1"),
        formula(7, (100, 300, 760, 360), "b=2"),
    ]
    source_words = [
        word("(1.1)", (800, 105, 860, 150)),
        word("(1.2)", (800, 305, 860, 350)),
    ]

    forward = geometry.build_geometry_tag_plan(elements, source_words, 1000, 1000)
    reversed_inputs = geometry.build_geometry_tag_plan(
        list(reversed(elements)), list(reversed(source_words)), 1000, 1000
    )

    assert forward == reversed_inputs
    assert forward["tags_by_element_index"] == {7: "(1.2)", 50: "(1.1)"}


@pytest.mark.parametrize(
    ("content", "closing"),
    [
        ("$$\nx^2+y^2=1\n$$", "$$"),
        (r"\[x^2+y^2=1\]", r"\]"),
        (r"\begin{equation}x^2+y^2=1\end{equation}", r"\end{equation}"),
        (r"\begin{align}x&=1\\y&=2\end{align}", r"\end{align}"),
        (r"\begin{align*}x&=1\\y&=2\end{align*}", r"\end{align*}"),
    ],
)
def test_injects_tag_immediately_before_supported_display_closer(
    content: str, closing: str
) -> None:
    repaired, status = geometry.inject_tag_into_display(content, "(2.7)")

    assert status == "inserted"
    assert r"\tag{2.7}" in repaired
    assert repaired.index(r"\tag{2.7}") < repaired.rindex(closing)


def test_existing_spaced_tag_is_idempotent() -> None:
    content = "$$\nx=1 \\tag { 2.7 }\n$$"

    repaired, status = geometry.inject_tag_into_display(content, "(2.7)")

    assert status == "already_present"
    assert repaired == content


def test_conflicting_existing_tag_is_refused_without_mutation() -> None:
    content = "$$\nx=1 \\tag{2.6}\n$$"

    repaired, status = geometry.inject_tag_into_display(content, "(2.7)")

    assert status == "conflicting_tag"
    assert repaired == content


@pytest.mark.parametrize("content", ["x=1", "$x=1$", r"\(x=1\)"])
def test_no_display_delimiter_is_refused_without_mutation(content: str) -> None:
    repaired, status = geometry.inject_tag_into_display(content, "(3)")

    assert status == "no_display_delimiter"
    assert repaired == content


@pytest.mark.parametrize(
    "content",
    [
        "$$\n\\frac{x_1 + y}{z}=q\n$$",
        r"\[\frac{x_1 + y}{z}=q\]",
        r"\begin{equation}\frac{x_1 + y}{z}=q\end{equation}",
        r"\begin{align}x_1+y&=z\\z/q&=4\end{align}",
    ],
)
def test_injection_changes_only_the_added_tag(content: str) -> None:
    repaired, status = geometry.inject_tag_into_display(content, "(A.12)")
    assert status == "inserted"

    stripped = re.sub(r"\s*\\tag\s*\{\s*A\.12\s*\}\s*", "", repaired, count=1)
    # The insertion may consume whitespace immediately before the closer, so compare
    # mathematical content rather than formatting whitespace.
    assert re.sub(r"\s+", "", stripped) == re.sub(r"\s+", "", content)
