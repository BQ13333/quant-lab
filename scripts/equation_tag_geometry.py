#!/usr/bin/env python3
"""Geometry-backed, formula-body-preserving equation-tag recovery.

This module does not infer tags from formula counts or reading order.  It
matches a source PDF label to a display-formula bounding box only when the
label is spatially to the formula's right, vertically close, uniquely matched,
and not a citation-like year.  Ambiguous inputs are recorded but not changed.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any


EQUATION_LABEL_RE = re.compile(r"^\((?:[A-Z]\.)?\d+(?:\.\d+){0,3}[a-z]?\)$")
TAG_RE = re.compile(r"\\tag\s*\{\s*([^{}]+?)\s*\}")
DISPLAY_ENV_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|displaymath)\}"
)


def canonical_label(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    if len(text) >= 2 and text[0] in "([" and text[-1] in ")]":
        text = text[1:-1]
    return text


def parenthesized_label(value: Any) -> str:
    inner = canonical_label(value)
    return f"({inner})" if inner else ""


def is_year_like_label(label: Any) -> bool:
    """Return true only for a standalone four-digit 1800--2099 year."""
    inner = canonical_label(label)
    return bool(re.fullmatch(r"(?:18|19|20)\d{2}", inner))


def is_numeric_value_like_label(label: Any) -> bool:
    """Reject high-magnitude decimals that are overwhelmingly table values."""
    inner = canonical_label(label)
    match = re.fullmatch(r"(\d+)\.(\d{2,})", inner)
    return bool(match and int(match.group(1)) >= 30)


def formula_tags(content: str) -> list[str]:
    return [canonical_label(match.group(1)) for match in TAG_RE.finditer(content)]


def formula_body_hash(content: str) -> str:
    without_tags = TAG_RE.sub("", content)
    normalized = re.sub(r"\s+", "", without_tags)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def inject_tag_into_display(content: str, label: str) -> tuple[str, str]:
    """Insert one canonical tag without changing the mathematical body."""
    inner = canonical_label(label)
    existing = formula_tags(content)
    if inner in existing:
        return content, "already_present"
    if existing:
        return content, "conflicting_tag"

    insertion = rf"\tag{{{inner}}}"
    closing_start = -1
    if content.count("$$") >= 2:
        closing_start = content.rfind("$$")
    elif r"\[" in content and r"\]" in content:
        closing_start = content.rfind(r"\]")
    else:
        begin = DISPLAY_ENV_RE.search(content)
        if begin:
            closing = rf"\end{{{begin.group(1)}}}"
            closing_start = content.rfind(closing)
    if closing_start <= 0:
        return content, "no_display_delimiter"

    prefix = content[:closing_start].rstrip()
    suffix = content[closing_start:]
    repaired = f"{prefix} {insertion}\n{suffix}"
    if formula_body_hash(repaired) != formula_body_hash(content):
        return content, "body_hash_mismatch"
    return repaired, "inserted"


def is_formula_element(element: dict[str, Any]) -> bool:
    return (
        str(element.get("label") or "") == "formula"
        or "formula" in str(element.get("native_label") or "").lower()
    )


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = map(float, value)
    except (TypeError, ValueError):
        return None
    if x1 < x0 or y1 < y0:
        return None
    return x0, y0, x1, y1


def _scaled_word_bbox(
    word: dict[str, Any], source_width: float, source_height: float, target_width: float, target_height: float
) -> tuple[float, float, float, float] | None:
    box = _bbox(word.get("bbox"))
    if box is None and all(field in word for field in ("x0", "top", "x1", "bottom")):
        box = _bbox([word["x0"], word["top"], word["x1"], word["bottom"]])
    if box is None or source_width <= 0 or source_height <= 0:
        return None
    sx, sy = target_width / source_width, target_height / source_height
    return box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy


def _decision_sort_key(decision: dict[str, Any]) -> tuple[Any, ...]:
    box = decision.get("source_bbox") or [0, 0, 0, 0]
    return (str(decision.get("label") or ""), *box, str(decision.get("status") or ""))


def build_geometry_tag_plan(
    elements: list[dict[str, Any]],
    source_words: list[dict[str, Any]],
    source_width: float,
    source_height: float,
) -> dict[str, Any]:
    """Return safe element-index tag assignments and full candidate decisions."""
    formulas: list[dict[str, Any]] = []
    blocked_regions: list[tuple[float, float, float, float]] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        box = _bbox(element.get("bbox_2d"))
        element_label = str(element.get("label") or "").lower()
        native_label = str(element.get("native_label") or "").lower()
        if box is not None and ("table" in element_label or "table" in native_label):
            blocked_regions.append(box)
        if not is_formula_element(element):
            continue
        if box is None:
            continue
        try:
            index = int(element.get("index"))
            width = float(element.get("width") or 0)
            height = float(element.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        formulas.append(
            {
                "index": index,
                "bbox": box,
                "width": width,
                "height": height,
                "content": str(element.get("content") or ""),
            }
        )
    formulas.sort(key=lambda item: item["index"])
    if not formulas:
        decisions = [
            {
                "label": parenthesized_label(word.get("text")),
                "status": "no_formula_geometry",
                "source_bbox": list(_bbox(word.get("bbox")) or ()),
            }
            for word in source_words
            if EQUATION_LABEL_RE.fullmatch(parenthesized_label(word.get("text")))
        ]
        return {"tags_by_element_index": {}, "decisions": sorted(decisions, key=_decision_sort_key)}

    target_width = formulas[0]["width"]
    target_height = formulas[0]["height"]
    candidates: list[dict[str, Any]] = []
    for word in source_words:
        label = parenthesized_label(word.get("text"))
        if not EQUATION_LABEL_RE.fullmatch(label):
            continue
        source_box = _bbox(word.get("bbox"))
        if source_box is None and all(field in word for field in ("x0", "top", "x1", "bottom")):
            source_box = _bbox([word["x0"], word["top"], word["x1"], word["bottom"]])
        candidates.append({"label": label, "word": word, "source_bbox": list(source_box or ())})
    candidates.sort(key=lambda item: (item["label"], item["source_bbox"]))

    decisions: list[dict[str, Any]] = []
    provisional: list[dict[str, Any]] = []
    for candidate in candidates:
        label = candidate["label"]
        base = {"label": label, "source_bbox": candidate["source_bbox"]}
        if is_year_like_label(label):
            decisions.append({**base, "status": "year_excluded"})
            continue
        if canonical_label(label) == "0":
            decisions.append({**base, "status": "zero_excluded"})
            continue
        matching_existing = [
            item for item in formulas if canonical_label(label) in formula_tags(item["content"])
        ]
        if matching_existing:
            decisions.append(
                {
                    **base,
                    "status": "already_present",
                    "element_index": matching_existing[0]["index"],
                    "formula_bbox": list(matching_existing[0]["bbox"]),
                }
            )
            continue
        word_box = _scaled_word_bbox(
            candidate["word"], source_width, source_height, target_width, target_height
        )
        if word_box is None:
            decisions.append({**base, "status": "invalid_source_geometry"})
            continue
        label_left = word_box[0]
        label_y = (word_box[1] + word_box[3]) / 2
        label_x = (word_box[0] + word_box[2]) / 2
        if any(x0 <= label_x <= x1 and y0 <= label_y <= y1 for x0, y0, x1, y1 in blocked_regions):
            decisions.append({**base, "status": "table_excluded"})
            continue
        if is_numeric_value_like_label(label):
            decisions.append({**base, "status": "numeric_value_excluded"})
            continue
        matches: list[tuple[float, float, float, int, dict[str, Any]]] = []
        for item in formulas:
            x0, y0, x1, y1 = item["bbox"]
            horizontal_gap = label_left - x1
            if horizontal_gap < -0.002 * target_width or horizontal_gap > 0.12 * target_width:
                continue
            edge_distance = 0.0 if y0 <= label_y <= y1 else min(abs(label_y - y0), abs(label_y - y1))
            if edge_distance > 0.02 * target_height:
                continue
            center_distance = abs(label_y - (y0 + y1) / 2)
            cost = edge_distance * 10 + center_distance + max(horizontal_gap, 0) * 0.1
            matches.append((cost, edge_distance, center_distance, item["index"], item))
        matches.sort(key=lambda value: (value[0], value[3]))
        if not matches:
            decisions.append({**base, "status": "no_spatial_match"})
            continue
        ambiguity_margin = max(10.0, 0.004 * target_height)
        if len(matches) > 1 and matches[1][0] - matches[0][0] < ambiguity_margin:
            decisions.append(
                {
                    **base,
                    "status": "ambiguous",
                    "candidate_element_indices": [match[3] for match in matches],
                }
            )
            continue
        if matches[0][1] > 0.01 * target_height:
            decisions.append({**base, "status": "no_spatial_match"})
            continue
        best = matches[0][4]
        existing = formula_tags(best["content"])
        if existing and canonical_label(label) not in existing:
            decisions.append(
                {
                    **base,
                    "status": "conflicting_tag",
                    "element_index": best["index"],
                    "formula_bbox": list(best["bbox"]),
                    "existing_tags": existing,
                }
            )
            continue
        repaired, injection_status = inject_tag_into_display(best["content"], label)
        if injection_status != "inserted":
            decisions.append(
                {
                    **base,
                    "status": injection_status,
                    "element_index": best["index"],
                    "formula_bbox": list(best["bbox"]),
                }
            )
            continue
        provisional.append(
            {
                **base,
                "status": "mapped",
                "element_index": best["index"],
                "formula_bbox": list(best["bbox"]),
                "body_sha256_before": formula_body_hash(best["content"]),
                "body_sha256_after": formula_body_hash(repaired),
                "spatial_cost": round(matches[0][0], 6),
                "vertical_edge_distance": round(matches[0][1], 6),
                "vertical_center_distance": round(matches[0][2], 6),
                "horizontal_gap": round(label_left - best["bbox"][2], 6),
                "label_x_ratio": round(label_x / target_width, 6),
            }
        )

    if len(provisional) > 20:
        decisions.extend({**item, "status": "density_excluded"} for item in provisional)
        decisions.sort(key=_decision_sort_key)
        return {"tags_by_element_index": {}, "decisions": decisions}

    label_counts = Counter(item["label"] for item in provisional)
    element_counts = Counter(item["element_index"] for item in provisional)
    tags: dict[int, str] = {}
    for item in provisional:
        if label_counts[item["label"]] > 1:
            decisions.append({**item, "status": "duplicate_label"})
            continue
        if element_counts[item["element_index"]] > 1:
            decisions.append({**item, "status": "ambiguous"})
            continue
        tags[int(item["element_index"])] = str(item["label"])
        decisions.append(item)
    tags = dict(sorted(tags.items()))
    decisions.sort(key=_decision_sort_key)
    return {"tags_by_element_index": tags, "decisions": decisions}
