"""Read the historical cost inventory. This is not a fee calculator.

The numbers are copied from the unmerged quant-lab PR #1 rules file so the
gap is reviewable here. They are not certified, and they are not applied.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_cost_coverage(path: Path | None = None) -> dict[str, object]:
    target = path or Path(__file__).resolve().parents[3] / "configs" / "historical_cost_coverage.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cost coverage inventory must be an object")
    return payload


def historical_cost_gate_met(path: Path | None = None) -> bool:
    payload = load_cost_coverage(path)
    return payload.get("certified") is True and payload.get("phase1_historical_cost_gate") == "met"
