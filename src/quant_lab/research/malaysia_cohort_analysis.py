from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.research.malaysia_strategy_registry import (
    COHORT_VERSION,
    malaysia_atomic_cohort,
    malaysia_predeclared_combinations,
)
from quant_lab.research.trial_ledger import TrialLedger
from quant_lab.validation.metrics import max_drawdown, sharpe_ratio
from quant_lab.validation.overfit import probability_of_backtest_overfitting

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore[assignment]


def build_malaysia_cohort_report(ledger: TrialLedger) -> dict[str, Any]:
    expected = malaysia_atomic_cohort()
    expected_candidate_ids = {item.strategy_id: f"MYS-C-{item.strategy_id}" for item in expected}
    matrix = ledger.candidate_return_matrix(campaign="malaysia")
    columns_by_candidate: dict[str, list[str]] = {}
    for column in matrix.columns:
        candidate_id = str(column).split("::", 1)[0]
        columns_by_candidate.setdefault(candidate_id, []).append(str(column))
    duplicates = {
        candidate_id: columns
        for candidate_id, columns in columns_by_candidate.items()
        if candidate_id in expected_candidate_ids.values() and len(columns) > 1
    }
    if duplicates:
        raise RuntimeError("an atomic cohort strategy was executed more than once")
    observed = {
        strategy_id: columns_by_candidate[candidate_id][0]
        for strategy_id, candidate_id in expected_candidate_ids.items()
        if candidate_id in columns_by_candidate
    }
    missing = [item.strategy_id for item in expected if item.strategy_id not in observed]
    report: dict[str, Any] = {
        "cohort_version": COHORT_VERSION,
        "expected_strategy_count": len(expected),
        "completed_strategy_count": len(observed),
        "completed_strategy_ids": [item.strategy_id for item in expected if item.strategy_id in observed],
        "missing_strategy_ids": missing,
        "cohort_closed": not missing,
        "ranking_available": not missing,
        "atomic_metrics": None,
        "correlation_matrix": None,
        "pbo": None,
        "predeclared_combination_metrics": None,
    }
    if missing:
        return report

    cohort_matrix = pd.DataFrame(
        {strategy_id: matrix[column] for strategy_id, column in observed.items()}
    ).sort_index()
    report["returns_matrix_hash"] = _frame_hash(cohort_matrix)
    raw_pvalues = {column: _one_sided_hac_pvalue(cohort_matrix[column]) for column in cohort_matrix}
    adjusted = _holm_adjust(raw_pvalues)
    report["atomic_metrics"] = {
        column: {
            "net_sharpe": float(sharpe_ratio(cohort_matrix[column])),
            "max_drawdown": float(max_drawdown(cohort_matrix[column])),
            "mean_daily_return": float(cohort_matrix[column].mean()),
            "hac_one_sided_pvalue": raw_pvalues[column],
            "holm_adjusted_pvalue": adjusted[column],
        }
        for column in cohort_matrix
    }
    report["correlation_matrix"] = {
        row: {column: float(value) for column, value in values.items()}
        for row, values in cohort_matrix.corr().to_dict(orient="index").items()
    }
    report["pbo"] = float(
        probability_of_backtest_overfitting(cohort_matrix, n_splits=8, strict=True)
    )
    combination_metrics: dict[str, Any] = {}
    for combination in malaysia_predeclared_combinations():
        combined = cohort_matrix[list(combination.component_strategy_ids)].mean(axis=1)
        component_sharpes = {
            item: float(sharpe_ratio(cohort_matrix[item])) for item in combination.component_strategy_ids
        }
        best_component = max(component_sharpes, key=component_sharpes.get)
        active = combined - cohort_matrix[best_component]
        combination_metrics[combination.combination_id] = {
            "components": list(combination.component_strategy_ids),
            "weighting": combination.weighting,
            "net_sharpe": float(sharpe_ratio(combined)),
            "max_drawdown": float(max_drawdown(combined)),
            "best_component": best_component,
            "best_component_sharpe": component_sharpes[best_component],
            "incremental_ir_vs_best_component": float(sharpe_ratio(active)),
        }
    report["predeclared_combination_metrics"] = combination_metrics
    return report


def persist_malaysia_cohort_report(ledger: TrialLedger, target: str | Path) -> Path:
    """Atomically persist the closed-cohort report as strict, content-stable JSON."""

    report = build_malaysia_cohort_report(ledger)
    if not report["cohort_closed"]:
        raise RuntimeError("Malaysia cohort report cannot be frozen before cohort closure")
    path = Path(target)
    content = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return path


def _frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update("|".join(str(item) for item in frame.columns).encode("utf-8"))
    return digest.hexdigest()


def _one_sided_hac_pvalue(series: pd.Series, *, max_lag: int = 5) -> float:
    values = series.to_numpy(dtype=float)
    centered = values - values.mean()
    n = len(values)
    long_run_variance = float(centered @ centered) / n
    for lag in range(1, min(max_lag, n - 1) + 1):
        covariance = float(centered[lag:] @ centered[:-lag]) / n
        long_run_variance += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    if long_run_variance <= 0.0:
        return 1.0
    statistic = float(values.mean() / math.sqrt(long_run_variance / n))
    if stats is not None:
        return float(stats.norm.sf(statistic))
    return float(0.5 * math.erfc(statistic / math.sqrt(2.0)))


def _holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    previous = 0.0
    for rank, key in enumerate(ordered):
        value = min(1.0, (count - rank) * pvalues[key])
        previous = max(previous, value)
        adjusted[key] = previous
    return adjusted
