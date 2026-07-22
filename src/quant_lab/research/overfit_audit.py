from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.validation.overfit import probability_of_backtest_overfitting


@dataclass(frozen=True)
class OverfitAuditResult:
    summary: dict[str, Any]
    top_family_summary: pd.DataFrame
    top_source_summary: pd.DataFrame
    correlation_summary: pd.DataFrame


def _off_diagonal_values(matrix: pd.DataFrame) -> np.ndarray:
    if matrix.empty or matrix.shape[0] < 2:
        return np.array([], dtype=float)
    values = matrix.to_numpy(dtype=float)
    mask = ~np.eye(values.shape[0], dtype=bool)
    return values[mask]


def _effective_independent_trials(correlation: pd.DataFrame) -> float:
    if correlation.empty:
        return 0.0
    corr = correlation.fillna(0.0).to_numpy(dtype=float)
    if corr.shape[0] == 1:
        return 1.0
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    denom = float(np.square(eigenvalues).sum())
    if denom == 0.0:
        return 0.0
    return float(np.square(eigenvalues.sum()) / denom)


def _summarize_top_group(scoreboard: pd.DataFrame, column: str, top_n: int) -> pd.DataFrame:
    if scoreboard.empty or column not in scoreboard.columns:
        return pd.DataFrame()
    top = scoreboard.head(top_n).copy()
    rows = []
    for value, group in top.groupby(column, dropna=False):
        rows.append(
            {
                column: value,
                "top_count": int(len(group)),
                "top_share": float(len(group) / max(1, len(top))),
                "best_score": float(group["score"].max()) if "score" in group else 0.0,
                "best_net_ir": float(group["net_ir"].max()) if "net_ir" in group else 0.0,
                "best_deflated_sharpe": float(group["deflated_sharpe"].max()) if "deflated_sharpe" in group else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["top_count", "best_score"], ascending=[False, False])


def build_overfit_audit(
    scoreboard: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    top_fraction: float = 0.10,
) -> OverfitAuditResult:
    """Build an anti-overfit audit using CSCV selection-rank PBO."""

    scoreboard = scoreboard.copy()
    if not scoreboard.empty and "score" in scoreboard.columns:
        scoreboard = scoreboard.sort_values(["passed_all", "score"], ascending=[False, False]).reset_index(drop=True)

    returns = returns_matrix.copy()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0.0)
    candidate_count = int(len(scoreboard))
    feasible_count = int(scoreboard["passed_all"].sum()) if "passed_all" in scoreboard else 0
    top_n = max(1, int(np.ceil(candidate_count * top_fraction))) if candidate_count else 0

    pbo_cscv = probability_of_backtest_overfitting(returns) if returns.shape[1] >= 2 else 0.0
    corr = returns.corr() if returns.shape[1] >= 2 else pd.DataFrame()
    abs_off_diag = np.abs(_off_diagonal_values(corr))
    effective_trials = _effective_independent_trials(corr)

    best = scoreboard.iloc[0].to_dict() if not scoreboard.empty else {}
    deflated_pass_rate = (
        float(scoreboard["constraint_deflated_sharpe_passed"].mean())
        if "constraint_deflated_sharpe_passed" in scoreboard and len(scoreboard)
        else 0.0
    )
    feasible_rate = float(feasible_count / candidate_count) if candidate_count else 0.0
    median_abs_corr = float(np.nanmedian(abs_off_diag)) if abs_off_diag.size else 0.0
    max_abs_corr = float(np.nanmax(abs_off_diag)) if abs_off_diag.size else 0.0

    warnings: list[str] = []
    if candidate_count >= 100 and feasible_rate > 0.20:
        warnings.append("high_feasible_rate_after_many_trials")
    if pbo_cscv > 0.45:
        warnings.append("high_cscv_pbo")
    if median_abs_corr > 0.75:
        warnings.append("high_candidate_return_correlation")
    if effective_trials and effective_trials < max(3.0, np.sqrt(max(1, candidate_count)) / 2.0):
        warnings.append("low_effective_independent_trials")
    if deflated_pass_rate < 0.01:
        warnings.append("very_few_candidates_pass_deflated_sharpe")
    if not feasible_count:
        warnings.append("no_candidate_passed_all_gates")

    summary = {
        "candidate_count": candidate_count,
        "feasible_candidates": feasible_count,
        "feasible_rate": feasible_rate,
        "top_n": top_n,
        "pbo_cscv": float(pbo_cscv),
        "pbo_even_odd_proxy": float(pbo_cscv),
        "effective_independent_trials": effective_trials,
        "median_abs_candidate_return_correlation": median_abs_corr,
        "max_abs_candidate_return_correlation": max_abs_corr,
        "deflated_sharpe_pass_rate": deflated_pass_rate,
        "best_candidate_id": best.get("candidate_id", ""),
        "best_model_name": best.get("model_name", ""),
        "best_family": best.get("family", best.get("meta_family", "")),
        "best_source_strategy_id": best.get("meta_source_strategy_id", ""),
        "best_net_ir": float(best.get("net_ir", 0.0) or 0.0),
        "best_net_sharpe": float(best.get("net_sharpe", 0.0) or 0.0),
        "best_deflated_sharpe": float(best.get("deflated_sharpe", 0.0) or 0.0),
        "best_passed_all": bool(best.get("passed_all", False)),
        "warnings": warnings,
        "audit_posture": "research_grade" if feasible_count and not warnings else "diagnostic_only",
    }

    family_column = "family" if "family" in scoreboard.columns else "meta_family"
    top_family_summary = _summarize_top_group(scoreboard, family_column, top_n)
    source_column = "meta_source_strategy_id"
    top_source_summary = _summarize_top_group(scoreboard, source_column, top_n)
    correlation_summary = pd.DataFrame(
        [
            {
                "metric": "median_abs_candidate_return_correlation",
                "value": median_abs_corr,
            },
            {
                "metric": "max_abs_candidate_return_correlation",
                "value": max_abs_corr,
            },
            {
                "metric": "effective_independent_trials",
                "value": effective_trials,
            },
            {
                "metric": "pbo_cscv",
                "value": float(pbo_cscv),
            },
        ]
    )
    return OverfitAuditResult(
        summary=summary,
        top_family_summary=top_family_summary,
        top_source_summary=top_source_summary,
        correlation_summary=correlation_summary,
    )


def write_overfit_audit(
    scoreboard: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    artifact_dir: str | Path,
) -> OverfitAuditResult:
    artifact_path = Path(artifact_dir)
    audit = build_overfit_audit(scoreboard, returns_matrix)
    (artifact_path / "overfit_audit.json").write_text(json.dumps(audit.summary, indent=2), encoding="utf-8")
    audit.top_family_summary.to_csv(artifact_path / "overfit_top_families.csv", index=False)
    audit.top_source_summary.to_csv(artifact_path / "overfit_top_sources.csv", index=False)
    audit.correlation_summary.to_csv(artifact_path / "overfit_correlation_summary.csv", index=False)
    return audit
