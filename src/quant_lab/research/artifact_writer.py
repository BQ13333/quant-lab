from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    """Convert a DataFrame to a Markdown table string."""
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).fillna("")
    columns = list(clipped.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in clipped.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def _write_md(path: Path, title: str, body_lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + "\n".join(body_lines) + "\n", encoding="utf-8")
    return path


def write_index_md(
    run_id: str,
    artifact_dir: str | Path,
    leaderboard: pd.DataFrame,
    coverage: pd.DataFrame,
    eligible_prices: pd.DataFrame,
    diagnostics_paths: dict[str, Path] | None,
    save_dir: str | Path,
) -> Path:
    """Experiment summary with best result, caveats, and links."""
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    artifact_link = Path(artifact_dir).resolve()

    body = [
        f"Run ID: `{run_id}`",
        f"Artifact directory: `{artifact_link}`",
        f"Eligible universe: `{eligible_prices.shape[1]}` tickers across `{eligible_prices.shape[0]}` dates.",
        "",
        "## Best Result",
        "No leaderboard was produced." if not best else _markdown_table(pd.DataFrame([best])),
        "",
        "## Diagnostics Artifacts",
        "\n".join(f"- `{k}`: `{v}`" for k, v in (diagnostics_paths or {}).items()),
        "",
        "## Coverage Snapshot",
        _markdown_table(coverage.head(20)) if not coverage.empty else "_No coverage data._",
        "",
        "## Caveats",
        "- Results are based on historical backtests and do not guarantee future performance.",
        "- Transaction costs and slippage estimates are model-dependent.",
        "- Data quality issues may affect strategy results.",
    ]

    path = Path(save_dir) / f"{run_id}_index.md"
    return _write_md(path, f"{run_id} Experiment Index", body)


def write_validation_report_md(
    run_id: str,
    artifact_dir: str | Path,
    leaderboard: pd.DataFrame,
    coverage: pd.DataFrame,
    failures: dict[str, dict[str, str]] | None,
    save_dir: str | Path,
) -> Path:
    """Full leaderboard, coverage snapshot, failures, and promotion gate outcomes."""
    artifact_link = Path(artifact_dir).resolve()

    body = [
        f"Run ID: `{run_id}`",
        f"Artifact directory: `{artifact_link}`",
        "",
        "## Leaderboard",
        _markdown_table(leaderboard) if not leaderboard.empty else "_No leaderboard._",
        "",
        "## Coverage Snapshot",
        _markdown_table(coverage.head(50)) if not coverage.empty else "_No coverage data._",
        "",
        "## Failures",
    ]

    for tier, tier_failures in (failures or {}).items():
        body.append(f"### {tier}")
        body.append(
            "None"
            if not tier_failures
            else "\n".join(f"- `{ticker}`: {reason}" for ticker, reason in tier_failures.items())
        )

    body.extend([
        "",
        "## Promotion Gate Outcomes",
        "See model cards for detailed gate evaluations.",
    ])

    path = Path(save_dir) / f"{run_id}_validation_report.md"
    return _write_md(path, f"{run_id} Validation Report", body)


def write_cost_sensitivity_md(
    run_id: str,
    cost_sensitivity: pd.DataFrame,
    best_by_cost: pd.DataFrame,
    save_dir: str | Path,
) -> Path:
    """Cost sensitivity table and interpretation prompts."""
    body = [
        f"Run ID: `{run_id}`",
        "",
        "## Cost Sensitivity",
        _markdown_table(cost_sensitivity) if not cost_sensitivity.empty else "_No cost sensitivity data._",
        "",
        "## Best By Cost",
        _markdown_table(best_by_cost) if not best_by_cost.empty else "_No best-by-cost data._",
        "",
        "## Interpretation Prompts",
        "- If the winner changes dramatically between `0` and `15` bps, the edge is friction-sensitive.",
        "- A robust candidate should retain positive active return and acceptable drawdown at stressed costs.",
        "- Compare `annual_excess_return`, `tracking_error`, and `information_ratio`, not only Sharpe.",
    ]

    path = Path(save_dir) / f"{run_id}_cost_sensitivity.md"
    return _write_md(path, f"{run_id} Cost Sensitivity", body)


def write_model_card_md(
    run_id: str,
    model_name: str,
    model_card_df: pd.DataFrame,
    save_dir: str | Path,
) -> Path:
    """Model stats, stress results, pass/fail flags, and caveats."""
    body = [
        f"Run ID: `{run_id}`",
        f"Model: `{model_name}`",
        "",
        "## Model Statistics",
        _markdown_table(model_card_df) if not model_card_df.empty else "_No model card data._",
        "",
        "## Caveats",
        "- Model statistics are based on historical backtest data.",
        "- Past performance does not guarantee future results.",
        "- Cost assumptions may vary by broker and market conditions.",
    ]

    path = Path(save_dir) / f"{run_id}_{model_name}_model_card.md"
    return _write_md(path, f"{run_id} {model_name} Model Card", body)


def write_failure_modes_md(
    run_id: str,
    failed_models: list[str],
    reasons: list[str],
    save_dir: str | Path,
) -> Path:
    """List of models that failed promotion gates and why."""
    body = [
        f"Run ID: `{run_id}`",
        "",
        "## Failed Models",
    ]

    if not failed_models:
        body.append("_No models failed promotion gates._")
    else:
        for model_name, reason in zip(failed_models, reasons):
            body.append(f"- **{model_name}**: {reason}")

    path = Path(save_dir) / f"{run_id}_failure_modes.md"
    return _write_md(path, f"{run_id} Failure Modes", body)


def write_next_research_queue_md(
    run_id: str,
    passed_models: list[str],
    gaps: list[str],
    save_dir: str | Path,
) -> Path:
    """Recommended next steps, model gaps, and data improvements."""
    body = [
        f"Run ID: `{run_id}`",
        "",
        "## Passed Models",
    ]

    if not passed_models:
        body.append("_No models passed all promotion gates._")
    else:
        for model in passed_models:
            body.append(f"- {model}")

    body.extend([
        "",
        "## Recommended Next Steps",
    ])

    if not gaps:
        body.append("_No specific gaps identified._")
    else:
        for gap in gaps:
            body.append(f"- {gap}")

    path = Path(save_dir) / f"{run_id}_next_research_queue.md"
    return _write_md(path, f"{run_id} Next Research Queue", body)
