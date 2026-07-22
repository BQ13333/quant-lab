from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.portfolio.costs import (
    capacity_score,
    linear_transaction_cost,
    turnover_decomposition,
)
from quant_lab.validation.gates import GateDecision, ValidationGate
from quant_lab.validation.metrics import max_drawdown, sharpe_ratio
from quant_lab.validation.stress import (
    cost_stress,
    liquidity_stress,
    monthly_stability,
    no_single_country_dominance,
    no_single_month_dominance,
)


def _save(df: pd.DataFrame, save_dir: Path | None, filename: str) -> None:
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_dir / filename, index=False)


def generate_leaderboard(
    tournament_results: pd.DataFrame,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """Return net results after costs, sorted by deflated_sharpe, sharpe."""
    if tournament_results.empty:
        _save(pd.DataFrame(), save_dir, "leaderboard.csv")
        return pd.DataFrame()

    required = {"model_name", "sharpe", "deflated_sharpe"}
    if not required.issubset(tournament_results.columns):
        missing = required - set(tournament_results.columns)
        raise ValueError(f"leaderboard missing required columns: {missing}")

    result = tournament_results.sort_values(
        ["deflated_sharpe", "sharpe"], ascending=False
    ).reset_index(drop=True)
    _save(result, save_dir, "leaderboard.csv")
    return result


def generate_diagnostics_summary(
    leaderboard: pd.DataFrame,
    benchmarks: pd.DataFrame | None = None,
    save_dir: Path | None = None,
    weights_dict: dict[str, pd.DataFrame] | None = None,
    group_map: pd.Series | None = None,
) -> pd.DataFrame:
    """Merge leaderboard with benchmarks and add pass/fail flags for each gate."""
    if leaderboard.empty:
        _save(pd.DataFrame(), save_dir, "diagnostics_summary.csv")
        return pd.DataFrame()

    gate = ValidationGate()
    frames = [leaderboard.copy()]
    if benchmarks is not None and not benchmarks.empty:
        frames.append(benchmarks)

    merged = pd.concat(frames, ignore_index=True)

    # Basic gate evaluation
    gate_passed = []
    gate_reasons = []
    promote_passed = []
    promote_reasons = []

    for _, row in merged.iterrows():
        dsr = float(row.get("deflated_sharpe", 0.0))
        decision = gate.evaluate(
            sharpe=float(row.get("sharpe", 0.0)),
            max_drawdown=float(row.get("max_drawdown", 0.0)),
            turnover=float(row.get("turnover", 0.0)),
            deflated_sharpe_probability=dsr,
        )
        gate_passed.append(decision.passed)
        gate_reasons.append("; ".join(decision.reasons) if decision.reasons else "")

        # Promotion gate if returns/weights available
        model_name = row.get("model_name", "")
        returns = None
        if weights_dict is not None and model_name in weights_dict:
            returns = weights_dict[model_name]
        if returns is not None and not returns.empty:
            promote_row = row.to_dict()
            promote_row["returns"] = returns
            promote_decision = gate.promote(
                promote_row, weights=weights_dict.get(model_name) if weights_dict else None, group_map=group_map
            )
            promote_passed.append(promote_decision.passed)
            promote_reasons.append("; ".join(promote_decision.reasons) if promote_decision.reasons else "")
        else:
            promote_passed.append(False)
            promote_reasons.append("insufficient data")

    merged["gate_passed"] = gate_passed
    merged["gate_reasons"] = gate_reasons
    merged["promote_passed"] = promote_passed
    merged["promote_reasons"] = promote_reasons

    _save(merged, save_dir, "diagnostics_summary.csv")
    return merged


def generate_cost_sensitivity(
    returns: pd.Series | pd.DataFrame,
    weights: pd.DataFrame | None,
    cost_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 25.0, 50.0),
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """Wrapper around cost_stress, formatted for CSV."""
    if isinstance(returns, pd.DataFrame):
        # Assume portfolio returns are already computed or compute from first column
        series = returns.iloc[:, 0] if returns.shape[1] > 0 else pd.Series(dtype=float)
    else:
        series = returns

    result = cost_stress(series, weights=weights, cost_grid=cost_grid)
    _save(result, save_dir, "cost_sensitivity.csv")
    return result


def generate_country_attribution(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    group_map: pd.Series,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """P&L attribution by country: contribution, weight, return, selection effect, allocation effect."""
    if returns.empty or weights.empty or group_map.empty:
        _save(pd.DataFrame(), save_dir, "country_attribution.csv")
        return pd.DataFrame()

    aligned_weights = weights.reindex(columns=returns.columns).fillna(0.0)
    aligned_returns = returns.reindex(index=weights.index).fillna(0.0)

    # Equal-weight benchmark across all assets
    overall_equal_return = aligned_returns.mean(axis=1).mean()
    n_countries = len(group_map.dropna().unique())
    benchmark_weight = 1.0 / n_countries if n_countries > 0 else 0.0

    rows = []
    for country in sorted(group_map.dropna().unique()):
        country_assets = group_map[group_map == country].index
        common_assets = [a for a in country_assets if a in aligned_returns.columns]
        if not common_assets:
            continue

        w = aligned_weights[common_assets]
        r = aligned_returns[common_assets]

        # Daily contribution, then average
        contribution = (w * r).sum(axis=1).mean()
        avg_weight = w.abs().sum(axis=1).mean()
        avg_return = r.mean(axis=1).mean()  # equal-weight within country

        # Simplified Brinson-style attribution
        selection_effect = avg_weight * (avg_return - overall_equal_return)
        allocation_effect = (avg_weight - benchmark_weight) * (avg_return - overall_equal_return)

        rows.append(
            {
                "country": country,
                "contribution": contribution,
                "avg_weight": avg_weight,
                "avg_return": avg_return,
                "selection_effect": selection_effect,
                "allocation_effect": allocation_effect,
            }
        )

    df = pd.DataFrame(rows)
    _save(df, save_dir, "country_attribution.csv")
    return df


def generate_monthly_attribution(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 5.0,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """Month-by-month return, volatility, drawdown, turnover, cost."""
    if returns.empty or weights.empty:
        _save(pd.DataFrame(), save_dir, "monthly_attribution.csv")
        return pd.DataFrame()

    portfolio_returns = (
        weights.shift(1).fillna(0.0) * returns.reindex(weights.index)
    ).sum(axis=1)

    # Ensure DatetimeIndex
    if not hasattr(portfolio_returns.index, "month"):
        portfolio_returns.index = pd.date_range(
            start="2000-01-01", periods=len(portfolio_returns), freq="B"
        )

    monthly = portfolio_returns.groupby(
        [portfolio_returns.index.year, portfolio_returns.index.month]
    )

    rows = []
    for (year, month), group in monthly:
        month_weights = weights.loc[group.index] if not group.empty else weights.iloc[:0]

        turnover = (
            month_weights.diff().abs().sum(axis=1).mean()
            if not month_weights.empty
            else 0.0
        )
        cost = (
            linear_transaction_cost(month_weights, cost_bps).mean()
            if not month_weights.empty
            else 0.0
        )
        total_return = group.sum()
        vol = (
            group.std(ddof=1) * (252**0.5)
            if len(group) > 1
            else 0.0
        )
        dd = max_drawdown(group)

        rows.append(
            {
                "year": year,
                "month": month,
                "return": total_return,
                "volatility": vol,
                "drawdown": dd,
                "turnover": turnover,
                "cost": cost,
            }
        )

    df = pd.DataFrame(rows)
    _save(df, save_dir, "monthly_attribution.csv")
    return df


def generate_fx_attribution(
    returns: pd.DataFrame,
    fx_returns: pd.Series | pd.DataFrame,
    weights: pd.DataFrame,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """Contribution of FX moves vs local returns."""
    if returns.empty or weights.empty:
        _save(pd.DataFrame(), save_dir, "fx_attribution.csv")
        return pd.DataFrame()

    aligned_weights = weights.reindex(columns=returns.columns).fillna(0.0)
    aligned_returns = returns.reindex(index=weights.index).fillna(0.0)

    # Ensure fx_returns is a DataFrame aligned with returns
    if isinstance(fx_returns, pd.Series):
        fx = pd.DataFrame(
            {
                col: fx_returns.reindex(aligned_returns.index).fillna(0.0)
                for col in aligned_returns.columns
            },
            index=aligned_returns.index,
        )
    else:
        fx = fx_returns.reindex(
            index=aligned_returns.index, columns=aligned_returns.columns
        ).fillna(0.0)

    local_returns = aligned_returns - fx

    # Per-asset mean contributions
    fx_contribution = (aligned_weights * fx).mean()
    local_contribution = (aligned_weights * local_returns).mean()
    total_contribution = (aligned_weights * aligned_returns).mean()

    df = pd.DataFrame(
        {
            "asset": aligned_returns.columns,
            "fx_contribution": fx_contribution.values,
            "local_contribution": local_contribution.values,
            "total_contribution": total_contribution.values,
        }
    )
    _save(df, save_dir, "fx_attribution.csv")
    return df


def generate_turnover_report(
    weights: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """Turnover decomposition by model, country, month, asset (top 10 contributors)."""
    if weights.empty:
        _save(pd.DataFrame(), save_dir, "turnover_report.csv")
        return pd.DataFrame()

    decomp = turnover_decomposition(weights, metadata=metadata)
    turnover = weights.diff().abs().fillna(weights.abs())
    total_by_asset = turnover.sum().sort_values(ascending=False)

    rows = []

    # Overall
    rows.append(
        {
            "category": "overall",
            "subcategory": "all",
            "mean_turnover": decomp["total_turnover"].mean(),
            "max_turnover": decomp["total_turnover"].max(),
            "total_turnover": decomp["total_turnover"].sum(),
        }
    )

    # By month
    by_month = (
        decomp.groupby("by_month")["total_turnover"]
        .agg(["mean", "max", "sum"])
        .reset_index()
    )
    for _, row in by_month.iterrows():
        rows.append(
            {
                "category": "month",
                "subcategory": str(row["by_month"]),
                "mean_turnover": row["mean"],
                "max_turnover": row["max"],
                "total_turnover": row["sum"],
            }
        )

    # By country
    if metadata is not None and "country" in metadata.columns:
        for country in sorted(metadata["country"].dropna().unique()):
            assets = metadata[metadata["country"] == country].index
            common = [a for a in assets if a in weights.columns]
            if common:
                country_turnover = turnover[common].sum(axis=1)
                rows.append(
                    {
                        "category": "country",
                        "subcategory": country,
                        "mean_turnover": country_turnover.mean(),
                        "max_turnover": country_turnover.max(),
                        "total_turnover": country_turnover.sum(),
                    }
                )

    # By model
    if isinstance(weights.columns, pd.MultiIndex):
        for model in sorted(weights.columns.get_level_values(0).unique()):
            model_assets = weights.columns[weights.columns.get_level_values(0) == model]
            model_turnover = turnover[model_assets].sum(axis=1)
            rows.append(
                {
                    "category": "model",
                    "subcategory": model,
                    "mean_turnover": model_turnover.mean(),
                    "max_turnover": model_turnover.max(),
                    "total_turnover": model_turnover.sum(),
                }
            )
    elif metadata is not None and "model" in metadata.columns:
        for model in sorted(metadata["model"].dropna().unique()):
            assets = metadata[metadata["model"] == model].index
            common = [a for a in assets if a in weights.columns]
            if common:
                model_turnover = turnover[common].sum(axis=1)
                rows.append(
                    {
                        "category": "model",
                        "subcategory": model,
                        "mean_turnover": model_turnover.mean(),
                        "max_turnover": model_turnover.max(),
                        "total_turnover": model_turnover.sum(),
                    }
                )

    # Top 10 asset contributors
    for asset, tot in total_by_asset.head(10).items():
        rows.append(
            {
                "category": "top_asset",
                "subcategory": asset,
                "mean_turnover": tot / len(turnover),
                "max_turnover": turnover[asset].max(),
                "total_turnover": tot,
            }
        )

    df = pd.DataFrame(rows)
    _save(df, save_dir, "turnover_report.csv")
    return df


def generate_liquidity_report(
    weights: pd.DataFrame,
    volumes: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """ADV utilization, participation, and capacity constraints."""
    if weights.empty or volumes.empty:
        _save(pd.DataFrame(), save_dir, "liquidity_report.csv")
        return pd.DataFrame()

    adv = volumes.mean()
    capital = 1e8  # $100M default

    dollar_trade = weights.diff().abs().mul(capital).fillna(0.0)
    aligned_adv = adv.reindex(weights.columns).replace(0.0, pd.NA)
    participation = dollar_trade.div(aligned_adv, axis=1)

    max_participation = participation.max()
    avg_participation = participation.mean()

    rows = []
    for asset in weights.columns:
        rows.append(
            {
                "asset": asset,
                "adv": float(aligned_adv.get(asset, np.nan)),
                "max_participation": float(max_participation.get(asset, np.nan)),
                "avg_participation": float(avg_participation.get(asset, np.nan)),
                "capacity_score": float(
                    capacity_score(weights, adv, capital).max()
                    if not weights.empty
                    else np.nan
                ),
            }
        )

    # Country aggregation
    if metadata is not None and "country" in metadata.columns:
        for country in sorted(metadata["country"].dropna().unique()):
            assets = metadata[metadata["country"] == country].index
            common = [a for a in assets if a in weights.columns]
            if common:
                country_adv = adv.reindex(common).sum()
                country_trade = dollar_trade[common].sum(axis=1)
                country_participation = (
                    country_trade / country_adv.replace(0, pd.NA)
                ).mean()
                rows.append(
                    {
                        "asset": f"country:{country}",
                        "adv": float(country_adv),
                        "max_participation": float(country_participation),
                        "avg_participation": float(country_participation),
                        "capacity_score": np.nan,
                    }
                )

    df = pd.DataFrame(rows)
    _save(df, save_dir, "liquidity_report.csv")
    return df


def generate_model_cards(
    leaderboard: pd.DataFrame,
    returns_dict: dict[str, pd.Series],
    weights_dict: dict[str, pd.DataFrame],
    save_dir: Path | None = None,
    group_map: pd.Series | None = None,
    cost_bps: float = 5.0,
) -> dict[str, pd.DataFrame]:
    """One DataFrame per model with summary stats, cost sensitivity, stress results, gate pass/fail."""
    if leaderboard.empty:
        return {}

    gate = ValidationGate()
    cards: dict[str, pd.DataFrame] = {}

    for _, row in leaderboard.iterrows():
        model_name = str(row.get("model_name", ""))
        if not model_name:
            continue

        returns = returns_dict.get(model_name, pd.Series(dtype=float))
        weights = weights_dict.get(model_name, pd.DataFrame())

        summary: dict[str, Any] = {}
        for col in leaderboard.columns:
            summary[col] = row.get(col, np.nan)

        # Cost sensitivity
        if not returns.empty and not weights.empty:
            cs = cost_stress(returns, weights, cost_grid=(0.0, cost_bps, cost_bps * 2))
            for _, cs_row in cs.iterrows():
                summary[f"cost_sharpe_{cs_row['cost_bps']:.0f}bps"] = cs_row["net_sharpe"]
        else:
            for cost in (0.0, cost_bps, cost_bps * 2):
                summary[f"cost_sharpe_{cost:.0f}bps"] = np.nan

        # Monthly stability
        if not returns.empty:
            ms = monthly_stability(returns)
            if not ms.empty:
                for col in ms.columns:
                    summary[f"monthly_{col}"] = ms[col].iloc[0]

        # Gate evaluation
        dsr = float(row.get("deflated_sharpe", 0.0))
        decision = gate.evaluate(
            sharpe=float(row.get("sharpe", 0.0)),
            max_drawdown=float(row.get("max_drawdown", 0.0)),
            turnover=float(row.get("turnover", 0.0)),
            deflated_sharpe_probability=dsr,
        )
        summary["gate_passed"] = decision.passed
        summary["gate_reasons"] = "; ".join(decision.reasons) if decision.reasons else ""

        # Promotion gate
        if not returns.empty:
            promote_row = row.to_dict()
            promote_row["returns"] = returns
            promote_decision = gate.promote(
                promote_row,
                weights=weights if not weights.empty else None,
                group_map=group_map,
            )
            summary["promote_passed"] = promote_decision.passed
            summary["promote_reasons"] = (
                "; ".join(promote_decision.reasons) if promote_decision.reasons else ""
            )
        else:
            summary["promote_passed"] = False
            summary["promote_reasons"] = "insufficient data"

        # Country / month dominance
        if not returns.empty and not weights.empty and group_map is not None and not group_map.empty:
            summary["country_dominance_ok"] = no_single_country_dominance(
                returns, weights, group_map
            )
            summary["month_dominance_ok"] = no_single_month_dominance(returns)
        else:
            summary["country_dominance_ok"] = np.nan
            summary["month_dominance_ok"] = np.nan

        card_df = pd.DataFrame([summary])
        cards[model_name] = card_df

        if save_dir is not None:
            model_dir = Path(save_dir) / "model_cards"
            model_dir.mkdir(parents=True, exist_ok=True)
            card_df.to_csv(model_dir / f"{model_name}.csv", index=False)

    return cards
