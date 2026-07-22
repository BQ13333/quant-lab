from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from quant_lab.models.signal_factory import (
    SignalFactoryConfig,
    build_signal_models,
    signal_manifest,
)
from quant_lab.notes.obsidian import ObsidianNote, write_note
from quant_lab.research.global_markets import GlobalResearchResult, run_global_research_from_prices
from quant_lab.validation.gates import ValidationGate


ResearchUniverse = Literal["global_etf_proxy", "asean_equities"]


@dataclass(frozen=True)
class ResearchLoopConfig:
    run_id: str
    universe: ResearchUniverse = "global_etf_proxy"
    signal_profile: Literal["lite", "slow", "full"] = "lite"
    baseline_cost_bps: float = 5.0
    max_candidates_for_promotion: int = 20
    status: str = "research_only"


@dataclass(frozen=True)
class ResearchLoopResult:
    config: ResearchLoopConfig
    artifact_dir: Path
    research_result: GlobalResearchResult
    promotion_decisions: pd.DataFrame
    signal_manifest: pd.DataFrame
    manifest_path: Path
    note_path: Path


def evaluate_promotion_decisions(
    leaderboard: pd.DataFrame,
    gate: ValidationGate | None = None,
    max_candidates: int = 20,
) -> pd.DataFrame:
    """Evaluate leaderboard rows against promotion gates."""

    gate = gate or ValidationGate()
    if leaderboard.empty:
        return pd.DataFrame(columns=["rank", "model_name", "promoted", "reasons"])

    if "weighting" in leaderboard.columns:
        candidates = leaderboard[leaderboard["weighting"] != "benchmark"].copy()
    else:
        candidates = leaderboard.copy()

    rows = []
    for rank, (_, row) in enumerate(candidates.head(max_candidates).iterrows(), start=1):
        decision = gate.promote(row.to_dict())
        rows.append(
            {
                "rank": rank,
                "model_name": row.get("model_name", ""),
                "promoted": decision.passed,
                "reasons": "; ".join(decision.reasons),
                "sharpe": row.get("sharpe"),
                "deflated_sharpe": row.get("deflated_sharpe"),
                "annual_excess_return": row.get("annual_excess_return"),
                "information_ratio": row.get("information_ratio"),
                "max_drawdown": row.get("max_drawdown"),
                "turnover": row.get("turnover"),
                "cost_bps": row.get("cost_bps"),
            }
        )
    return pd.DataFrame(rows)


def run_global_research_loop_from_prices(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    artifact_root: str | Path,
    vault_root: str | Path = "vault",
    config: ResearchLoopConfig | None = None,
    cost_bps_grid: tuple[float, ...] = (0.0, 5.0, 15.0, 50.0),
    train_size: int = 504,
    test_size: int = 63,
    embargo: int = 5,
    min_rows: int = 756,
) -> ResearchLoopResult:
    """Run one persistent global research loop from an existing price panel."""

    config = config or ResearchLoopConfig(
        run_id=datetime.now(timezone.utc).strftime("LOOP-GLOBAL-%Y%m%d-%H%M%S")
    )
    artifact_dir = Path(artifact_root) / config.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    group_map = metadata.set_index("ticker")["region"] if "region" in metadata.columns else None
    models = build_signal_models(
        SignalFactoryConfig(
            prefix=f"{config.universe}_{config.signal_profile}",
            profile=config.signal_profile,
            include_group_models=group_map is not None,
            groups=group_map,
        )
    )
    model_manifest = signal_manifest(models)
    model_manifest.to_csv(artifact_dir / "signal_manifest.csv", index=False)

    research = run_global_research_from_prices(
        prices=prices,
        metadata=metadata,
        artifact_dir=artifact_dir,
        vault_root=vault_root,
        run_id=config.run_id,
        cost_bps_grid=cost_bps_grid,
        baseline_cost_bps=config.baseline_cost_bps,
        train_size=train_size,
        test_size=test_size,
        embargo=embargo,
        min_rows=min_rows,
    )

    promotion_decisions = evaluate_promotion_decisions(
        research.leaderboard,
        max_candidates=config.max_candidates_for_promotion,
    )
    promotion_decisions.to_csv(artifact_dir / "promotion_decisions.csv", index=False)

    manifest = {
        "config": asdict(config),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir.resolve()),
        "leaderboard_rows": int(len(research.leaderboard)),
        "cost_sensitivity_rows": int(len(research.cost_sensitivity)),
        "promoted_count": int(promotion_decisions["promoted"].sum()) if not promotion_decisions.empty else 0,
        "signal_count": int(len(model_manifest)),
        "note_paths": [str(path) for path in research.note_paths],
    }
    manifest_path = artifact_dir / "loop_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    note_path = _write_loop_note(
        vault_root=Path(vault_root),
        config=config,
        artifact_dir=artifact_dir,
        promotion_decisions=promotion_decisions,
        model_manifest=model_manifest,
        manifest_path=manifest_path,
    )

    return ResearchLoopResult(
        config=config,
        artifact_dir=artifact_dir,
        research_result=research,
        promotion_decisions=promotion_decisions,
        signal_manifest=model_manifest,
        manifest_path=manifest_path,
        note_path=note_path,
    )


def _write_loop_note(
    vault_root: Path,
    config: ResearchLoopConfig,
    artifact_dir: Path,
    promotion_decisions: pd.DataFrame,
    model_manifest: pd.DataFrame,
    manifest_path: Path,
) -> Path:
    promoted = promotion_decisions[promotion_decisions["promoted"]] if not promotion_decisions.empty else pd.DataFrame()
    failed = promotion_decisions[~promotion_decisions["promoted"]] if not promotion_decisions.empty else pd.DataFrame()

    body = [
        f"Run id: `{config.run_id}`",
        f"Universe: `{config.universe}`",
        f"Signal profile: `{config.signal_profile}`",
        f"Artifact directory: `{artifact_dir.resolve()}`",
        f"Manifest: `{manifest_path.resolve()}`",
        "",
        "## Signal Manifest",
        _markdown_table(model_manifest.head(40)),
        "",
        "## Promotion Summary",
        f"Promoted: `{len(promoted)}`",
        f"Rejected: `{len(failed)}`",
        "",
        "## Promotion Decisions",
        _markdown_table(promotion_decisions.head(40)),
        "",
        "## Operating Rule",
        "Research-loop output may feed paper trading only after human review. Live trading remains blocked.",
    ]

    return write_note(
        vault_root,
        Path("30_experiments") / config.run_id / "research_loop.md",
        ObsidianNote(
            title=f"{config.run_id} Research Loop",
            note_type="research_loop",
            tags=["quant-lab", "research-loop", config.universe, config.signal_profile],
            metadata={"experiment_id": config.run_id, "artifact_dir": str(artifact_dir.resolve())},
            body="\n".join(body),
        ),
    )


def _markdown_table(frame: pd.DataFrame, max_rows: int = 60) -> str:
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).fillna("")
    columns = list(clipped.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in clipped.iterrows()
    ]
    return "\n".join([header, separator, *rows])
