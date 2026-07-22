from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from quant_lab.agents.japan_data_quality_loop import default_japan_data_quality_paths
from quant_lab.data.japan_g5_decision_calendar_audit import (
    JapanG5AuditConfig,
    JapanG5Input,
    write_japan_g5_decision_calendar_audit,
)


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce the fail-closed Japan G5 true decision-calendar audit."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--artifact-dir",
        help="Exact write-once run directory; it must not already exist.",
    )
    parser.add_argument("--commands-log", default="commands.log")

    parser.add_argument("--decision-calendar", required=True)
    parser.add_argument("--decision-manifest", required=True)
    parser.add_argument("--decision-artifact-key", default="decision_calendar")
    parser.add_argument("--listing-windows", required=True)
    parser.add_argument("--listing-manifest", required=True)
    parser.add_argument("--listing-artifact-key", default="listing_windows")
    parser.add_argument("--feature-observations", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--feature-artifact-key", default="feature_observations")

    parser.add_argument(
        "--features",
        required=True,
        help="Comma-separated frozen feature names; every feature must cover every decision row.",
    )
    parser.add_argument("--purge-days", type=int, required=True)
    parser.add_argument("--embargo-days", type=int, required=True)
    parser.add_argument("--train-end", required=True, help="Timezone-aware timestamp.")
    parser.add_argument("--test-start", required=True, help="Timezone-aware timestamp.")
    parser.add_argument("--test-end", required=True, help="Timezone-aware timestamp.")
    parser.add_argument("--holdout-start", required=True, help="Timezone-aware timestamp.")
    parser.add_argument("--decision-schedule", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = _parser().parse_args(raw_argv)
    project_root = Path(args.project_root).resolve()
    paths = default_japan_data_quality_paths(project_root)
    default_artifact_dir = paths.leakage_qa_dir
    if args.artifact_dir:
        artifact_dir = _resolve(args.artifact_dir, project_root)
    elif default_artifact_dir is not None and default_artifact_dir.name == args.run_id:
        artifact_dir = default_artifact_dir
    elif default_artifact_dir is not None:
        artifact_dir = default_artifact_dir.parent / args.run_id
    else:
        artifact_dir = None
    if artifact_dir is None:
        raise ValueError("artifact directory could not be resolved")
    commands_log = _resolve(args.commands_log, project_root)
    paths = replace(
        paths,
        leakage_qa_dir=artifact_dir,
        immutable_hashes_path=artifact_dir / "immutable_hashes.json",
        commands_log=commands_log,
    )
    features = tuple(feature.strip() for feature in args.features.split(",") if feature.strip())
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *raw_argv])
    result = write_japan_g5_decision_calendar_audit(
        paths=paths,
        run_id=args.run_id,
        decision_calendar=JapanG5Input(
            path=_resolve(args.decision_calendar, project_root),
            manifest_path=_resolve(args.decision_manifest, project_root),
            artifact_key=args.decision_artifact_key,
        ),
        listing_windows=JapanG5Input(
            path=_resolve(args.listing_windows, project_root),
            manifest_path=_resolve(args.listing_manifest, project_root),
            artifact_key=args.listing_artifact_key,
        ),
        feature_observations=JapanG5Input(
            path=_resolve(args.feature_observations, project_root),
            manifest_path=_resolve(args.feature_manifest, project_root),
            artifact_key=args.feature_artifact_key,
        ),
        config=JapanG5AuditConfig(
            required_features=features,
            purge_days=args.purge_days,
            embargo_days=args.embargo_days,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
            holdout_start=args.holdout_start,
            decision_schedule=args.decision_schedule,
        ),
        command=command,
    )
    print(f"Status: {result.manifest['status']}")
    print(f"Scorecard eligible: {result.manifest['scorecard_eligible']}")
    print(f"Decision rows: {result.manifest['decision_rows']}")
    print(f"Leakage violations: {result.manifest['total_leakage_violations']}")
    print(f"Artifact dir: {result.artifact_dir}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Immutable inventory: {result.immutable_hashes_path}")
    if not result.manifest["scorecard_eligible"]:
        for blocker in result.manifest["critical_blockers"]:
            print(f"BLOCKER: {blocker}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
