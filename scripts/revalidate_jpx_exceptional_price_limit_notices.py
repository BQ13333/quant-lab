from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from quant_lab.data import jpx_exceptional_price_limit_notices as parser_module
from quant_lab.data.jpx_exceptional_price_limit_notices import (
    _load_sessions,
    parse_archived_notice_pages,
    sha256_file,
)


CATEGORIES = (
    "parser_defect",
    "calendar_boundary_artifact",
    "official_source_typo",
    "irreducible_ambiguity",
)

# This inventory is deliberately keyed to the immutable retained corpus. Revalidation fails
# closed if the baseline failure set changes, so a new anomaly cannot inherit an old decision.
STRICT_FAILURE_CLASSIFICATIONS: dict[tuple[str, str], dict[str, str]] = {
    ("JPX-EPL-00074", "3810"): {
        "category": "official_source_typo",
        "evidence": (
            "The page says base price 3,065 yen but labels the 1,400/700 yen range amounts as "
            "absolute upper/lower prices, producing the impossible literal order 1,400 < 3,065."
        ),
    },
    ("JPX-EPL-00324", "8462"): {
        "category": "official_source_typo",
        "evidence": (
            "The only explicit effective date is 'Apirl 7'; the malformed official month has no "
            "independent valid date on the page."
        ),
    },
    ("JPX-EPL-00511", "6658"): {
        "category": "official_source_typo",
        "evidence": (
            "Page/archive date July 22 conflicts with URL July 26 and explicit next business day "
            "July 27; neighboring July 25 and July 27 cancellations confirm stale date metadata."
        ),
    },
    ("JPX-EPL-00530", "4845"): {
        "category": "official_source_typo",
        "evidence": (
            "The August 17 page calls August 17 the next business day; the next admitted session "
            "is August 18."
        ),
    },
    ("JPX-EPL-00541", "3856"): {
        "category": "official_source_typo",
        "evidence": (
            "Page/archive date September 15 is stale: the paired preliminary notice is September "
            "30 and the cancellation URL is October 3."
        ),
    },
    ("JPX-EPL-00583", "1401"): {
        "category": "official_source_typo",
        "evidence": (
            "The January 11 page calls January 11 the next business day; the paired cancellation "
            "is January 12."
        ),
    },
    ("JPX-EPL-00742", "3083"): {
        "category": "official_source_typo",
        "evidence": (
            "The URL slug says February 9, while page/archive date February 28 and the February 27 "
            "preliminary notice establish a February 28 cancellation."
        ),
    },
    ("JPX-EPL-00916", "6838"): {
        "category": "official_source_typo",
        "evidence": (
            "Page/archive date May 22 conflicts with the May 23 URL and follows the paired May 22 "
            "preliminary notice."
        ),
    },
    ("JPX-EPL-01172", "3189"): {
        "category": "official_source_typo",
        "evidence": (
            "Page/archive date December 10 conflicts with URL December 13 and the explicit next "
            "business day December 14."
        ),
    },
    ("JPX-EPL-01308", "6938"): {
        "category": "parser_defect",
        "evidence": (
            "The prose unambiguously says next business day March 28 but omits only the closing "
            "parenthesis before 'as follows'."
        ),
    },
    ("JPX-EPL-01322", "6557"): {
        "category": "official_source_typo",
        "evidence": (
            "Page/archive date June 23 conflicts with URL July 23 and explicit July 24 effective "
            "date; its internally repeated 2x range also conflicts with the post-2020 4x rule."
        ),
    },
    ("JPX-EPL-01322", "7776"): {
        "category": "official_source_typo",
        "evidence": (
            "Page/archive date June 23 conflicts with URL July 23 and the page's explicit July 24 "
            "effective date."
        ),
    },
    ("JPX-EPL-01389", "260A"): {
        "category": "parser_defect",
        "evidence": (
            "The first date says 'May l', but Note 1 independently and unambiguously states the "
            "effective date as May 1."
        ),
    },
    ("JPX-EPL-01509", "584A"): {
        "category": "calendar_boundary_artifact",
        "evidence": (
            "The retained calendar ends June 30, 2026 while the notice explicitly takes effect "
            "July 1, immediately outside the validated calendar boundary."
        ),
    },
    ("JPX-EPL-01509", "7815"): {
        "category": "calendar_boundary_artifact",
        "evidence": (
            "The retained calendar ends June 30, 2026 while the notice explicitly takes effect "
            "July 1, immediately outside the validated calendar boundary."
        ),
    },
}


def build_revalidation_report(
    artifact_dir: str | Path,
    session_calendar_path: str | Path,
) -> dict[str, Any]:
    root = Path(artifact_dir).resolve()
    calendar = Path(session_calendar_path).resolve()
    plan_path = root / "notice_plan.csv"
    state_path = root / "page_state.csv"
    events_path = root / "events.csv"
    quality_path = root / "event_quality.csv"
    parse_quarantine_path = root / "parse_quarantine.csv"

    plan = pd.read_csv(plan_path, dtype=str, keep_default_na=False)
    state = pd.read_csv(state_path, dtype=str, keep_default_na=False)
    baseline_events = pd.read_csv(events_path, dtype=str, keep_default_na=False)
    baseline_quality = pd.read_csv(quality_path, dtype=str, keep_default_na=False)
    baseline_parse_quarantine = pd.read_csv(
        parse_quarantine_path, dtype=str, keep_default_na=False
    )
    _assert_quality_alignment(baseline_events, baseline_quality, "baseline")

    current_events, current_quality, current_parse_quarantine = parse_archived_notice_pages(
        plan,
        state,
        _load_sessions(calendar),
    )
    _assert_quality_alignment(current_events, current_quality, "current")

    baseline = baseline_quality.copy()
    baseline.insert(0, "notice_id", baseline_events["notice_id"])
    current = current_quality.copy()
    current.insert(0, "notice_id", current_events["notice_id"])
    baseline_failures = baseline.loc[baseline["validation_pass"].ne("True")]
    current_failures = current.loc[current["validation_pass"].ne(True)]

    baseline_keys = {
        (str(row["notice_id"]), str(row["security_code"]))
        for row in baseline_failures.to_dict("records")
    }
    expected_keys = set(STRICT_FAILURE_CLASSIFICATIONS)
    if baseline_keys != expected_keys:
        raise RuntimeError(
            "baseline strict-failure keys changed: "
            f"missing={sorted(expected_keys - baseline_keys)}, "
            f"unexpected={sorted(baseline_keys - expected_keys)}"
        )

    current_by_key = {
        (str(row["notice_id"]), str(row["security_code"])): row
        for row in current.to_dict("records")
    }
    classifications = []
    for row in baseline_failures.sort_values(["notice_id", "security_code"]).to_dict("records"):
        key = (str(row["notice_id"]), str(row["security_code"]))
        decision = STRICT_FAILURE_CLASSIFICATIONS[key]
        after = current_by_key[key]
        after_pass = after["validation_pass"] is True
        classifications.append(
            {
                "notice_id": key[0],
                "security_code": key[1],
                "source_url": str(row["source_url"]),
                "category": decision["category"],
                "evidence": decision["evidence"],
                "before_issues": str(row["validation_issues"]),
                "after_status": "pass" if after_pass else "retained_failure",
                "after_issues": str(after["validation_issues"]),
            }
        )

    expected_remaining = {
        key
        for key, decision in STRICT_FAILURE_CLASSIFICATIONS.items()
        if decision["category"] != "parser_defect"
    }
    current_failure_keys = {
        (str(row["notice_id"]), str(row["security_code"]))
        for row in current_failures.to_dict("records")
    }
    if current_failure_keys != expected_remaining:
        raise RuntimeError(
            "post-fix strict-failure keys changed: "
            f"missing={sorted(expected_remaining - current_failure_keys)}, "
            f"unexpected={sorted(current_failure_keys - expected_remaining)}"
        )

    before_categories = Counter(
        decision["category"] for decision in STRICT_FAILURE_CLASSIFICATIONS.values()
    )
    after_categories = Counter(
        STRICT_FAILURE_CLASSIFICATIONS[key]["category"] for key in current_failure_keys
    )
    return {
        "schema_version": "1.0",
        "artifact_run_id": root.name,
        "deterministic": True,
        "source_hashes": {
            "event_quality.csv": sha256_file(quality_path),
            "events.csv": sha256_file(events_path),
            "notice_plan.csv": sha256_file(plan_path),
            "page_state.csv": sha256_file(state_path),
            "parser_module": sha256_file(Path(parser_module.__file__).resolve()),
            "parse_quarantine.csv": sha256_file(parse_quarantine_path),
            "revalidation_helper": sha256_file(Path(__file__).resolve()),
            "session_calendar": sha256_file(calendar),
        },
        "counts": {
            "event_rows_before": int(len(baseline_events)),
            "event_rows_after": int(len(current_events)),
            "page_parse_quarantine_before": int(len(baseline_parse_quarantine)),
            "page_parse_quarantine_after": int(len(current_parse_quarantine)),
            "strict_validation_quarantine_before": int(len(baseline_failures)),
            "strict_validation_quarantine_after": int(len(current_failures)),
            "strict_validation_quarantine_delta": int(
                len(current_failures) - len(baseline_failures)
            ),
        },
        "category_counts_before": {
            category: before_categories[category] for category in CATEGORIES
        },
        "category_counts_after": {
            category: after_categories[category] for category in CATEGORIES
        },
        "classifications": classifications,
    }


def _assert_quality_alignment(
    events: pd.DataFrame,
    quality: pd.DataFrame,
    label: str,
) -> None:
    identity = ["publication_date", "event_type", "security_code", "source_url"]
    if len(events) != len(quality) or not events[identity].reset_index(drop=True).equals(
        quality[identity].reset_index(drop=True)
    ):
        raise RuntimeError(f"{label} event and quality rows are not positionally aligned")


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically revalidate the retained JPX exceptional-limit corpus."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--session-calendar", required=True)
    parser.add_argument(
        "--output-root",
        default="artifacts/jpx_exceptional_price_limit_revalidation",
    )
    parser.add_argument("--run-id", required=True)
    return parser


def write_revalidation_artifact(
    report: dict[str, Any],
    *,
    source_artifact_dir: str | Path,
    output_root: str | Path,
    run_id: str,
) -> Path:
    source = Path(source_artifact_dir).resolve()
    root = Path(output_root).resolve()
    destination = root / run_id
    if destination.exists():
        raise FileExistsError(f"revalidation artifact already exists: {destination}")
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("revalidation output root must be separate from the retained corpus")

    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    try:
        report_path = staging / "strict_revalidation_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "artifact_type": "jpx_exceptional_price_limit_revalidation",
            "run_id": run_id,
            "source_artifact_dir": str(source),
            "status": "complete",
            "scorecard_eligible": False,
            "promotion_eligible": False,
            "output_sha256": {
                "strict_revalidation_report.json": sha256_file(report_path),
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    report = build_revalidation_report(args.artifact_dir, args.session_calendar)
    output_dir = write_revalidation_artifact(
        report,
        source_artifact_dir=args.artifact_dir,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
