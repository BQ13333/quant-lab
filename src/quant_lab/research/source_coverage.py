from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


STRATEGY_KEYWORDS: tuple[str, ...] = (
    "strategy",
    "strategies",
    "alpha",
    "alphas",
    "factor",
    "factors",
    "momentum",
    "reversion",
    "mean-reversion",
    "trend",
    "breakout",
    "channel",
    "moving average",
    "crossover",
    "pairs",
    "statistical arbitrage",
    "value",
    "quality",
    "volatility",
    "low-volatility",
    "carry",
    "skew",
    "regression",
    "machine learning",
    "portfolio",
    "optimization",
    "risk",
    "backtest",
    "overfit",
    "execution",
    "market-making",
    "microstructure",
)

DATA_BLOCKED_KEYWORDS: tuple[str, ...] = (
    "earnings",
    "fundamental",
    "fundamentals",
    "analyst",
    "revision",
    "revisions",
    "sentiment",
    "news",
    "implied volatility",
    "options",
    "option",
    "intraday",
    "order book",
    "bid-ask",
    "volume",
    "short interest",
    "ownership",
)


@dataclass(frozen=True)
class MarkdownHeading:
    file: str
    line_number: int
    level: int
    heading: str
    category: str
    matched_keywords: tuple[str, ...]


DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = ("quant_lab", ".ocr_cache")


def _iter_markdown_files(root: Path, exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS) -> Iterable[Path]:
    for path in root.rglob("*.md"):
        if any(part.startswith(".") for part in path.parts):
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in exclude_dirs for part in rel_parts):
            continue
        yield path


def _classify_heading(heading: str) -> tuple[str, tuple[str, ...]]:
    lower = heading.lower()
    matched = tuple(keyword for keyword in STRATEGY_KEYWORDS if keyword in lower)
    data_matched = tuple(keyword for keyword in DATA_BLOCKED_KEYWORDS if keyword in lower)
    if matched and data_matched:
        return "strategy_or_model_requires_data", tuple(sorted(set(matched + data_matched)))
    if matched:
        return "strategy_or_model", matched
    if data_matched:
        return "data_dependency", data_matched
    return "general", tuple()


def mine_markdown_headings(
    source_root: str | Path,
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS,
) -> pd.DataFrame:
    """Inventory headings from markdown quant source files.

    This intentionally records headings and line numbers only. It does not copy
    book text into artifacts.
    """

    root = Path(source_root)
    rows: list[MarkdownHeading] = []
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    for path in _iter_markdown_files(root, exclude_dirs=exclude_dirs):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(root))
        for line_number, line in enumerate(text.splitlines(), start=1):
            match = heading_pattern.match(line)
            if not match:
                continue
            marker, heading = match.groups()
            category, keywords = _classify_heading(heading)
            rows.append(
                MarkdownHeading(
                    file=rel,
                    line_number=line_number,
                    level=len(marker),
                    heading=heading.strip(),
                    category=category,
                    matched_keywords=keywords,
                )
            )
    return pd.DataFrame(
        [
            {
                "file": row.file,
                "line_number": row.line_number,
                "level": row.level,
                "heading": row.heading,
                "category": row.category,
                "matched_keywords": ",".join(row.matched_keywords),
            }
            for row in rows
        ]
    )


def _file_scope(path: Path) -> str:
    parts = path.parts
    if "Quant_Lib" in parts:
        return "quant_lib_book"
    return "root_book"


def _heading_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^#{1,6}\s+", line))


def markdown_file_inventory(
    source_root: str | Path,
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS,
) -> pd.DataFrame:
    root = Path(source_root)
    rows = []
    for path in _iter_markdown_files(root, exclude_dirs=exclude_dirs):
        try:
            stat = path.stat()
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(root)
        rows.append(
            {
                "file": str(rel),
                "scope": _file_scope(rel),
                "bytes": int(stat.st_size),
                "line_count": len(text.splitlines()),
                "content_hash": hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16],
                "heading_count": _heading_count(text),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)


def _section_tokens(section: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z-]{2,}", section.lower())
    stopwords = {"strategy", "strategies", "section", "chapter", "with", "and", "the"}
    return {token for token in tokens if token not in stopwords}


def source_strategy_spec_coverage(source_root: str | Path, headings: pd.DataFrame) -> pd.DataFrame:
    from quant_lab.models.source_strategies import build_source_strategy_specs

    root = Path(source_root)
    rows = []
    for spec in build_source_strategy_specs():
        source_path = root / spec.source_file
        file_exists = source_path.exists()
        file_headings = headings[headings["file"] == spec.source_file].copy() if not headings.empty else pd.DataFrame()
        matched_line = None
        matched_heading = ""
        match_method = "none"
        if not file_headings.empty:
            section_lower = spec.source_section.lower()
            exact = file_headings[file_headings["heading"].str.lower() == section_lower]
            if not exact.empty:
                match = exact.iloc[0]
                matched_line = int(match["line_number"])
                matched_heading = str(match["heading"])
                match_method = "exact_heading"
            else:
                tokens = _section_tokens(spec.source_section)
                best_score = 0
                best_row = None
                for row in file_headings.itertuples(index=False):
                    heading_tokens = _section_tokens(str(row.heading))
                    score = len(tokens & heading_tokens)
                    if score > best_score:
                        best_score = score
                        best_row = row
                if best_row is not None and best_score > 0:
                    matched_line = int(best_row.line_number)
                    matched_heading = str(best_row.heading)
                    match_method = "token_overlap"

        if not file_exists:
            coverage_status = "missing_source_file"
            gap_reason = "source_file_not_found"
        elif matched_line is None:
            coverage_status = "unmatched_section"
            gap_reason = "source_section_not_matched_to_heading"
        elif spec.implementation_status == "implemented":
            coverage_status = "implemented"
            gap_reason = ""
        else:
            coverage_status = "blocked"
            gap_reason = spec.implementation_status

        rows.append(
            {
                "strategy_id": spec.strategy_id,
                "family": spec.family,
                "source_file": spec.source_file,
                "source_section": spec.source_section,
                "implementation_status": spec.implementation_status,
                "file_exists": file_exists,
                "matched_heading_line": matched_line,
                "matched_heading": matched_heading,
                "match_method": match_method,
                "coverage_status": coverage_status,
                "gap_reason": gap_reason,
            }
        )
    return pd.DataFrame(rows)


def source_file_coverage(
    files: pd.DataFrame,
    strategy_headings: pd.DataFrame,
    spec_coverage: pd.DataFrame,
) -> pd.DataFrame:
    heading_counts = strategy_headings.groupby("file").size().rename("strategy_heading_count")
    mapped_counts = spec_coverage[spec_coverage["file_exists"]].groupby("source_file").size().rename("mapped_spec_count")
    implemented_counts = (
        spec_coverage[spec_coverage["coverage_status"] == "implemented"].groupby("source_file").size().rename("implemented_spec_count")
    )
    blocked_counts = spec_coverage[spec_coverage["coverage_status"] == "blocked"].groupby("source_file").size().rename("blocked_spec_count")
    result = files[["file", "scope", "bytes", "line_count", "heading_count"]].copy()
    result = result.join(heading_counts, on="file")
    result = result.join(mapped_counts, on="file")
    result = result.join(implemented_counts, on="file")
    result = result.join(blocked_counts, on="file")
    for col in ("strategy_heading_count", "mapped_spec_count", "implemented_spec_count", "blocked_spec_count"):
        result[col] = result[col].fillna(0).astype(int)
    result["uncovered_strategy_heading_count"] = (
        result["strategy_heading_count"] - result["mapped_spec_count"]
    ).clip(lower=0)
    return result.sort_values(
        ["uncovered_strategy_heading_count", "strategy_heading_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def write_markdown_source_coverage(
    source_root: str | Path,
    artifact_root: str | Path = "artifacts/source_coverage",
    run_id: str | None = None,
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS,
) -> Path:
    run_id = run_id or datetime.now(timezone.utc).strftime("SOURCE-COVERAGE-%Y%m%d-%H%M%S")
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    files = markdown_file_inventory(source_root, exclude_dirs=exclude_dirs)
    headings = mine_markdown_headings(source_root, exclude_dirs=exclude_dirs)
    strategy_headings = headings[headings["category"].isin(["strategy_or_model", "strategy_or_model_requires_data"])].copy()
    data_dependencies = headings[headings["category"].isin(["data_dependency", "strategy_or_model_requires_data"])].copy()
    spec_coverage = source_strategy_spec_coverage(source_root, headings)
    file_coverage = source_file_coverage(files, strategy_headings, spec_coverage)

    files.to_csv(artifact_dir / "markdown_files.csv", index=False)
    headings.to_csv(artifact_dir / "heading_inventory.csv", index=False)
    strategy_headings.to_csv(artifact_dir / "strategy_heading_inventory.csv", index=False)
    data_dependencies.to_csv(artifact_dir / "data_dependency_heading_inventory.csv", index=False)
    spec_coverage.to_csv(artifact_dir / "source_strategy_spec_coverage.csv", index=False)
    file_coverage.to_csv(artifact_dir / "source_file_coverage.csv", index=False)

    manifest = {
        "run_id": run_id,
        "source_root": str(Path(source_root).resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "markdown_files": int(len(files)),
        "headings": int(len(headings)),
        "strategy_or_model_headings": int(len(strategy_headings)),
        "data_dependency_headings": int(len(data_dependencies)),
        "source_strategy_specs": int(len(spec_coverage)),
        "implemented_source_strategy_specs": int((spec_coverage["coverage_status"] == "implemented").sum()),
        "blocked_source_strategy_specs": int((spec_coverage["coverage_status"] == "blocked").sum()),
        "exclude_dirs": list(exclude_dirs),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Heading inventory only; no long copyrighted source passages are copied.",
    }
    (artifact_dir / "source_coverage_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_source_coverage_summary(artifact_dir, manifest, file_coverage, spec_coverage)
    return artifact_dir


def _write_source_coverage_summary(
    artifact_dir: Path,
    manifest: dict[str, object],
    file_coverage: pd.DataFrame,
    spec_coverage: pd.DataFrame,
) -> None:
    top_uncovered = file_coverage.head(15)
    status_counts = spec_coverage["coverage_status"].value_counts().to_dict() if not spec_coverage.empty else {}
    lines = [
        "# Source Coverage Summary",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- Source root: `{manifest['source_root']}`",
        f"- Markdown files: `{manifest['markdown_files']}`",
        f"- Headings: `{manifest['headings']}`",
        f"- Strategy/model headings: `{manifest['strategy_or_model_headings']}`",
        f"- Data dependency headings: `{manifest['data_dependency_headings']}`",
        f"- Source strategy specs: `{manifest['source_strategy_specs']}`",
        f"- Implemented specs: `{manifest['implemented_source_strategy_specs']}`",
        f"- Blocked specs: `{manifest['blocked_source_strategy_specs']}`",
        "",
        "## Spec Coverage",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: `{count}`")
    lines.extend(["", "## Top Uncovered Source Files", ""])
    if top_uncovered.empty:
        lines.append("No uncovered strategy headings found.")
    else:
        lines.append("| File | Strategy Headings | Mapped Specs | Uncovered |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in top_uncovered.itertuples(index=False):
            lines.append(
                f"| `{row.file}` | {row.strategy_heading_count} | {row.mapped_spec_count} | "
                f"{row.uncovered_strategy_heading_count} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This inventory records headings and line numbers only; it does not copy long source passages.",
            "- Uncovered heading counts are triage inputs, not proof that every heading is an implementable strategy.",
            "- Data-blocked strategy specs should stay blocked until point-in-time data exists.",
        ]
    )
    (artifact_dir / "source_coverage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
