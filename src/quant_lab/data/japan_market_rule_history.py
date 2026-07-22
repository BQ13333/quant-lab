from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pypdf import PdfReader

from quant_lab.data.jpx_market_rules_public import parse_price_limits


UTC = timezone.utc
SOURCES = {
    "tick_grid_baseline_20060501": "https://www.jpx.co.jp/files/tse/rules/regulations/060428_a2-1.pdf",
    "tick_grid_current_20060926_partial": "https://www.jpx.co.jp/files/tse/about/press/060926s.pdf",
    "rules_amendment_20070930": "https://www.jpx.co.jp/files/tse/rules/regulations/070905_a2-2.pdf",
    "rules_amendment_20080115": "https://www.jpx.co.jp/files/tse/rules/regulations/071221_b2.pdf",
    "tick_revision_proposal_20080129": "https://www.jpx.co.jp/files/tse/rules-participants/public-comment/data/080129ks.pdf",
    "price_limit_amendment_20080617": "https://www.jpx.co.jp/files/tse/rules/regulations/080527_a2.pdf",
    "tick_amendment_20080722_summary": "https://www.jpx.co.jp/files/tse/rules/regulations/080612_a1.pdf",
    "tick_amendment_20080722_rules": "https://www.jpx.co.jp/files/tse/rules/regulations/080612_a2.pdf",
    "rules_amendment_20080707": "https://www.jpx.co.jp/files/tse/rules/regulations/080626_b2.pdf",
    "rules_amendment_20090105": "https://www.jpx.co.jp/files/tse/rules/regulations/081215_a2.pdf",
    "tick_amendment_20100104_rules": "https://www.jpx.co.jp/files/tse/rules/regulations/091105_b2.pdf",
    "tick_amendment_20100104_notice": "https://www.jpx.co.jp/rules-participants/rules/revise/detail/detail_18461.html",
    "price_limit_amendment_20091230": "https://www.jpx.co.jp/files/tse/rules/regulations/091222_a2.pdf",
    "rule_revision_archive_2011": "https://www.jpx.co.jp/rules-participants/rules/revise/archives-15.html",
    "rule_revision_archive_2012": "https://www.jpx.co.jp/rules-participants/rules/revise/archives-14.html",
    "rule_revision_archive_2013": "https://www.jpx.co.jp/rules-participants/rules/revise/00-archives-13.html",
    "price_limit_amendment_20060501": "https://www.jpx.co.jp/files/tse/rules/regulations/060428_a2-2.pdf",
    "price_limit_arrowhead_outline_20100104": "https://www.jpx.co.jp/files/tse/rules/regulations/091105_b1.pdf",
    "price_limit_persistence_20100301": "https://www.jpx.co.jp/files/tse/rules/regulations/b7gje6000000myd3-att/100224_b2.pdf",
    "price_limit_broadening_rules_20200803": "https://www.jpx.co.jp/english/news/1030/b5b4pj000003iwhq-att/b5b4pj000003iwm6.pdf",
    "topix100_baseline_20131201": "https://www.jpx.co.jp/files/tse/news/20/b7gje6000004313n-att/leaflet_english.pdf",
    "topix100_20141031": "https://www.jpx.co.jp/files/tse/english/news/20/b7gje60000025vi1-att/TOPIX100list_english.xlsx",
    "mid400_snapshot_20230526": "https://www.jpx.co.jp/english/news/1030/dreu2500000076sg-att/eg.xlsx",
    "topix_size_guidebook": "https://www.jpx.co.jp/english/markets/indices/line-up/files/e_cal2_12_size.pdf",
    "phase_1_2_outline": "https://www.jpx.co.jp/files/tse/rules-participants/public-comment/data/b7gje60000048plb-att/130514_kabu-_e.pdf",
    "phase_3_rules": "https://www.jpx.co.jp/files/tse/rules-participants/public-comment/data/b7gje6000005712m-att/201412_kabuso_c.pdf",
    "phase_summary": "https://www.jpx.co.jp/english/corporate/news-releases/0060/b5b4pj000000qc6s-att/02.pdf",
    "phase_2_topix100_snapshot": "https://www.jpx.co.jp/files/tse/english/news/20/b7gje60000021q06-att/list_english.xls",
    "mid400_implementation": "https://www.jpx.co.jp/english/news/1030/20230602-01.html",
    "mid400_snapshot": "https://www.jpx.co.jp/english/news/1030/dreu2500000076ox-att/eg.xlsx",
    "current_tick_rules": "https://www.jpx.co.jp/english/equities/trading/domestic/07.html",
    "current_price_limits": "https://www.jpx.co.jp/english/equities/trading/domestic/06.html",
    "price_limit_rules_20141201": "https://www.jpx.co.jp/english/rules-participants/rules/regulations/tvdivq0000001vyt-att/bids_and_offers_price_limits_20141201.pdf",
    "price_limit_broadening_2020": "https://www.jpx.co.jp/english/news/1030/20200630-01.html",
    "topix100_20151030": "https://www.jpx.co.jp/english/news/1030/b5b4pj000000r2o1-att/TOPIX100list_english.xlsx",
    "topix100_20160401": "https://www.jpx.co.jp/english/news/1030/b5b4pj000000za9q-att/TOPIX100list_english.xlsx",
    "topix100_20170801": "https://www.jpx.co.jp/english/news/1030/b5b4pj000001pevq-att/TOPIX100list_english.xlsx",
    "topix100_20171031": "https://www.jpx.co.jp/english/news/1030/b5b4pj000001ue8n-att/TOPIX100list_english.xlsx",
    "topix100_20181031": "https://www.jpx.co.jp/english/news/1030/b5b4pj000002iprl-att/TOPIX100list_english.xlsx",
    "topix100_20190131": "https://www.jpx.co.jp/english/news/1030/b5b4pj000002niy5-att/TOPIX100list_english.xlsx",
    "topix100_20191031": "https://www.jpx.co.jp/english/news/1030/b5b4pj0000033wdv-att/TOPIX100list_english.xlsx",
    "topix100_20201030": "https://www.jpx.co.jp/english/news/1030/b5b4pj000003qu8e-att/TOPIX100list_english.xlsx",
    "topix100_20211029": "https://www.jpx.co.jp/english/news/1030/b5b4pj000004e9iq-att/TOPIX100list_english.xlsx",
    "topix100_20221031": "https://www.jpx.co.jp/english/news/1030/b5b4pj0000055o6l-att/TOPIX100list_english.xlsx",
}

TOPIX100_SNAPSHOT_DATES = {
    "topix100_20141031": "2014-10-31",
    "topix100_20151030": "2015-10-30",
    "topix100_20160401": "2016-04-01",
    "topix100_20170801": "2017-08-01",
    "topix100_20171031": "2017-10-31",
    "topix100_20181031": "2018-10-31",
    "topix100_20190131": "2019-01-31",
    "topix100_20191031": "2019-10-31",
    "topix100_20201030": "2020-10-30",
    "topix100_20211029": "2021-10-29",
    "topix100_20221031": "2022-10-31",
}

MID400_SNAPSHOT_DATES = {
    "mid400_snapshot_20230526": "2023-05-26",
    "mid400_snapshot": "2023-06-05",
}


@dataclass(frozen=True)
class JapanMarketRuleHistoryArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


Fetcher = Callable[[str], tuple[int, bytes, dict[str, str]]]


def write_japan_market_rule_history(
    *,
    run_id: str,
    artifact_root: str | Path = "artifacts/japan_market_rule_history",
    fetcher: Fetcher | None = None,
) -> JapanMarketRuleHistoryArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    root = Path(artifact_root) / run_id
    if (root / "manifest.json").exists():
        raise FileExistsError(f"immutable artifact already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc
    try:
        raw_dir = root / "raw"
        raw_dir.mkdir(exist_ok=True)
        fetch = fetcher or _fetch
        source_rows = []
        payloads: dict[str, bytes] = {}
        for source_id, url in SOURCES.items():
            suffix = Path(url).suffix or ".html"
            raw_path = raw_dir / f"{source_id}{suffix}"
            status, body, headers = fetch(url)
            if status == 200 and body:
                _atomic_bytes(raw_path, body)
                payloads[source_id] = body
            source_rows.append(
                {
                    "source_id": source_id,
                    "url": url,
                    "http_status": status,
                    "content_type": headers.get("Content-Type", headers.get("content-type", "")),
                    "raw_path": str(raw_path.resolve()) if raw_path.is_file() else "",
                    "sha256": _sha256(raw_path) if raw_path.is_file() else "",
                    "bytes": raw_path.stat().st_size if raw_path.is_file() else 0,
                }
            )
        sources = pd.DataFrame(source_rows)
        source_path = root / "source_manifest.csv"
        _atomic_csv(source_path, sources)

        windows = tick_regime_windows()
        bands = tick_rule_bands()
        transitions = tick_transition_evidence()
        amendment_ledger = rule_amendment_ledger()
        price_limit_transitions = price_limit_transition_evidence()
        assignments = _assignment_snapshots(payloads)
        price_limits = price_limit_rule_bands()
        current_price_limits = parse_price_limits(payloads.get("current_price_limits", b""))
        coverage = rule_coverage(assignments)
        outputs = {
            "tick_regime_windows.csv": windows,
            "tick_rule_bands.csv": bands,
            "tick_transition_evidence.csv": transitions,
            "rule_amendment_ledger.csv": amendment_ledger,
            "price_limit_transition_evidence.csv": price_limit_transitions,
            "assignment_snapshots.csv": assignments,
            "price_limit_rule_bands.csv": price_limits,
            "rule_coverage.csv": coverage,
        }
        for filename, frame in outputs.items():
            _atomic_csv(root / filename, frame)

        blockers = _producer_blockers(
            sources,
            windows,
            bands,
            amendment_ledger,
            assignments,
            price_limits,
            current_price_limits,
        )
        remaining = [
            "complete PIT TOPIX100/TOPIX500 constituent-change ledger is not yet retained; official baselines and checkpoints are retained but the full notice and corporate-action replay is incomplete",
            "issue-specific temporary price-limit expansion or removal events are not yet complete",
        ]
        manifest = {
            "schema_version": "2.0-source-backed-rule-continuity",
            "run_id": run_id,
            "artifact_type": "japan_market_rule_history_rule_continuity_candidate",
            "status": "partial_candidate" if not blockers else "failed_candidate",
            "component": "price_limits",
            "scorecard_eligible": False,
            "component_scorecard_eligible": False,
            "independent_validation_required": True,
            "official_jpx_source": True,
            "research_only": True,
            "evidence_scope": "official_common_stock_tick_and_normal_daily_price_limit_rule_continuity_plus_constituent_checkpoints_partial",
            "completed_lanes": [
                "common_stock_tick_size_rule_continuity_2006-07-01_2026-06-30",
                "normal_daily_price_limit_rule_continuity_2006-07-01_2026-06-30",
            ],
            "counts": {
                "sources": len(sources),
                "source_failures": int(sources["http_status"].ne(200).sum()),
                "tick_regime_windows": len(windows),
                "tick_band_rows": len(bands),
                "tick_transition_evidence_rows": len(transitions),
                "rule_amendment_ledger_rows": len(amendment_ledger),
                "price_limit_transition_evidence_rows": len(price_limit_transitions),
                "assignment_snapshot_rows": len(assignments),
                "assignment_snapshot_dates": int(assignments["effective_date"].nunique()),
                "price_limit_band_rows": len(price_limits),
            },
            "producer_blockers": blockers,
            "remaining_completion_blockers": remaining,
            "critical_blockers": [*blockers, *remaining],
            "artifacts": {
                "source_manifest": str(source_path.resolve()),
                **{Path(name).stem: str((root / name).resolve()) for name in outputs},
            },
            "output_sha256": {
                path.name: _sha256(path)
                for path in [source_path, *(root / name for name in outputs)]
            },
            "promotion_policy": {
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return JapanMarketRuleHistoryArtifacts(run_id, root, manifest_path, manifest)
    finally:
        lock.rmdir()


def tick_regime_windows() -> pd.DataFrame:
    rows = [
        ("legacy_20060501", "all_common_stocks", "2006-07-01", "2008-07-21", True, "tick_grid_baseline_20060501|tick_grid_current_20060926_partial|rule_amendment_ledger"),
        ("legacy_20080722", "all_common_stocks", "2008-07-22", "2010-01-03", True, "tick_amendment_20080722_rules|rule_amendment_ledger"),
        ("legacy_20100104", "all_common_stocks", "2010-01-04", "2014-01-13", True, "tick_amendment_20100104_rules|rule_revision_archive_2011|rule_revision_archive_2012|rule_revision_archive_2013"),
        ("phase_1", "topix100", "2014-01-14", "2014-07-21", True, "phase_1_2_outline"),
        ("phase_2", "topix100", "2014-07-22", "2015-09-23", True, "phase_summary"),
        ("phase_3_topix100", "topix100", "2015-09-24", "2023-06-04", True, "phase_3_rules|phase_summary"),
        ("phase_3_topix500", "topix500", "2023-06-05", "2026-06-30", True, "mid400_implementation|current_tick_rules"),
        ("other_issues_verified", "non_narrow_common_stocks", "2014-01-14", "2026-06-30", True, "tick_amendment_20100104_rules|phase_1_2_outline|phase_3_rules|current_tick_rules"),
    ]
    return pd.DataFrame(
        rows,
        columns=["regime_id", "eligible_group", "valid_from", "valid_to", "rule_table_verified", "source_ids"],
    )


def tick_rule_bands() -> pd.DataFrame:
    bounds = [1_000, 3_000, 5_000, 10_000, 30_000, 50_000, 100_000, 300_000, 500_000, 1_000_000, 3_000_000, 5_000_000, 10_000_000, 30_000_000, 50_000_000, None]
    other = [1, 1, 5, 10, 10, 50, 100, 100, 500, 1_000, 1_000, 5_000, 10_000, 10_000, 50_000, 100_000]
    phase1 = [1, 1, 1, 1, 5, 5, 10, 50, 50, 100, 500, 500, 1_000, 5_000, 5_000, 10_000]
    phase2 = [0.1, 0.5, 0.5, 1, 5, 5, 10, 50, 50, 100, 500, 500, 1_000, 5_000, 5_000, 10_000]
    phase3 = [0.1, 0.5, 1, 1, 5, 10, 10, 50, 100, 100, 500, 1_000, 1_000, 5_000, 10_000, 10_000]
    rows: list[dict[str, object]] = []
    for regime, regime_bounds, values, source in (
        (
            "legacy_20060501",
            [2_000, 3_000, 30_000, 50_000, 100_000, 1_000_000, 20_000_000, 30_000_000, None],
            [1, 5, 10, 50, 100, 1_000, 10_000, 50_000, 100_000],
            "tick_grid_baseline_20060501|tick_grid_current_20060926_partial",
        ),
        (
            "legacy_20080722",
            [2_000, 3_000, 30_000, 50_000, 300_000, 3_000_000, 20_000_000, 30_000_000, None],
            [1, 5, 10, 50, 100, 1_000, 10_000, 50_000, 100_000],
            "tick_amendment_20080722_rules",
        ),
        (
            "legacy_20100104",
            bounds,
            other,
            "tick_amendment_20100104_rules|phase_1_2_outline",
        ),
    ):
        _append_tick_bands(rows, regime, regime_bounds, values, source)
    for regime, values, source in (
        ("other_issues_verified", other, "phase_1_2_outline|phase_3_rules|current_tick_rules"),
        ("phase_1", phase1, "phase_1_2_outline|phase_summary"),
        ("phase_2", phase2, "phase_1_2_outline|phase_summary"),
        ("phase_3_topix100", phase3, "phase_3_rules|phase_summary"),
        ("phase_3_topix500", phase3, "mid400_implementation|current_tick_rules"),
    ):
        _append_tick_bands(rows, regime, bounds, values, source)
    return pd.DataFrame(rows)


def _append_tick_bands(
    rows: list[dict[str, object]],
    regime: str,
    bounds: list[int | None],
    values: list[float],
    source: str,
) -> None:
    lower: int | None = 0
    for upper, tick in zip(bounds, values):
        rows.append(
            {
                "regime_id": regime,
                "price_lower_exclusive_yen": lower,
                "price_upper_inclusive_yen": upper,
                "tick_size_yen": tick,
                "source_ids": source,
            }
        )
        lower = upper


def tick_transition_evidence() -> pd.DataFrame:
    rows = [
        (
            "2006-05-01",
            "effective_baseline_full_grid",
            "tick_grid_baseline_20060501",
            "Article 14 old/new comparison prints the full common-stock grid effective before the lake start",
            True,
        ),
        (
            "2006-09-26",
            "corroborating_dated_grid",
            "tick_grid_current_20060926_partial",
            "contemporaneous TSE system-plan table corroborates the baseline grid and normal-limit table",
            True,
        ),
        (
            "2008-01-29",
            "corroborating_full_prechange_grid",
            "tick_revision_proposal_20080129",
            "full pre-change grid in proposal attachment",
            True,
        ),
        (
            "2008-07-22",
            "effective_transition",
            "tick_amendment_20080722_summary|tick_amendment_20080722_rules",
            "formal Article 14 amendment effective date and old/new grid",
            True,
        ),
        (
            "2010-01-04",
            "effective_transition",
            "tick_amendment_20100104_rules|tick_amendment_20100104_notice",
            "formal arrowhead Article 14 amendment effective date and old/new grid",
            True,
        ),
        (
            "2013-05-14",
            "dated_endpoint_full_other_issues_grid",
            "phase_1_2_outline",
            "full Other Issues grid in consultation attachment",
            True,
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["evidence_date", "evidence_type", "source_ids", "scope", "continuous_interval_proven"],
    )


def price_limit_transition_evidence() -> pd.DataFrame:
    rows = [
        ("2006-05-01", "effective_baseline_article_2", "price_limit_amendment_20060501", "Article 2 paragraph 1 table retained unchanged immediately before the lake start", True),
        ("2006-09-26", "corroborating_full_baseline_table", "tick_grid_current_20060926_partial", "complete pre-arrowhead normal-limit table shown as current", True),
        ("2010-01-04", "effective_grid_transition", "price_limit_arrowhead_outline_20100104|tick_amendment_20100104_rules", "complete old/new Article 2 normal-limit tables", True),
        ("2010-03-01", "post_transition_persistence", "price_limit_persistence_20100301", "Article 2 paragraphs 1-2 explicitly unchanged", True),
        ("2014-12-01", "current_rulebook_full_table", "price_limit_rules_20141201", "complete current 34-band Article 2 table", True),
        ("2020-08-03", "exceptional_range_only_transition", "price_limit_broadening_rules_20200803", "exceptional broadening trigger changed; normal Article 2 table unchanged", True),
        ("2026-06-30", "cutoff_current_rule_table", "current_price_limits", "same current 34-band table observed at research cutoff", True),
    ]
    return pd.DataFrame(
        rows,
        columns=["evidence_date", "evidence_type", "source_ids", "scope", "continuous_interval_proven"],
    )


def rule_amendment_ledger() -> pd.DataFrame:
    rows = [
        ("2006-05-01", "tick_size", "Business Regulations Article 14", "tick_grid_baseline_20060501", "company-law terminology and instrument definitions", "full_grid_unchanged", "baseline_continuity"),
        ("2008-01-15", "tick_size", "Business Regulations Article 14", "rules_amendment_20080115", "bond trading additions", "common_stock_grid_unchanged", "continuity"),
        ("2008-07-07", "tick_size", "Business Regulations Article 14", "rules_amendment_20080707", "stock-class and listing-rule definitions", "common_stock_grid_unchanged", "continuity"),
        ("2008-07-22", "tick_size", "Business Regulations Article 14", "tick_amendment_20080722_summary|tick_amendment_20080722_rules", "two common-stock price bands narrowed", "grid_changed", "new_regime"),
        ("2009-01-05", "tick_size", "Business Regulations Article 14", "rules_amendment_20090105", "dematerialization terminology", "common_stock_grid_unchanged", "continuity"),
        ("2010-01-04", "tick_size", "Business Regulations Article 14", "tick_amendment_20100104_rules", "arrowhead common-stock tick table", "grid_changed", "new_regime"),
        ("2006-05-01", "normal_price_limit", "Rules Concerning Price Limits Article 2", "price_limit_amendment_20060501", "company-law terminology and exceptional instruments", "paragraph_1_grid_unchanged", "baseline_continuity"),
        ("2007-09-30", "normal_price_limit", "Rules Concerning Price Limits Article 2", "rules_amendment_20070930", "Financial Instruments and Exchange Act terminology", "paragraph_1_grid_unchanged", "continuity"),
        ("2008-01-15", "normal_price_limit", "Rules Concerning Price Limits Articles 2 and 2-2", "rules_amendment_20080115", "bond-only Article 2-2 added", "common_stock_grid_unchanged", "continuity"),
        ("2008-06-17", "normal_price_limit", "Rules Concerning Price Limits Articles 2 and 4", "price_limit_amendment_20080617", "foreign-security exceptions and base-price mechanics", "paragraph_1_grid_unchanged", "continuity"),
        ("2008-07-07", "normal_price_limit", "Rules Concerning Price Limits Article 2", "rules_amendment_20080707", "stock-class and direct-listing definitions", "paragraph_1_grid_unchanged", "continuity"),
        ("2009-01-05", "normal_price_limit", "Rules Concerning Price Limits Article 2", "rules_amendment_20090105", "dematerialization terminology", "paragraph_1_grid_unchanged", "continuity"),
        ("2009-12-30", "normal_price_limit", "Rules Concerning Price Limits Article 2", "price_limit_amendment_20091230", "new-share and subscription-right calculations", "paragraph_1_grid_unchanged", "continuity"),
        ("2010-01-04", "normal_price_limit", "Rules Concerning Price Limits Article 2", "price_limit_arrowhead_outline_20100104|tick_amendment_20100104_rules", "arrowhead normal common-stock limit table", "grid_changed", "new_regime"),
        ("2010-03-01", "normal_price_limit", "Rules Concerning Price Limits Article 2", "price_limit_persistence_20100301", "new-listing initial-price exception", "paragraphs_1_2_unchanged", "continuity"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "effective_date",
            "rule_family",
            "article",
            "source_ids",
            "amendment_scope",
            "normal_grid_effect",
            "continuity_disposition",
        ],
    )


def price_limit_rule_bands() -> pd.DataFrame:
    pre_bounds = [
        100, 200, 500, 1_000, 1_500, 2_000, 3_000, 5_000, 10_000, 20_000,
        30_000, 50_000, 70_000, 100_000, 150_000, 200_000, 300_000, 500_000,
        1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000,
        15_000_000, 20_000_000, 30_000_000, 50_000_000, None,
    ]
    pre_limits = [
        30, 50, 80, 100, 200, 300, 400, 500, 1_000, 2_000, 3_000, 4_000,
        5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 100_000, 200_000,
        300_000, 400_000, 500_000, 1_000_000, 2_000_000, 3_000_000,
        4_000_000, 5_000_000, 10_000_000,
    ]
    post_bounds = [
        100, 200, 500, 700, 1_000, 1_500, 2_000, 3_000, 5_000, 7_000,
        10_000, 15_000, 20_000, 30_000, 50_000, 70_000, 100_000, 150_000,
        200_000, 300_000, 500_000, 700_000, 1_000_000, 1_500_000, 2_000_000,
        3_000_000, 5_000_000, 7_000_000, 10_000_000, 15_000_000, 20_000_000,
        30_000_000, 50_000_000, None,
    ]
    post_limits = [
        30, 50, 80, 100, 150, 300, 400, 500, 700, 1_000, 1_500, 3_000,
        4_000, 5_000, 7_000, 10_000, 15_000, 30_000, 40_000, 50_000, 70_000,
        100_000, 150_000, 300_000, 400_000, 500_000, 700_000, 1_000_000,
        1_500_000, 3_000_000, 4_000_000, 5_000_000, 7_000_000, 10_000_000,
    ]
    rows: list[dict[str, object]] = []
    for regime, valid_from, valid_to, bounds, limits, sources in (
        (
            "pre_arrowhead_20060701",
            "2006-07-01",
            "2010-01-03",
            pre_bounds,
            pre_limits,
            "price_limit_amendment_20060501|tick_grid_current_20060926_partial|tick_amendment_20100104_rules",
        ),
        (
            "arrowhead_20100104",
            "2010-01-04",
            "2026-06-30",
            post_bounds,
            post_limits,
            "tick_amendment_20100104_rules|price_limit_persistence_20100301|price_limit_rules_20141201|current_price_limits",
        ),
    ):
        lower = 0
        for upper, limit in zip(bounds, limits):
            rows.append(
                {
                    "regime_id": regime,
                    "instrument_scope": "normal_common_stocks",
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "base_price_lower_inclusive_yen": lower,
                    "base_price_upper_exclusive_yen": upper,
                    "daily_limit_yen": limit,
                    "source_ids": sources,
                    "continuity_status": "verified_official_rule_and_amendment_ledger",
                }
            )
            lower = upper
    return pd.DataFrame(rows)


def rule_coverage(assignments: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subcomponent": "tick_rule_tables_2006_present", "status": "pass_continuity_candidate", "coverage": 1.0, "source_rows": len(tick_rule_bands())},
            {
                "subcomponent": "pre_2014_tick_rule_table",
                "status": "pass_direct_baseline_and_complete_2006_2010_amendment_ledger",
                "coverage": 1.0,
                "source_rows": len(tick_transition_evidence()),
            },
            {"subcomponent": "pit_narrow_tick_assignment", "status": "partial_snapshots", "coverage": 0.0, "source_rows": len(assignments)},
            {
                "subcomponent": "normal_price_limit_table",
                "status": "pass_two_regimes_direct_transition_and_complete_2006_2010_amendment_ledger",
                "coverage": 1.0,
                "source_rows": len(price_limit_rule_bands()),
            },
            {"subcomponent": "temporary_price_limit_events", "status": "missing", "coverage": 0.0, "source_rows": 0},
        ]
    )


def _assignment_snapshots(payloads: dict[str, bytes]) -> pd.DataFrame:
    rows = []
    phase = payloads.get("phase_2_topix100_snapshot", b"")
    if phase:
        frame = pd.read_excel(io.BytesIO(phase))
        for item in frame.to_dict("records"):
            rows.append(
                {
                    "effective_date": "2014-07-22",
                    "index_scope": "TOPIX100",
                    "code": str(item.get("code", "")).replace(".0", "").zfill(4),
                    "issue_name": str(item.get("name", "")).strip(),
                    "source_id": "phase_2_topix100_snapshot",
                }
            )
    baseline = payloads.get("topix100_baseline_20131201", b"")
    if baseline:
        text = "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(baseline)).pages)
        section = text.split("TOPIX100 Constituents (as of December 1, 2013)", 1)[-1]
        section = section.split("(Note) The constituents may change", 1)[0]
        codes = re.findall(r"(?:^|\s)(\d{1,3})\s+([0-9A-Z]{4})(?=\s)", section)
        seen = set()
        for _, code in codes:
            if code in seen:
                continue
            seen.add(code)
            rows.append(
                {
                    "effective_date": "2014-01-14",
                    "index_scope": "TOPIX100",
                    "code": code,
                    "issue_name": "",
                    "source_id": "topix100_baseline_20131201",
                }
            )
    for source_id, effective_date in MID400_SNAPSHOT_DATES.items():
        mid = payloads.get(source_id, b"")
        if not mid:
            continue
        frame = pd.read_excel(io.BytesIO(mid), header=2)
        frame = frame.rename(columns={frame.columns[0]: "code", frame.columns[1]: "name"})
        for item in frame.dropna(subset=["code"]).to_dict("records"):
            rows.append(
                {
                    "effective_date": effective_date,
                    "index_scope": "TOPIX_MID400",
                    "code": str(item.get("code", "")).replace(".0", "").zfill(4),
                    "issue_name": str(item.get("name", "")).strip(),
                    "source_id": source_id,
                }
            )
    for source_id, effective_date in TOPIX100_SNAPSHOT_DATES.items():
        payload = payloads.get(source_id, b"")
        if not payload:
            continue
        frame = pd.read_excel(io.BytesIO(payload), header=1)
        frame = frame.rename(columns={"Code": "code", "Issue": "name"})
        for item in frame.dropna(subset=["code"]).to_dict("records"):
            rows.append(
                {
                    "effective_date": effective_date,
                    "index_scope": "TOPIX100",
                    "code": str(item.get("code", "")).replace(".0", "").zfill(4),
                    "issue_name": str(item.get("name", "")).strip(),
                    "source_id": source_id,
                }
            )
    return pd.DataFrame(rows, columns=["effective_date", "index_scope", "code", "issue_name", "source_id"])


def _producer_blockers(
    sources: pd.DataFrame,
    windows: pd.DataFrame,
    bands: pd.DataFrame,
    amendment_ledger: pd.DataFrame,
    assignments: pd.DataFrame,
    price_limits: pd.DataFrame,
    current_price_limits: pd.DataFrame,
) -> list[str]:
    blockers = []
    if len(sources) != len(SOURCES) or sources["http_status"].ne(200).any() or sources["sha256"].eq("").any():
        blockers.append("official_source_download_or_hash_failure")
    if len(windows) != 8 or len(bands) != 114:
        blockers.append("tick_rule_window_or_band_denominator_invalid")
    if len(amendment_ledger) != 15:
        blockers.append("rule_amendment_ledger_denominator_invalid")
    counts = assignments.groupby("index_scope").size().to_dict() if not assignments.empty else {}
    # The official June 5, 2023 file contains 398 populated Mid400 rows as of
    # that date. Preserve the source denominator instead of forcing the label.
    if counts != {
        "TOPIX100": 100 * (2 + len(TOPIX100_SNAPSHOT_DATES)),
        "TOPIX_MID400": 398 * len(MID400_SNAPSHOT_DATES),
    }:
        blockers.append(f"assignment_snapshot_denominator_invalid:{counts}")
    topix_sets = {
        date: set(frame["code"])
        for date, frame in assignments.loc[assignments["index_scope"].eq("TOPIX100")].groupby("effective_date")
    }
    if topix_sets.get("2014-01-14") != topix_sets.get("2014-07-22"):
        blockers.append("topix100_201312_baseline_does_not_match_201407_checkpoint")
    mid_sets = {
        date: set(frame["code"])
        for date, frame in assignments.loc[assignments["index_scope"].eq("TOPIX_MID400")].groupby("effective_date")
    }
    if mid_sets.get("2023-05-26") != mid_sets.get("2023-06-05"):
        blockers.append("mid400_20230526_does_not_match_20230605_checkpoint")
    if assignments["code"].str.fullmatch(r"[0-9A-Z]{4}").ne(True).any():
        blockers.append("assignment_snapshot_codes_invalid")
    if len(price_limits) != 63:
        blockers.append("historical_price_limit_table_denominator_invalid")
    post = price_limits.loc[price_limits["regime_id"].eq("arrowhead_20100104")]
    observed_current = [_numeric_yen(value) for value in current_price_limits.get("daily_limit_yen", [])]
    if observed_current != post["daily_limit_yen"].astype(int).tolist():
        blockers.append("current_price_limit_table_does_not_match_post_2010_regime")
    return blockers


def _numeric_yen(value: object) -> int:
    text = str(value).strip().lower().replace("±", "").replace(",", "")
    if text.endswith("million"):
        return int(float(text.removesuffix("million").strip()) * 1_000_000)
    return int(float(text))


def _fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "quant-lab-jpx-rule-history/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return int(response.status), response.read(64 * 1024 * 1024 + 1), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
