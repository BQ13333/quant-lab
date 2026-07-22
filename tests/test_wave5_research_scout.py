from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_wave5_research_scout.py"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_matches_arxiv_and_reports_unmatched(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    discovered = tmp_path / "discovered.csv"
    roster = tmp_path / "roster.csv"
    output = tmp_path / "output"
    gold_fields = [
        "candidate_id", "title", "authors", "year", "doi", "arxiv_id", "landing_url", "pdf_url", "license",
        "license_status", "peer_review_status", "primary_specialist", "secondary_specialists", "admission_class",
    ]
    write_csv(
        gold,
        gold_fields,
        [
            {
                "candidate_id": "W5-01",
                "title": "Canonical Hidden Markov Result",
                "authors": "Alice Example; Bob Scholar",
                "year": "2024",
                "doi": "",
                "arxiv_id": "1234.56789",
                "landing_url": "https://arxiv.org/abs/1234.56789",
                "pdf_url": "https://arxiv.org/pdf/1234.56789",
                "license": "review required",
                "license_status": "review_required",
                "peer_review_status": "published_peer_reviewed",
                "primary_specialist": "agent_a",
                "secondary_specialists": "agent_b",
                "admission_class": "core",
            }
        ],
    )
    write_csv(
        discovered,
        ["title", "authors", "year", "doi", "arxiv_id", "landing_url", "pdf_url", "license", "license_status", "license_evidence_url", "peer_review_status", "peer_review_evidence_url", "primary_page_opened"],
        [
            {
                "title": "Canonical Hidden Markov Result",
                "authors": "Alice Example; Bob Scholar",
                "year": "2024",
                "doi": "",
                "arxiv_id": "1234.56789v2",
                "landing_url": "https://arxiv.org/abs/1234.56789",
                "pdf_url": "",
                "license": "review required",
                "license_status": "review_required",
                "license_evidence_url": "",
                "peer_review_status": "published_peer_reviewed",
                "peer_review_evidence_url": "https://jmlr.org/papers/example.html",
                "primary_page_opened": "true",
            },
            {
                "title": "Unmatched Blog Post",
                "authors": "Unknown",
                "year": "2024",
                "doi": "",
                "arxiv_id": "",
                "landing_url": "https://example.com/blog",
                "pdf_url": "",
                "license": "unknown",
                "license_status": "unknown",
                "license_evidence_url": "",
                "peer_review_status": "unknown",
                "peer_review_evidence_url": "",
                "primary_page_opened": "false",
            },
        ],
    )
    write_csv(
        roster,
        ["agent_id"],
        [{"agent_id": "agent_a"}, {"agent_id": "agent_b"}],
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gold",
            str(gold),
            "--discovered",
            str(discovered),
            "--roster",
            str(roster),
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["core_recall_pct"] == 100.0
    assert metrics["unmatched_discoveries"] == 1
    assert metrics["route_error_count"] == 0
