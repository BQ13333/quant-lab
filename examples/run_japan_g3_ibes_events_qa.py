from __future__ import annotations

import argparse

from quant_lab.data.japan_g3_ibes_events import (
    derive_analyst_revision_events,
    extract_analyst_snapshots_qa,
    extract_consensus_history_qa,
    extract_earnings_actuals_qa,
    write_japan_g3_ibes_events_qa_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded Japan IBES and earnings-event QA.")
    parser.add_argument("--rics", default="7203.T,6758.T,8306.T,9984.T,9432.T")
    parser.add_argument("--consensus-start-date", default="2021-01-01")
    parser.add_argument("--revision-start-date", default="2024-01-01")
    parser.add_argument("--events-start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--run-id", default="REFINITIV-JPN-G3-IBES-EVENTS-QA-20260712")
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_japan_g3_ibes_events")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rejected-misaligned-event-rows", type=int, default=0)
    args = parser.parse_args()
    rics = _csv(args.rics)

    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        consensus = extract_consensus_history_qa(
            adapter.ld, rics, args.consensus_start_date, args.end_date
        )
        snapshots = extract_analyst_snapshots_qa(
            adapter.ld, rics, args.revision_start_date, args.end_date
        )
        revisions = derive_analyst_revision_events(snapshots)
        events = extract_earnings_actuals_qa(
            adapter.ld, rics, args.events_start_date, args.end_date
        )
    finally:
        adapter.close()
    artifacts = write_japan_g3_ibes_events_qa_artifacts(
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        consensus=consensus,
        analyst_snapshots=snapshots,
        revision_events=revisions,
        earnings_events=events,
        requested_rics=rics,
        consensus_start_date=args.consensus_start_date,
        revision_start_date=args.revision_start_date,
        end_date=args.end_date,
        session_name=args.session,
        rejected_misaligned_event_rows=args.rejected_misaligned_event_rows,
    )
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Consensus rows: {len(consensus)}")
    print(f"Analyst snapshots: {len(snapshots)}")
    print(f"Revision events: {len(revisions)}")
    print(f"Earnings events: {len(events)}")
    print(artifacts.quality.to_string(index=False))


if __name__ == "__main__":
    main()
