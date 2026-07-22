from __future__ import annotations

import argparse

from quant_lab.data.japan_g3_fundamentals import (
    extract_japan_g3_fundamentals_qa,
    write_japan_g3_fundamentals_qa_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded Japan parameterized-fundamentals QA shard.")
    parser.add_argument("--rics", default="7203.T,6758.T,8306.T,9984.T,9432.T")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--frequencies", default="FY,FQ")
    parser.add_argument("--run-id", default="REFINITIV-JPN-G3-FUNDAMENTALS-QA-20260712")
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_japan_g3_fundamentals")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    rics = _csv(args.rics)
    frequencies = _csv(args.frequencies)
    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        records = extract_japan_g3_fundamentals_qa(
            adapter.ld,
            rics=rics,
            start_date=args.start_date,
            end_date=args.end_date,
            frequencies=frequencies,
        )
    finally:
        adapter.close()
    artifacts = write_japan_g3_fundamentals_qa_artifacts(
        records,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        requested_rics=rics,
        start_date=args.start_date,
        end_date=args.end_date,
        frequencies=frequencies,
        session_name=args.session,
    )
    print(f"Run ID: {args.run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"History rows: {len(records)}")
    print(artifacts.quality.to_string(index=False))


if __name__ == "__main__":
    main()
