from __future__ import annotations

import argparse

from quant_lab.data.data_depth import write_data_depth_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Write data-depth diagnostics for a partitioned long OHLCV root.")
    parser.add_argument("long_ohlcv_root", help="Path to prices_long/<run_id>/ohlcv.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/data_depth")
    parser.add_argument("--extraction-manifest-path", default=None)
    args = parser.parse_args()

    report = write_data_depth_report(
        args.long_ohlcv_root,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        extraction_manifest_path=args.extraction_manifest_path,
    )
    print(f"Artifact dir: {report.artifact_dir}")
    print(f"Manifest: {report.manifest_path}")
    print(f"Summary: {report.markdown_path}")
    print(report.overall.to_string(index=False))


if __name__ == "__main__":
    main()
