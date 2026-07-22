from __future__ import annotations

import argparse

from quant_lab.data.lake_ops import compact_long_ohlcv_partitions, write_partition_health_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run partition-health and optional compaction for long OHLCV data.")
    parser.add_argument("long_ohlcv_root", help="Partitioned long OHLCV root, for example artifacts/data_lake/prices_long/RUN/ohlcv.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/data_lake_ops")
    parser.add_argument("--min-file-size-bytes", type=int, default=1_000_000)
    parser.add_argument("--max-files-per-partition", type=int, default=1)
    parser.add_argument("--compact", action="store_true", help="Also write a compacted year-level OHLCV root.")
    parser.add_argument("--output-root", default=None, help="Compaction output root. Defaults to sibling ohlcv_compacted.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.compact:
        result = compact_long_ohlcv_partitions(
            args.long_ohlcv_root,
            output_root=args.output_root,
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            overwrite=args.overwrite,
            min_file_size_bytes=args.min_file_size_bytes,
            max_files_per_partition=args.max_files_per_partition,
        )
        print(f"Run ID: {result.run_id}")
        print(f"Output root: {result.output_root}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Compaction manifest: {result.compaction_manifest_path}")
        print(f"Health before: {result.partition_health_before_path}")
        print(f"Health after: {result.partition_health_after_path}")
        print(f"Status: {result.manifest['status']}")
        print(f"Source files: {result.manifest['source_files']}")
        print(f"Output files: {result.manifest['output_files']}")
        print(f"Source rows: {result.manifest['source_rows']}")
        print(f"Output rows: {result.manifest['output_rows']}")
        return

    report = write_partition_health_report(
        args.long_ohlcv_root,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        min_file_size_bytes=args.min_file_size_bytes,
        max_files_per_partition=args.max_files_per_partition,
    )
    print(f"Run ID: {report.run_id}")
    print(f"Manifest: {report.manifest_path}")
    print(f"File inventory: {report.file_inventory_path}")
    print(f"Partition health: {report.partition_health_path}")
    print(f"Summary: {report.summary_path}")
    print(f"Status: {report.manifest['status']}")
    print(f"Parquet files: {report.manifest['parquet_files']}")
    print(f"Partitions: {report.manifest['partitions']}")
    print(f"Warning partitions: {report.manifest['warn_partitions']}")


if __name__ == "__main__":
    main()
