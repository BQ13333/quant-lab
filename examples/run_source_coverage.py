from __future__ import annotations

import argparse

from quant_lab.research.source_coverage import write_markdown_source_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory markdown source headings for quant strategy coverage.")
    parser.add_argument(
        "source_root",
        nargs="?",
        default="..",
        help="Root folder containing quant markdown sources. Defaults to the parent 08_Quantitative folder.",
    )
    parser.add_argument("--artifact-root", default="artifacts/source_coverage")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--include-repo-docs",
        action="store_true",
        help="Include quant_lab markdown/docs/vault files. Default scans external book corpus only.",
    )
    args = parser.parse_args()

    artifact_dir = write_markdown_source_coverage(
        source_root=args.source_root,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        exclude_dirs=() if args.include_repo_docs else ("quant_lab", ".ocr_cache"),
    )
    print("=" * 80)
    print("SOURCE COVERAGE COMPLETE")
    print("=" * 80)
    print(f"Artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
