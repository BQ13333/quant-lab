#!/usr/bin/env python3
"""Audit localized verified Markdown for page, formula, and image durability."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_zai_formula_vision_repairs as vision


IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
PAGE_RE = re.compile(r"<!-- OCR_PAGE \d+ [^>]*-->")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown-dir", required=True)
    parser.add_argument("--expected-documents", type=int, required=True)
    parser.add_argument("--expected-pages", type=int, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    root = Path(args.markdown_dir).expanduser().resolve()
    files = sorted(root.glob("*.md"))
    errors: list[str] = []
    image_paths: list[Path] = []
    page_markers = 0
    formula_issue_files: dict[str, list[str]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8")
        page_markers += len(PAGE_RE.findall(text))
        issues = vision.markdown_structure_issues(text)
        if issues:
            formula_issue_files[path.name] = issues
        for target in IMAGE_RE.findall(text):
            if target.startswith(("http://", "https://")):
                errors.append(f"remote image remains: {path.name}")
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.is_file():
                errors.append(f"missing local image: {path.name}: {target}")
                continue
            try:
                with Image.open(resolved) as image:
                    image.verify()
            except Exception as exc:
                errors.append(f"invalid image: {resolved}: {exc}")
            image_paths.append(resolved)
    if len(files) != args.expected_documents:
        errors.append(f"document count {len(files)} != {args.expected_documents}")
    if page_markers != args.expected_pages:
        errors.append(f"page marker count {page_markers} != {args.expected_pages}")
    if len(image_paths) != args.expected_images:
        errors.append(f"localized image references {len(image_paths)} != {args.expected_images}")
    if formula_issue_files:
        errors.append(f"formula structure issues in {len(formula_issue_files)} files")
    report = {
        "status": "pass" if not errors else "fail",
        "markdown_directory": str(root),
        "documents": len(files),
        "page_markers": page_markers,
        "localized_image_references": len(image_paths),
        "unique_localized_images": len(set(image_paths)),
        "total_image_bytes": sum(path.stat().st_size for path in set(image_paths)),
        "formula_structure_issue_files": formula_issue_files,
        "errors": errors,
        "corpus_sha256": hashlib.sha256("".join(hashlib.sha256(path.read_bytes()).hexdigest() for path in files).encode()).hexdigest(),
    }
    output = Path(args.output_json).expanduser().resolve()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Localized Markdown audit {report['status'].upper()}: {len(files)} documents, {page_markers} pages, {len(image_paths)} images")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
