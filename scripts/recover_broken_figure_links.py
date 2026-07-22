#!/usr/bin/env python3
"""Recover known page-repair image placeholders from immutable first-pass layout crops."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import requests
from PIL import Image


RECOVERIES = {
    ("n05-adaptive-checkpoint-adjoint-method-for-gradient-estimation-in-neural-ode", 5): {
        "indices": [20],
        "replacements": {
            "image": (20, "Figure 6 — absolute error in gradient estimation for ACA, adjoint, and naive methods"),
        },
    },
    ("ts03-local-linear-quantile-estimation-for-nonstationary-time-series", 17): {
        "indices": [0, 1],
        "replacements": {
            "image": [
                (0, "Global-warming temperature-anomaly time series"),
                (1, "U.S. monthly-precipitation time series"),
            ],
        },
    },
    ("b01-discovering-governing-equations-from-data-by-sparse-identification-of-nonlinear-dynamical-sy", 15): {
        "indices": [0, 2, 3, 5, 6],
        "replacements": {
            "image1.png": (0, "Figure 8 — full simulation and identified-system trajectories"),
            "image2.png": None,
            "image3.png": (2, "Figure 9 — full simulation trajectory"),
            "image4.png": (3, "Figure 9 — identified-system trajectory"),
            "image5.png": (5, "Figure 10 — full simulation trajectory"),
            "image6.png": (6, "Figure 10 — identified-system trajectory"),
        },
    },
    ("f03-abides-gym-gym-environments-for-multi-agent-discrete-event-simulation-and-application-to-fin", 3): {
        "indices": [0],
        "replacements": {"image.png": (0, "Figure 1 — ABIDES-Core kernel mechanism")},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    ocr_rows = list(csv.DictReader((run_dir / "ocr_manifest.csv").open(encoding="utf-8")))
    records: list[dict[str, object]] = []
    for (document_id, page), spec in RECOVERIES.items():
        chunk = next(
            (
                row for row in ocr_rows
                if row["document_id"] == document_id and int(row["start_page"]) <= page <= int(row["end_page"])
            ),
            None,
        )
        # The registry is shared by multiple OCR waves.
        if chunk is None:
            continue
        raw = json.load(open(chunk["raw_response_path"], encoding="utf-8"))
        elements = raw["layout_details"][page - int(chunk["start_page"])]
        by_index = {int(element["index"]): element for element in elements}
        local_by_index: dict[int, Path] = {}
        for index in spec["indices"]:
            element = by_index[index]
            url = str(element["content"])
            response = requests.get(url, timeout=120, headers={"User-Agent": "QuantLab-Figure-Recovery/1.0"})
            response.raise_for_status()
            data = response.content
            digest = hashlib.sha256(data).hexdigest()
            target = run_dir / "assets_verified" / document_id / f"recovered-p{page:04d}-e{index:03d}__{digest[:12]}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            with Image.open(target) as image:
                image.verify()
            local_by_index[index] = target
            records.append(
                {"document_id": document_id, "page_number": page, "element_index": index,
                 "original_url": url, "local_path": str(target), "sha256": digest, "bytes": len(data)}
            )
        markdown_path = run_dir / "markdown_verified_localized" / f"{document_id}.md"
        if not markdown_path.is_file():
            continue
        text = markdown_path.read_text(encoding="utf-8")
        for placeholder, mapping in spec["replacements"].items():
            if mapping is None:
                # Remove only the broken image line; retain its nearby heading/caption prose.
                import re
                text = re.sub(rf"^!\[[^\]]*\]\({re.escape(placeholder)}\)\n?", "", text, flags=re.MULTILINE)
                continue
            import re
            mappings = mapping if isinstance(mapping, list) else [mapping]
            replacement = "\n\n".join(
                f"![{alt}](../{local_by_index[index].relative_to(run_dir).as_posix()})"
                for index, alt in mappings
            )
            text, count = re.subn(
                rf"!\[[^\]]*\]\({re.escape(placeholder)}\)", replacement, text, count=1
            )
            if count != 1:
                raise RuntimeError(f"Expected one placeholder {placeholder} in {markdown_path}")
        markdown_path.write_text(text, encoding="utf-8")
    manifest = run_dir / "recovered_figure_manifest.csv"
    fields = ["document_id", "page_number", "element_index", "original_url", "local_path", "sha256", "bytes"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Recovered {len(records)} figure crops; manifest: {manifest}")


if __name__ == "__main__":
    main()
