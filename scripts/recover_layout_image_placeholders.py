#!/usr/bin/env python3
"""Recover broken OCR image placeholders from source-PDF layout geometry.

The recovery is deliberately source based: it renders the immutable source PDF at
high resolution, crops only bounding boxes recorded by the first-pass GLM layout
response, verifies the resulting pixels, and then replaces the broken Markdown
reference.  A manifest records all source hashes, raw-response hashes, geometry,
and recovered asset hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageStat


PAGE_RE = re.compile(r"<!-- OCR_PAGE (\d+) [^>]*-->")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


RECOVERIES: list[dict[str, object]] = [
    {
        "document_id": "w5-06-bayesian-learning-and-inference-in-recurrent-switching-linear-dynamical-systems",
        "page": 8,
        "groups": [[0]],
        "collapse": {
            "start": "**\"Run to Left Corner\"**",
            "end": "**Figure 5:**",
            "expected_placeholders": 6,
            "alt": "Figure 5: NBA player trajectory vector fields and posterior state-usage probabilities",
        },
    },
    {
        "document_id": "w3-gen-01-deep-unsupervised-learning-using-nonequilibrium-thermodynamics",
        "page": 14,
        "groups": [[0]],
        "placeholders": ["image"],
    },
    {
        "document_id": "w3-gen-02-denoising-diffusion-probabilistic-models",
        "page": 25,
        "groups": [[0]],
        "placeholders": ["image.png"],
    },
    {
        "document_id": "w3-gen-06-csdi-conditional-score-based-diffusion-models-for-probabilistic-time-series-imputation",
        "page": 8,
        "groups": [[0, 1]],
        "placeholders": ["image"],
    },
    {
        "document_id": "w3-nds-02-recurrent-neural-networks-for-multivariate-time-series-with-missing-values",
        "page": 13,
        "groups": [[11]],
        "placeholders": ["image.png"],
    },
    {
        "document_id": "w3-nds-06-a-time-series-is-worth-64-words-long-term-forecasting-with-transformers",
        "page": 24,
        "groups": [[0, 1]],
        "placeholders": ["figure7.png"],
    },
    {
        "document_id": "w3-nop-03-learning-the-solution-operator-of-parametric-partial-differential-equations-with-physi",
        "page": 4,
        "groups": [[0]],
        "placeholders": ["image-placeholder.png"],
    },
    {
        "document_id": "w3-nop-03-learning-the-solution-operator-of-parametric-partial-differential-equations-with-physi",
        "page": 24,
        "groups": [[1], [3], [6]],
        "placeholders": ["image", "image", "image"],
    },
    {
        "document_id": "w3-nop-03-learning-the-solution-operator-of-parametric-partial-differential-equations-with-physi",
        "page": 26,
        "groups": [
            [0, 1, 2, 3, 5, 6, 9, 10],
            [13, 14, 15, 17, 19, 20, 22, 23],
        ],
        "placeholders": ["image", "image"],
    },
    {
        "document_id": "w3-nop-03-learning-the-solution-operator-of-parametric-partial-differential-equations-with-physi",
        "page": 28,
        "groups": [[1], [3]],
        "placeholders": ["image", "image"],
    },
    {
        "document_id": "w3-nop-03-learning-the-solution-operator-of-parametric-partial-differential-equations-with-physi",
        "page": 32,
        "groups": [[1], [3]],
        "placeholders": ["image_link_a", "image_link_b"],
    },
    {
        "document_id": "w3-nop-05-pde-refiner-achieving-accurate-long-rollouts-with-neural-pde-solvers",
        "page": 28,
        "groups": [[0]],
        "collapse": {
            "start": "**Ground Truth**",
            "end": "*Figure 13:*",
            "expected_placeholders": 30,
            "alt": "Figure 13: Ground-truth and PDE-Refiner vorticity trajectories",
        },
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def page_bounds(text: str, page: int) -> tuple[int, int]:
    markers = list(PAGE_RE.finditer(text))
    for index, marker in enumerate(markers):
        if int(marker.group(1)) == page:
            return marker.start(), markers[index + 1].start() if index + 1 < len(markers) else len(text)
    raise RuntimeError(f"Missing OCR page marker {page}")


def render_source_page(source_pdf: Path, page: int, dpi: int, target_prefix: Path) -> Path:
    output = target_prefix.with_suffix(".png")
    target_prefix.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi),
            "-png", "-singlefile", str(source_pdf), str(target_prefix),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not output.is_file():
        raise RuntimeError(f"Poppler did not render {source_pdf} page {page}")
    return output


def crop_group(
    rendered_page: Path,
    elements: list[dict[str, object]],
    indices: list[int],
    output_dir: Path,
    document_id: str,
    page: int,
    group_number: int,
    padding_layout_units: int,
) -> tuple[Path, list[int], list[int]]:
    by_index = {int(element["index"]): element for element in elements}
    missing = [index for index in indices if index not in by_index]
    if missing:
        raise RuntimeError(f"Missing layout indices {missing} for {document_id} page {page}")
    boxes = [list(map(int, by_index[index]["bbox_2d"])) for index in indices]
    union = [
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    ]
    layout_width = int(by_index[indices[0]]["width"])
    layout_height = int(by_index[indices[0]]["height"])
    with Image.open(rendered_page) as page_image:
        scale_x = page_image.width / layout_width
        scale_y = page_image.height / layout_height
        padded = [
            max(0, union[0] - padding_layout_units),
            max(0, union[1] - padding_layout_units),
            min(layout_width, union[2] + padding_layout_units),
            min(layout_height, union[3] + padding_layout_units),
        ]
        pixel_box = [
            round(padded[0] * scale_x), round(padded[1] * scale_y),
            round(padded[2] * scale_x), round(padded[3] * scale_y),
        ]
        crop = page_image.convert("RGB").crop(tuple(pixel_box))
    if crop.width < 64 or crop.height < 64:
        raise RuntimeError(f"Recovered crop is implausibly small: {crop.size}")
    extrema = ImageStat.Stat(crop.convert("L")).extrema[0]
    if extrema[1] - extrema[0] < 8:
        raise RuntimeError(f"Recovered crop is visually blank: grayscale extrema {extrema}")
    staging = output_dir / f".recovered-p{page:04d}-g{group_number:02d}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    crop.save(staging, format="PNG", optimize=True)
    digest = sha256(staging)
    target = output_dir / f"recovered-p{page:04d}-g{group_number:02d}__{digest[:12]}.png"
    staging.replace(target)
    with Image.open(target) as verified:
        verified.verify()
    return target, union, pixel_box


def replace_placeholders(section: str, placeholders: list[str], assets: list[str]) -> str:
    if len(placeholders) != len(assets):
        raise RuntimeError("Placeholder and asset counts differ")
    for placeholder, asset in zip(placeholders, assets, strict=True):
        pattern = re.compile(rf"!\[([^\]]*)\]\({re.escape(placeholder)}\)")
        section, count = pattern.subn(lambda match: f"![{match.group(1)}]({asset})", section, count=1)
        if count != 1:
            raise RuntimeError(f"Expected exactly one remaining placeholder {placeholder!r}")
    return section


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--padding-layout-units", type=int, default=6)
    parser.add_argument("--tmp-dir", default="tmp/pdfs/layout_figure_recovery")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    markdown_dir = run_dir / "markdown_verified_localized"
    assets_root = run_dir / "assets_verified"
    tmp_dir = Path(args.tmp_dir).expanduser().resolve()
    chunk_rows = list(csv.DictReader((run_dir / "chunk_plan.csv").open(encoding="utf-8")))
    manifest_rows: list[dict[str, object]] = []

    for rule in RECOVERIES:
        document_id = str(rule["document_id"])
        page = int(rule["page"])
        chunk = next(
            (
                row for row in chunk_rows
                if row["document_id"] == document_id
                and int(row["start_page"]) <= page <= int(row["end_page"])
            ),
            None,
        )
        if chunk is None:
            continue
        source_pdf = Path(chunk["input_path"]).resolve()
        raw_path = Path(chunk["raw_response_path"]).resolve()
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        elements = raw["layout_details"][page - int(chunk["start_page"])]
        render_prefix = tmp_dir / document_id / f"p{page:04d}-dpi{args.dpi}"
        rendered = render_source_page(source_pdf, page, args.dpi, render_prefix)

        asset_relatives: list[str] = []
        for group_number, group in enumerate(rule["groups"], 1):
            asset, union, pixel_box = crop_group(
                rendered, elements, list(map(int, group)), assets_root / document_id,
                document_id, page, group_number, args.padding_layout_units,
            )
            relative = f"../{asset.relative_to(run_dir).as_posix()}"
            asset_relatives.append(relative)
            with Image.open(asset) as image:
                width, height = image.size
            manifest_rows.append(
                {
                    "document_id": document_id,
                    "page_number": page,
                    "group_number": group_number,
                    "layout_indices": ";".join(map(str, group)),
                    "layout_union_bbox": json.dumps(union, separators=(",", ":")),
                    "rendered_pixel_bbox": json.dumps(pixel_box, separators=(",", ":")),
                    "dpi": args.dpi,
                    "source_pdf": str(source_pdf),
                    "source_pdf_sha256": sha256(source_pdf),
                    "raw_response": str(raw_path),
                    "raw_response_sha256": sha256(raw_path),
                    "local_path": str(asset),
                    "sha256": sha256(asset),
                    "bytes": asset.stat().st_size,
                    "width": width,
                    "height": height,
                    "method": "source_pdf_crop_from_glm_layout_geometry",
                }
            )

        markdown_path = markdown_dir / f"{document_id}.md"
        text = markdown_path.read_text(encoding="utf-8")
        start, end = page_bounds(text, page)
        section = text[start:end]
        collapse = rule.get("collapse")
        if collapse:
            collapse = dict(collapse)
            left = section.find(str(collapse["start"]))
            right = section.find(str(collapse["end"]), left)
            if left < 0 or right < 0:
                raise RuntimeError(f"Collapse anchors missing in {document_id} page {page}")
            removed = section[left:right]
            broken = [target for _, target in IMAGE_RE.findall(removed)]
            if len(broken) != int(collapse["expected_placeholders"]):
                raise RuntimeError(
                    f"Expected {collapse['expected_placeholders']} collapsed placeholders, found {len(broken)}"
                )
            recovered = f"![{collapse['alt']}]({asset_relatives[0]})\n\n"
            section = section[:left] + recovered + section[right:]
        else:
            section = replace_placeholders(
                section, list(map(str, rule["placeholders"])), asset_relatives
            )
        text = text[:start] + section + text[end:]
        markdown_path.write_text(text, encoding="utf-8")
        rendered.unlink(missing_ok=True)

    fields = [
        "document_id", "page_number", "group_number", "layout_indices",
        "layout_union_bbox", "rendered_pixel_bbox", "dpi", "source_pdf",
        "source_pdf_sha256", "raw_response", "raw_response_sha256", "local_path",
        "sha256", "bytes", "width", "height", "method",
    ]
    manifest_path = run_dir / "recovered_layout_figure_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Recovered {len(manifest_rows)} source-faithful figures; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
