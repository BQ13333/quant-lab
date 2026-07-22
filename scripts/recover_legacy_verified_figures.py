#!/usr/bin/env python3
"""Recover known legacy figure placeholders from immutable first-pass OCR crops.

The original PDFs were deleted only after conversion was verified.  The provider
layout response is retained and includes the exact image crops and their page
coordinates.  This script downloads those crops, reconstructs multi-panel
figures on a white canvas, and rewrites only the known broken Markdown targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image


@dataclass(frozen=True)
class Recovery:
    document_id: str
    page: int
    target: str
    source_indices: tuple[int, ...]
    alt: str
    composite: bool = False
    remove: bool = False


RECOVERIES = (
    Recovery(
        "7eb78c77b9ca-fadin-fast-discretized-inference-for-hawkes-processes-with-general-parametric-kerne",
        8,
        "../assets_verified/7eb78c77b9ca-fadin-fast-discretized-inference-for-hawkes-processes-with-general-parametric-kerne/embedded__9857e7c40a5a.svg",
        (0,),
        "Figure 3 - spatial and temporal atom patterns and estimated stimulus-response intensities",
    ),
    Recovery(
        "1101-3422-modeling-microstructure-noise-with-mutually-exciting-point-processes",
        23, "figure8.png", (0, 1), "Figure 8 - empirical distributions of x", True,
    ),
    Recovery("1410-3394-volatility-is-rough", 17, "image.png", (0,), "Figure 3.1"),
    Recovery(
        "1704-04110-deepar-probabilistic-forecasting-with-autoregressive-recurrent-networks",
        7, "image", (0,), "Uncertainty growth over time for ISSM and DeepAR models",
    ),
    Recovery(
        "1704-04110-deepar-probabilistic-forecasting-with-autoregressive-recurrent-networks",
        7, "image", (2,), "Coverage plots for two spans on the ec-sub dataset",
    ),
    Recovery("1706-04599-on-calibration-of-modern-neural-networks", 14, "image_link", (0,), "Figure S2"),
    Recovery("1706-04599-on-calibration-of-modern-neural-networks", 14, "image_link", (2,), "Figure S3"),
    Recovery("1706-04599-on-calibration-of-modern-neural-networks", 14, "image_link", (4,), "Figure S4"),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1706-04599-on-calibration-of-modern-neural-networks", 14, "plot", (), "", remove=True,
    ),
    Recovery(
        "1b2b9d6c9339-inconsistency-of-cross-validation-for-structure-learning-in-gaussian-graphical-mode",
        9, "image1.png", (0,), "Figure 4 - TPR versus sample size and dimension",
    ),
    Recovery(
        "1b2b9d6c9339-inconsistency-of-cross-validation-for-structure-learning-in-gaussian-graphical-mode",
        9, "image2.png", (2,), "Figure 5 - FDR versus sample size and dimension",
    ),
    Recovery(
        "40d85bf2dc6f-neural-inverse-operators-for-solving-pde-inverse-problems",
        12, "image3.png", (3, 4, 5, 6, 7, 8, 9), "Figure 3", True,
    ),
    Recovery(
        "40d85bf2dc6f-neural-inverse-operators-for-solving-pde-inverse-problems",
        12, "image4.png", (11, 12, 13, 14, 15, 16, 17), "Figure 4", True,
    ),
    Recovery(
        "40d85bf2dc6f-neural-inverse-operators-for-solving-pde-inverse-problems",
        13, "image1.png", (0, 1), "Figure 5", True,
    ),
    Recovery(
        "40d85bf2dc6f-neural-inverse-operators-for-solving-pde-inverse-problems",
        13, "image2.png", (4, 5, 6, 7, 8, 9, 10), "Figure 6", True,
    ),
    Recovery(
        "40d85bf2dc6f-neural-inverse-operators-for-solving-pde-inverse-problems",
        14, "image1.png", (0,), "Figure 7",
    ),
    Recovery(
        "40d85bf2dc6f-neural-inverse-operators-for-solving-pde-inverse-problems",
        14, "image2.png", (2, 3), "Figure 8", True,
    ),
    Recovery(
        "97b7456a46d7-on-searching-for-generalized-instrumental-variables",
        8, "image-url", (0,), "Figure 6 - generalized instrumental set constructed from a 3-SAT instance",
    ),
    Recovery(
        "a375a1a3fd27-the-schrdinger-bridge-between-gaussian-measures-has-a-closed-form",
        26, "image1.png", (0,), "Principal-component visualizations for the two datasets",
    ),
    Recovery(
        "a375a1a3fd27-the-schrdinger-bridge-between-gaussian-measures-has-a-closed-form",
        26, "image2.png", (), "", remove=True,
    ),
    Recovery(
        "b34ff65ab4a6-noisy-natural-gradient-as-variational-inference",
        7, "image", (8,), "Figure 2 - reliability diagrams for K-FAC and noisy K-FAC",
    ),
    Recovery(
        "c057154ae66e-nonparametric-estimation-of-hawkes-processes-with-rkhss",
        2, "image-placeholder", (5,), "Figure 1 - Hawkes triggering-kernel estimates",
    ),
    Recovery(
        "c057154ae66e-nonparametric-estimation-of-hawkes-processes-with-rkhss",
        26, "image.png", (0,), "Figure 8 - interaction functions learned on the neuronal subnetwork",
    ),
    Recovery(
        "ebcb9a67b176-modelling-microstructure-noise-with-mutually-exciting-point-processes",
        24, "figure_8a.png", (0,), "Figure 8a - Bund front-month empirical distribution of x",
    ),
    Recovery(
        "ebcb9a67b176-modelling-microstructure-noise-with-mutually-exciting-point-processes",
        24, "figure_8b.png", (1,), "Figure 8b - Bobl empirical distribution of x",
    ),
    Recovery(
        "fb90546b95c5-spatially-relaxed-inference-on-high-dimensional-linear-models",
        28, "figure-6.png", (1,), "Figure 6 - simulation results across parameters",
    ),
)


def chunk_for_page(rows: list[dict[str, str]], document_id: str, page: int) -> dict[str, str]:
    for row in rows:
        if (
            row["document_id"] == document_id
            and int(row["start_page"]) <= page <= int(row["end_page"])
        ):
            return row
    raise KeyError(f"No OCR chunk for {document_id} page {page}")


def download_image(url: str, session: requests.Session) -> Image.Image:
    response = session.get(url, timeout=120)
    response.raise_for_status()
    with Image.open(io.BytesIO(response.content)) as image:
        image.load()
        return image.convert("RGBA")


def reconstruct(elements: list[dict[str, object]], indices: tuple[int, ...], session: requests.Session) -> Image.Image:
    selected = [next(e for e in elements if int(e["index"]) == index) for index in indices]
    images = [download_image(str(element["content"]), session) for element in selected]
    if len(images) == 1:
        return images[0]

    boxes = [element.get("bbox_2d") for element in selected]
    if any(not isinstance(box, list) or len(box) != 4 for box in boxes):
        # Old responses without geometry: retain every crop in a vertical plate.
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        canvas = Image.new("RGBA", (width, height), "white")
        y = 0
        for image in images:
            canvas.alpha_composite(image, ((width - image.width) // 2, y))
            y += image.height
        return canvas

    typed_boxes = [tuple(int(value) for value in box) for box in boxes]  # type: ignore[arg-type]
    left = min(box[0] for box in typed_boxes)
    top = min(box[1] for box in typed_boxes)
    right = max(box[2] for box in typed_boxes)
    bottom = max(box[3] for box in typed_boxes)
    canvas = Image.new("RGBA", (right - left, bottom - top), "white")
    for image, box in zip(images, typed_boxes):
        target_size = (max(1, box[2] - box[0]), max(1, box[3] - box[1]))
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        canvas.alpha_composite(image, (box[0] - left, box[1] - top))
    return canvas


def replace_one(text: str, target: str, replacement: str) -> str:
    pattern = rf"!\[[^\]]*\]\({re.escape(target)}\)"
    text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected exactly one remaining Markdown target: {target}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--only-document", action="append", default=[])
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    rows = list(csv.DictReader((run_dir / "ocr_manifest.csv").open(encoding="utf-8")))
    markdown_dir = run_dir / "markdown_verified_localized"
    assets_dir = run_dir / "assets_verified"
    cache: dict[tuple[str, int], list[dict[str, object]]] = {}
    documents: dict[str, str] = {}
    manifest = run_dir / "legacy_recovered_figure_manifest.csv"
    records: list[dict[str, object]] = []
    if manifest.is_file():
        records.extend(csv.DictReader(manifest.open(encoding="utf-8")))
    session = requests.Session()
    session.headers["User-Agent"] = "QuantLab-Legacy-Figure-Recovery/1.0"

    selected_documents = set(args.only_document)
    for recovery in RECOVERIES:
        if selected_documents and recovery.document_id not in selected_documents:
            continue
        markdown_path = markdown_dir / f"{recovery.document_id}.md"
        text = documents.setdefault(recovery.document_id, markdown_path.read_text(encoding="utf-8"))
        if recovery.remove:
            documents[recovery.document_id] = replace_one(text, recovery.target, "")
            continue

        key = (recovery.document_id, recovery.page)
        if key not in cache:
            row = chunk_for_page(rows, recovery.document_id, recovery.page)
            raw = json.loads(Path(row["raw_response_path"]).read_text(encoding="utf-8"))
            cache[key] = raw["layout_details"][recovery.page - int(row["start_page"])]
        elements = cache[key]
        image = reconstruct(elements, recovery.source_indices, session)
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG", optimize=True)
        data = output.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        indices = "-".join(f"{index:03d}" for index in recovery.source_indices)
        target = (
            assets_dir
            / recovery.document_id
            / f"recovered-p{recovery.page:04d}-e{indices}__{digest[:12]}.png"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        with Image.open(target) as verified:
            verified.verify()
        reference = f"../{target.relative_to(run_dir).as_posix()}"
        replacement = f"![{recovery.alt}]({reference})"
        documents[recovery.document_id] = replace_one(text, recovery.target, replacement)
        selected = [next(e for e in elements if int(e["index"]) == index) for index in recovery.source_indices]
        records.append(
            {
                "document_id": recovery.document_id,
                "page_number": recovery.page,
                "source_indices": ";".join(str(index) for index in recovery.source_indices),
                "source_urls": ";".join(str(element["content"]) for element in selected),
                "local_path": str(target),
                "sha256": digest,
                "bytes": len(data),
                "width": image.width,
                "height": image.height,
            }
        )

    for document_id, text in documents.items():
        (markdown_dir / f"{document_id}.md").write_text(text, encoding="utf-8")

    fields = [
        "document_id", "page_number", "source_indices", "source_urls", "local_path",
        "sha256", "bytes", "width", "height",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Recovered {len(records)} figure assets across {len(documents)} documents")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
