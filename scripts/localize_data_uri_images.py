#!/usr/bin/env python3
"""Replace embedded data-URI images with durable local assets."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import re
from pathlib import Path

from PIL import Image


DATA_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\(data:(?P<mime>image/[a-zA-Z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=\r\n]+)\)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    markdown_dir = run_dir / "markdown_verified_localized"
    records: list[dict[str, object]] = []
    for markdown_path in sorted(markdown_dir.glob("*.md")):
        text = markdown_path.read_text(encoding="utf-8")

        def replace(match: re.Match[str]) -> str:
            data = base64.b64decode(match.group("data"))
            digest = hashlib.sha256(data).hexdigest()
            subtype = match.group("mime").split("/", 1)[1].lower()
            extension = "svg" if "svg" in subtype else subtype.replace("jpeg", "jpg")
            target = (
                run_dir / "assets_verified" / markdown_path.stem
                / f"embedded__{digest[:12]}.{extension}"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if extension != "svg":
                with Image.open(target) as image:
                    image.verify()
            reference = f"../{target.relative_to(run_dir).as_posix()}"
            records.append(
                {
                    "document_id": markdown_path.stem,
                    "mime_type": match.group("mime"),
                    "local_path": str(target),
                    "sha256": digest,
                    "bytes": len(data),
                }
            )
            return f"![{match.group('alt')}]({reference})"

        localized = DATA_IMAGE_RE.sub(replace, text)
        if localized != text:
            markdown_path.write_text(localized, encoding="utf-8")
    manifest = run_dir / "localized_data_uri_manifest.csv"
    fields = ["document_id", "mime_type", "local_path", "sha256", "bytes"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"Localized {len(records)} embedded data-URI images; manifest: {manifest}")


if __name__ == "__main__":
    main()
