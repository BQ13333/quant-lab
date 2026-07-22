from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ImmutableHashInventoryArtifacts:
    inventory_path: Path
    file_count: int
    files: dict[str, str]


def write_immutable_hash_inventory(
    *,
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    inventory_id: str,
) -> ImmutableHashInventoryArtifacts:
    output = Path(output_path).resolve()
    files = _collect_files(input_paths, excluded={output})
    if not files:
        raise ValueError("immutable hash inventory has no input files")
    mapping = {str(path): _sha256(path) for path in files}
    payload = {
        "inventory_id": inventory_id,
        "algorithm": "sha256",
        "immutable": True,
        "coverage_complete": True,
        "file_count": len(mapping),
        "files": mapping,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        comparable = dict(existing)
        comparable.pop("generated_at_utc", None)
        expected = dict(payload)
        expected.pop("generated_at_utc", None)
        if comparable != expected:
            raise ValueError(f"immutable hash inventory drift: {output}")
    else:
        _atomic_write_bytes(output, encoded)
    return ImmutableHashInventoryArtifacts(
        inventory_path=output,
        file_count=len(mapping),
        files=mapping,
    )


def verify_immutable_hash_inventory(path: str | Path) -> tuple[bool, list[str]]:
    inventory_path = Path(path)
    if not inventory_path.is_file():
        return False, ["inventory_missing"]
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ["inventory_invalid_json"]
    issues: list[str] = []
    if payload.get("algorithm") != "sha256":
        issues.append("algorithm_not_sha256")
    if payload.get("immutable") is not True:
        issues.append("immutable_not_true")
    if payload.get("coverage_complete") is not True:
        issues.append("coverage_complete_not_true")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        return False, [*issues, "files_missing"]
    if int(payload.get("file_count", -1)) != len(files):
        issues.append("file_count_mismatch")
    for raw_path, expected in files.items():
        item = Path(str(raw_path))
        if not item.is_file():
            issues.append(f"missing:{item}")
        elif _sha256(item) != str(expected).lower():
            issues.append(f"digest_mismatch:{item}")
    return not issues, issues


def _collect_files(input_paths: Iterable[str | Path], excluded: set[Path]) -> list[Path]:
    collected: set[Path] = set()
    for raw in input_paths:
        path = Path(raw).resolve()
        if path.is_file():
            if path not in excluded:
                collected.add(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"hash inventory input missing: {path}")
        for item in path.rglob("*"):
            resolved = item.resolve()
            if item.is_file() and resolved not in excluded and not item.name.startswith("."):
                collected.add(resolved)
    return sorted(collected, key=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
