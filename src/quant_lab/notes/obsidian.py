from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ObsidianNote:
    title: str
    body: str
    note_type: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    backlinks: list[str] = field(default_factory=list)
    mermaid_diagram: str = ""
    dataview_query: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tags", list(dict.fromkeys(self.tags)))
        object.__setattr__(self, "backlinks", list(dict.fromkeys(self.backlinks)))


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(f'"{item}"' for item in value) + "]"
    return f'"{value}"'


def _frontmatter(note: ObsidianNote) -> str:
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    data = {
        "title": note.title,
        "type": note.note_type,
        "created": created,
        "tags": note.tags,
        **note.metadata,
    }
    if note.backlinks:
        data["backlinks"] = note.backlinks
    if note.mermaid_diagram:
        data["mermaid_diagram"] = note.mermaid_diagram
    if note.dataview_query:
        data["dataview_query"] = note.dataview_query
    lines = ["---"]
    for key, value in data.items():
        lines.append(f"{key}: {_format_value(value)}")
    lines.append("---")
    return "\n".join(lines)


def slugify(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {" ", "-", "_"}:
            allowed.append("-")
    slug = "".join(allowed).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "note"


def write_note(vault_root: str | Path, relative_path: str | Path, note: ObsidianNote) -> Path:
    path = Path(vault_root) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        counter = 1
        original = path
        while path.exists():
            path = original.parent / f"{original.stem}({counter}){original.suffix}"
            counter += 1

    path.write_text(f"{_frontmatter(note)}\n\n# {note.title}\n\n{note.body.rstrip()}\n", encoding="utf-8")
    return path
