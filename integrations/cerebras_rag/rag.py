#!/usr/bin/env python3
"""Read-only local RAG for the allowlisted RaaS Motion math research notes.

Source Markdown is never modified. The SQLite index, embeddings, labels, and
run metadata live under ~/.local/share/raas-cerebras instead of the Obsidian
vault. Cerebras receives only allowlisted text for labeling or top-ranked
evidence for reranking and synthesis.
"""

from __future__ import annotations

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = APP_DIR / "config.json"
LABEL_PROMPT_VERSION = "research-label-v2-paper-digest"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    required = {
        "source_root",
        "allowed_globs",
        "cloud_allowed_globs",
        "deny_terms",
        "database_path",
        "manifest_path",
        "embedding_model",
        "embedding_cache",
        "cerebras_model",
        "keychain_account",
        "keychain_service",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"Config is missing: {', '.join(missing)}")
    return cfg


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_keychain_key(cfg: dict[str, Any]) -> str:
    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-a",
            cfg["keychain_account"],
            "-s",
            cfg["keychain_service"],
            "-w",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    key = result.stdout.strip()
    if not key.startswith("csk-"):
        raise RuntimeError("Keychain entry does not look like a Cerebras API key")
    return key


def discover_documents(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = Path(cfg["source_root"]).resolve()
    deny = re.compile("|".join(re.escape(term) for term in cfg["deny_terms"]), re.I)
    max_bytes = int(cfg.get("max_file_bytes", 524288))
    candidates: set[Path] = set()
    for pattern in cfg["allowed_globs"]:
        candidates.update(root.glob(pattern))

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for path in sorted(candidates):
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            excluded.append({"path": str(path), "reason": "outside source root"})
            continue
        if not resolved.is_file() or resolved.suffix.lower() != ".md":
            excluded.append({"path": str(path), "reason": "not a Markdown file"})
            continue
        size = resolved.stat().st_size
        if size > max_bytes:
            excluded.append({"path": str(path), "reason": f"larger than {max_bytes} bytes"})
            continue
        rel_path = resolved.relative_to(root).as_posix()
        text = resolved.read_text(encoding="utf-8", errors="replace")
        cloud_allowed = any(
            Path(rel_path).match(pattern) for pattern in cfg["cloud_allowed_globs"]
        )
        deny_match = deny.search(rel_path) or deny.search(text)
        cloud_block_reason: str | None = None
        if deny_match:
            cloud_allowed = False
            cloud_block_reason = f"restricted term: {deny_match.group(0)}"
        elif not cloud_allowed:
            cloud_block_reason = "local-only corpus; outbound license not approved"
        parts = Path(rel_path).parts
        specialist = parts[1] if len(parts) >= 3 and parts[0] == "specialists" else "unknown"
        title_match = re.search(r"^#\s+(.+)$", text, re.M)
        title = title_match.group(1).strip() if title_match else resolved.stem
        accepted.append(
            {
                "path": resolved,
                "rel_path": rel_path,
                "specialist": specialist,
                "title": title,
                "text": text,
                "size": size,
                "mtime_ns": resolved.stat().st_mtime_ns,
                "sha256": sha256_text(text),
                "cloud_allowed": cloud_allowed,
                "cloud_block_reason": cloud_block_reason,
            }
        )
    return accepted, excluded


def split_markdown(text: str, max_chars: int, overlap: int) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = "Document"
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
        if match and buffer:
            body = "\n".join(buffer).strip()
            if body:
                sections.append((heading, body))
            heading = match.group(2).strip()
            buffer = [line]
        else:
            if match:
                heading = match.group(2).strip()
            buffer.append(line)
    body = "\n".join(buffer).strip()
    if body:
        sections.append((heading, body))

    chunks: list[tuple[str, str]] = []
    step = max(1, max_chars - overlap)
    for section_heading, section_text in sections:
        if len(section_text) <= max_chars:
            chunks.append((section_heading, section_text))
            continue
        start = 0
        part = 1
        while start < len(section_text):
            end = min(len(section_text), start + max_chars)
            if end < len(section_text):
                boundary = section_text.rfind("\n\n", start + max_chars // 2, end)
                if boundary > start:
                    end = boundary
            chunk = section_text[start:end].strip()
            if chunk:
                chunks.append((f"{section_heading} (part {part})", chunk))
            if end >= len(section_text):
                break
            start = max(start + 1, end - overlap)
            part += 1
    return chunks


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            rel_path TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            specialist TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            byte_size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            indexed_at TEXT NOT NULL,
            cloud_allowed INTEGER NOT NULL,
            cloud_block_reason TEXT,
            labels_json TEXT,
            label_model TEXT,
            label_prompt_version TEXT,
            labeled_at TEXT
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            heading TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            UNIQUE(doc_id, ordinal)
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            title,
            heading,
            specialist,
            content,
            labels_text,
            tokenize='porter unicode61'
        );
        CREATE TABLE runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation TEXT NOT NULL,
            model TEXT,
            source_count INTEGER,
            chunk_count INTEGER,
            details_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE findings (
            finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            draft_answer TEXT,
            model TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            verification_status TEXT NOT NULL DEFAULT 'unverified',
            verification_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE vision_enrichments (
            image_id TEXT PRIMARY KEY,
            chunk_id TEXT NOT NULL,
            image_rel_path TEXT NOT NULL,
            image_sha256 TEXT NOT NULL,
            kind TEXT NOT NULL,
            caption TEXT NOT NULL,
            equations_json TEXT NOT NULL,
            visible_text TEXT NOT NULL,
            table_text TEXT NOT NULL,
            confidence REAL NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX vision_enrichments_chunk_idx ON vision_enrichments(chunk_id);
        """
    )


def ensure_runtime_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS findings (
            finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            draft_answer TEXT,
            model TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            verification_status TEXT NOT NULL DEFAULT 'unverified',
            verification_json TEXT,
            created_at TEXT NOT NULL
        )"""
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(findings)")}
    if "draft_answer" not in columns:
        conn.execute("ALTER TABLE findings ADD COLUMN draft_answer TEXT")
        conn.execute("UPDATE findings SET draft_answer=answer WHERE draft_answer IS NULL")
    if "verification_status" not in columns:
        conn.execute(
            "ALTER TABLE findings ADD COLUMN verification_status TEXT NOT NULL DEFAULT 'legacy_unverified'"
        )
    if "verification_json" not in columns:
        conn.execute("ALTER TABLE findings ADD COLUMN verification_json TEXT")
    conn.execute(
        "UPDATE findings SET verification_status='review_required' WHERE verification_status='corrected'"
    )
    has_vision = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vision_enrichments'"
    ).fetchone()
    if has_vision:
        vision_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(vision_enrichments)")
        }
        if "prompt_version" not in vision_columns:
            conn.execute("ALTER TABLE vision_enrichments ADD COLUMN prompt_version TEXT")


def embedding_model(cfg: dict[str, Any]):
    from fastembed import TextEmbedding

    cache = Path(cfg["embedding_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    return TextEmbedding(model_name=cfg["embedding_model"], cache_dir=str(cache))


def vector_blob(values: Iterable[float]) -> tuple[bytes, int]:
    data = array("f", (float(value) for value in values))
    return data.tobytes(), len(data)


def command_audit(cfg: dict[str, Any]) -> int:
    docs, excluded = discover_documents(cfg)
    cloud_allowed = sum(1 for doc in docs if doc["cloud_allowed"])
    print(f"Source root: {cfg['source_root']}")
    print(f"Locally accepted Markdown files: {len(docs)}")
    print(f"Cerebras outbound-eligible files: {cloud_allowed}")
    print(f"Local-only files: {len(docs) - cloud_allowed}")
    print(f"Accepted bytes: {sum(doc['size'] for doc in docs)}")
    print(f"Excluded files: {len(excluded)}")
    for item in excluded:
        print(f"EXCLUDED {item['path']}: {item['reason']}")
    print("Source mode: read-only")
    print(f"Index path: {cfg['database_path']}")
    return 0


def load_existing_state(db_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "labels": {},
        "findings": [],
        "runs": [],
        "embeddings": {},
        "vision_enrichments": [],
    }
    if not db_path.exists():
        return state
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        ensure_runtime_schema(conn)
        conn.commit()
        for row in conn.execute(
            """SELECT rel_path, labels_json, label_model, label_prompt_version, labeled_at
            FROM documents WHERE labels_json IS NOT NULL"""
        ):
            state["labels"][row["rel_path"]] = dict(row)
        state["findings"] = [
            dict(row)
            for row in conn.execute(
                """SELECT query, answer, draft_answer, model, evidence_json,
                          verification_status, verification_json, created_at
                FROM findings ORDER BY finding_id"""
            )
        ]
        state["runs"] = [
            dict(row)
            for row in conn.execute(
                """SELECT operation, model, source_count, chunk_count, details_json, created_at
                FROM runs ORDER BY run_id"""
            )
        ]
        has_vision = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vision_enrichments'"
        ).fetchone()
        if has_vision:
            state["vision_enrichments"] = [
                dict(row)
                for row in conn.execute(
                    """SELECT image_id, chunk_id, image_rel_path, image_sha256, kind,
                              caption, equations_json, visible_text, table_text, confidence,
                              model, prompt_version, updated_at
                       FROM vision_enrichments ORDER BY image_id"""
                )
            ]
        last_build = next(
            (row for row in reversed(state["runs"]) if row["operation"] == "build"),
            None,
        )
        prior_model = None
        if last_build and last_build["details_json"]:
            prior_model = json.loads(last_build["details_json"]).get("embedding_model")
        if prior_model == cfg["embedding_model"]:
            state["embeddings"] = {
                row["chunk_id"]: (row["embedding"], row["embedding_dim"])
                for row in conn.execute(
                    "SELECT chunk_id, embedding, embedding_dim FROM chunks"
                )
            }
    finally:
        conn.close()
    return state


def command_build(cfg: dict[str, Any]) -> int:
    docs, excluded = discover_documents(cfg)
    if not docs:
        raise RuntimeError("The allowlist produced no safe documents")

    db_path = Path(cfg["database_path"])
    existing = load_existing_state(db_path, cfg)
    chunk_rows: list[dict[str, Any]] = []
    for doc in docs:
        doc_id = sha256_text(doc["rel_path"])[:24]
        doc["doc_id"] = doc_id
        pieces = split_markdown(
            doc["text"],
            int(cfg.get("max_chunk_chars", 3600)),
            int(cfg.get("chunk_overlap_chars", 400)),
        )
        for ordinal, (heading, content) in enumerate(pieces):
            chunk_id = sha256_text(f"{doc['rel_path']}\n{ordinal}\n{content}")[:24]
            chunk_rows.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "ordinal": ordinal,
                    "heading": heading,
                    "content": content,
                    "title": doc["title"],
                    "specialist": doc["specialist"],
                }
            )
    doc_rel_by_id = {doc["doc_id"]: doc["rel_path"] for doc in docs}
    active_chunk_ids = {row["chunk_id"] for row in chunk_rows}
    vision_rows = [
        row for row in existing["vision_enrichments"] if row["chunk_id"] in active_chunk_ids
    ]
    vision_by_chunk: dict[str, list[str]] = {}
    for item in vision_rows:
        attached = [item["caption"], item["visible_text"], item["table_text"]]
        try:
            attached.extend(json.loads(item["equations_json"]))
        except (json.JSONDecodeError, TypeError):
            pass
        vision_by_chunk.setdefault(item["chunk_id"], []).extend(
            str(value) for value in attached if value
        )

    reusable = existing["embeddings"]
    missing_rows = [row for row in chunk_rows if row["chunk_id"] not in reusable]
    print(
        f"Reusing {len(chunk_rows) - len(missing_rows)} local embeddings; "
        f"embedding {len(missing_rows)} new chunks with {cfg['embedding_model']} ...",
        flush=True,
    )
    new_vectors: list[Iterable[float]] = []
    if missing_rows:
        model = embedding_model(cfg)
        new_vectors = list(
            model.embed([row["content"] for row in missing_rows], batch_size=32)
        )
    if len(new_vectors) != len(missing_rows):
        raise RuntimeError("Embedding runtime returned an unexpected vector count")
    stored_vectors = dict(reusable)
    for row, vector in zip(missing_rows, new_vectors, strict=True):
        stored_vectors[row["chunk_id"]] = vector_blob(vector)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = db_path.with_suffix(db_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)
    conn = sqlite3.connect(temp_path)
    try:
        init_schema(conn)
        indexed_at = utc_now()
        for doc in docs:
            conn.execute(
                """INSERT INTO documents
                (doc_id, path, rel_path, title, specialist, sha256, byte_size, mtime_ns,
                 indexed_at, cloud_allowed, cloud_block_reason, labels_json, label_model,
                 label_prompt_version, labeled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc["doc_id"],
                    str(doc["path"]),
                    doc["rel_path"],
                    doc["title"],
                    doc["specialist"],
                    doc["sha256"],
                    doc["size"],
                    doc["mtime_ns"],
                    indexed_at,
                    int(doc["cloud_allowed"]),
                    doc["cloud_block_reason"],
                    existing["labels"].get(doc["rel_path"], {}).get("labels_json"),
                    existing["labels"].get(doc["rel_path"], {}).get("label_model"),
                    existing["labels"].get(doc["rel_path"], {}).get("label_prompt_version"),
                    existing["labels"].get(doc["rel_path"], {}).get("labeled_at"),
                ),
            )
        for row in chunk_rows:
            blob, dimension = stored_vectors[row["chunk_id"]]
            conn.execute(
                """INSERT INTO chunks
                (chunk_id, doc_id, ordinal, heading, content, embedding, embedding_dim)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["chunk_id"],
                    row["doc_id"],
                    row["ordinal"],
                    row["heading"],
                    row["content"],
                    blob,
                    dimension,
                ),
            )
            prior_label = existing["labels"].get(
                doc_rel_by_id[row["doc_id"]], {}
            ).get("labels_json")
            label_text = ""
            if prior_label:
                label_text = labels_as_text(json.loads(prior_label))
            attached = vision_by_chunk.get(row["chunk_id"], [])
            if attached:
                label_text += ("\n" if label_text else "") + "VISION ENRICHMENT:\n" + "\n".join(attached)
            conn.execute(
                "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row["chunk_id"],
                    row["title"],
                    row["heading"],
                    row["specialist"],
                    row["content"],
                    label_text,
                ),
            )
        conn.executemany(
            """INSERT INTO vision_enrichments
            (image_id, chunk_id, image_rel_path, image_sha256, kind, caption,
             equations_json, visible_text, table_text, confidence, model, prompt_version,
             updated_at)
            VALUES (:image_id, :chunk_id, :image_rel_path, :image_sha256, :kind, :caption,
                    :equations_json, :visible_text, :table_text, :confidence, :model,
                    :prompt_version, :updated_at)""",
            vision_rows,
        )
        conn.executemany(
            """INSERT INTO findings
            (query, answer, draft_answer, model, evidence_json, verification_status,
             verification_json, created_at)
            VALUES (:query, :answer, :draft_answer, :model, :evidence_json,
                    :verification_status, :verification_json, :created_at)""",
            existing["findings"],
        )
        conn.executemany(
            """INSERT INTO runs
            (operation, model, source_count, chunk_count, details_json, created_at)
            VALUES (:operation, :model, :source_count, :chunk_count, :details_json, :created_at)""",
            existing["runs"],
        )
        conn.execute(
            """INSERT INTO runs
            (operation, source_count, chunk_count, details_json, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                "build",
                len(docs),
                len(chunk_rows),
                json.dumps(
                    {
                        "embedding_model": cfg["embedding_model"],
                        "excluded": excluded,
                        "cloud_allowed": sum(1 for doc in docs if doc["cloud_allowed"]),
                    }
                ),
                indexed_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    os.replace(temp_path, db_path)

    manifest = {
        "created_at": utc_now(),
        "source_root": cfg["source_root"],
        "source_mode": "read-only",
        "allowed_globs": cfg["allowed_globs"],
        "cloud_allowed_globs": cfg["cloud_allowed_globs"],
        "deny_terms": cfg["deny_terms"],
        "embedding_model": cfg["embedding_model"],
        "document_count": len(docs),
        "cloud_allowed_count": sum(1 for doc in docs if doc["cloud_allowed"]),
        "local_only_count": sum(1 for doc in docs if not doc["cloud_allowed"]),
        "chunk_count": len(chunk_rows),
        "documents": [
            {
                "doc_id": doc["doc_id"],
                "rel_path": doc["rel_path"],
                "sha256": doc["sha256"],
                "byte_size": doc["size"],
                "cloud_allowed": doc["cloud_allowed"],
                "cloud_block_reason": doc["cloud_block_reason"],
            }
            for doc in docs
        ],
        "excluded": excluded,
    }
    manifest_path = Path(cfg["manifest_path"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built {db_path} from {len(docs)} documents and {len(chunk_rows)} chunks.")
    return 0


LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "primary_role": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "methods": {"type": "array", "items": {"type": "string"}},
        "research_tasks": {"type": "array", "items": {"type": "string"}},
        "assets_or_domains": {"type": "array", "items": {"type": "string"}},
        "failure_modes": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "data_classification": {"type": "string", "enum": ["non_lseg_research"]},
    },
    "required": [
        "summary",
        "primary_role",
        "topics",
        "methods",
        "research_tasks",
        "assets_or_domains",
        "failure_modes",
        "confidence",
        "data_classification",
    ],
    "additionalProperties": False,
}


def label_one(cfg: dict[str, Any], key: str, row: sqlite3.Row) -> tuple[str, dict[str, Any]]:
    from cerebras.cloud.sdk import Cerebras

    text = Path(row["path"]).read_text(encoding="utf-8", errors="replace")
    prompt = (
        "Label this approved, non-LSEG quantitative-research Markdown document. "
        "Describe only what the document supports. Use concise taxonomy terms. "
        "Confidence must be between 0 and 1. Do not provide trading advice.\n\n"
        f"FILE: {row['rel_path']}\nTITLE: {row['title']}\n\n{text[:24000]}"
    )
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            client = Cerebras(api_key=key)
            response = client.chat.completions.create(
                model=cfg["cerebras_model"],
                messages=[
                    {
                        "role": "system",
                        "content": "You are a conservative research-document taxonomist.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "research_document_label",
                        "strict": True,
                        "schema": LABEL_SCHEMA,
                    },
                },
                reasoning_effort="low",
                temperature=0,
                max_completion_tokens=1400,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Cerebras returned no structured label")
            label = json.loads(content)
            if label.get("data_classification") != "non_lseg_research":
                raise RuntimeError("Unexpected data classification")
            return row["doc_id"], label
        except Exception as exc:
            last_error = exc
            status = getattr(exc, "status_code", None)
            if attempt == 4 or status not in {429, 500, 502, 503, 504}:
                break
            time.sleep(1.5 * (2**attempt))
    assert last_error is not None
    raise last_error


def labels_as_text(label: dict[str, Any]) -> str:
    values = [label.get("summary", ""), label.get("primary_role", "")]
    for field in ("topics", "methods", "research_tasks", "assets_or_domains", "failure_modes"):
        values.extend(label.get(field, []))
    return "\n".join(str(value) for value in values if value)


def retry_cerebras_call(callback, attempts: int = 6):
    """Retry only transient Cerebras quota/server responses."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return callback()
        except Exception as exc:
            last_error = exc
            status = getattr(exc, "status_code", None)
            if attempt == attempts - 1 or status not in {429, 500, 502, 503, 504}:
                raise
            if status == 429:
                message = str(exc).lower()
                if "per hour" in message or "per day" in message:
                    raise
                response = getattr(exc, "response", None)
                headers = getattr(response, "headers", {})
                raw_reset = headers.get("x-ratelimit-reset-tokens-minute", "")
                try:
                    delay = min(60.0, max(5.0, float(raw_reset) + 1.0))
                except (TypeError, ValueError):
                    delay = 35.0
                print(f"RATE LIMIT: retrying in {delay:.0f}s", file=sys.stderr, flush=True)
            else:
                delay = 1.5 * (2**attempt)
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def paper_label_digest(text: str, max_chars: int = 5200) -> str:
    """Compact a paper into abstract/front matter, headings, and closing analysis."""
    front = text[:3000]
    headings = "\n".join(
        line.strip()
        for line in text.splitlines()
        if re.match(r"^#{1,6}\s+\S", line.strip())
    )[:1000]
    closing_matches = list(
        re.finditer(
            r"(?im)^#{1,6}\s+.*\b(conclusions?|discussion|summary|limitations?)\b.*$",
            text,
        )
    )
    if closing_matches:
        closing = text[closing_matches[-1].start() :][:1200]
    else:
        closing = text[-1000:]
    digest = "\n\n".join(
        [
            "FRONT MATTER / ABSTRACT / INTRODUCTION:\n" + front,
            "DOCUMENT HEADINGS:\n" + headings,
            "CLOSING SECTION:\n" + closing,
        ]
    )
    return digest[:max_chars]


def label_batch(
    cfg: dict[str, Any], key: str, rows: list[sqlite3.Row]
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int]]:
    """Label several approved documents in one strict, independently keyed response."""
    from cerebras.cloud.sdk import Cerebras

    doc_ids = [str(row["doc_id"]) for row in rows]
    item_schema = {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string", "enum": doc_ids},
            **LABEL_SCHEMA["properties"],
        },
        "required": ["doc_id", *LABEL_SCHEMA["required"]],
        "additionalProperties": False,
    }
    batch_schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": item_schema,
            }
        },
        "required": ["labels"],
        "additionalProperties": False,
    }

    documents: list[str] = []
    for row in rows:
        document_text = Path(row["path"]).read_text(encoding="utf-8", errors="replace")
        label_text = (
            paper_label_digest(document_text)
            if "/papers/" in str(row["rel_path"])
            else document_text[:24000]
        )
        documents.append(
            "\n".join(
                [
                    "<DOCUMENT>",
                    f"DOC_ID: {row['doc_id']}",
                    f"FILE: {row['rel_path']}",
                    f"TITLE: {row['title']}",
                    "CONTENT DIGEST:",
                    label_text,
                    "</DOCUMENT>",
                ]
            )
        )
    prompt = (
        "Label each approved, non-LSEG quantitative-research Markdown document "
        "independently. Return exactly one label for every DOC_ID and copy each "
        "DOC_ID exactly. Describe only what that document supports, use concise "
        "taxonomy terms, and do not provide trading advice. Confidence must be "
        "between 0 and 1.\n\n" + "\n\n".join(documents)
    )

    client = Cerebras(api_key=key, timeout=75.0, max_retries=0)

    def make_request():
        return client.chat.completions.create(
            model=cfg["cerebras_model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a conservative research-document taxonomist.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "research_document_label_batch",
                    "strict": True,
                    "schema": batch_schema,
                },
            },
            reasoning_effort="low",
            temperature=0,
            max_completion_tokens=max(1000, 600 * len(rows)),
        )

    response = retry_cerebras_call(make_request, attempts=3)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Cerebras returned no structured labels")
    payload = json.loads(content)
    labels = payload.get("labels", [])
    if len(labels) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} labels; received {len(labels)}")

    expected = set(doc_ids)
    received: set[str] = set()
    completed: list[tuple[str, dict[str, Any]]] = []
    for label_with_id in labels:
        doc_id = str(label_with_id.get("doc_id", ""))
        if doc_id not in expected or doc_id in received:
            raise RuntimeError(f"Unexpected or duplicate DOC_ID: {doc_id!r}")
        received.add(doc_id)
        label = {key: value for key, value in label_with_id.items() if key != "doc_id"}
        if label.get("data_classification") != "non_lseg_research":
            raise RuntimeError(f"Unexpected data classification for {doc_id}")
        confidence = label.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise RuntimeError(f"Invalid confidence for {doc_id}")
        completed.append((doc_id, label))
    if received != expected:
        raise RuntimeError("Cerebras response omitted one or more DOC_ID values")

    usage_obj = getattr(response, "usage", None)
    usage = {
        "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
    }
    return completed, usage


def store_label_batch(
    conn: sqlite3.Connection,
    cfg: dict[str, Any],
    completed: list[tuple[str, dict[str, Any]]],
) -> None:
    """Persist a finished batch immediately so interrupted runs are resumable."""
    labeled_at = utc_now()
    for doc_id, label in completed:
        payload = json.dumps(label, ensure_ascii=False, sort_keys=True)
        conn.execute(
            """UPDATE documents
            SET labels_json=?, label_model=?, label_prompt_version=?, labeled_at=?
            WHERE doc_id=?""",
            (payload, cfg["cerebras_model"], LABEL_PROMPT_VERSION, labeled_at, doc_id),
        )
        conn.execute(
            """UPDATE chunks_fts SET labels_text=?
            WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id=?)""",
            (labels_as_text(label), doc_id),
        )
    conn.commit()


def command_label(
    cfg: dict[str, Any], workers: int, limit: int | None, batch_size: int, min_interval: float
) -> int:
    db_path = Path(cfg["database_path"])
    if not db_path.exists():
        raise RuntimeError("Build the index before labeling")
    if batch_size < 1 or batch_size > 8:
        raise ValueError("--batch-size must be between 1 and 8")
    if min_interval < 0:
        raise ValueError("--min-interval must be non-negative")
    if workers != 1:
        print("Quota-safe labeling uses one sequential worker.", flush=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT doc_id, path, rel_path, title
        FROM documents
        WHERE labels_json IS NULL AND cloud_allowed=1
        ORDER BY rel_path"""
    ).fetchall()
    conn.close()
    if limit is not None:
        rows = rows[:limit]

    if not rows:
        print("No unlabeled Cerebras-eligible documents remain.")
        return 0

    key = read_keychain_key(cfg)
    batches = [rows[offset : offset + batch_size] for offset in range(0, len(rows), batch_size)]
    completed_count = 0
    failures: list[tuple[str, str]] = []
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    write_conn = sqlite3.connect(db_path)
    last_started: float | None = None
    for batch_number, batch in enumerate(batches, start=1):
        if last_started is not None:
            remaining_wait = min_interval - (time.monotonic() - last_started)
            if remaining_wait > 0:
                print(
                    f"QUOTA WAIT {remaining_wait:.0f}s before batch "
                    f"{batch_number}/{len(batches)}",
                    flush=True,
                )
                time.sleep(remaining_wait)
        last_started = time.monotonic()
        try:
            completed, usage = label_batch(cfg, key, batch)
            store_label_batch(write_conn, cfg, completed)
            completed_count += len(completed)
            for field in token_usage:
                token_usage[field] += usage[field]
            for row in batch:
                print(f"LABELED {row['rel_path']}", flush=True)
            print(
                f"PROGRESS {completed_count}/{len(rows)} documents "
                f"({len(failures)} failed documents)",
                flush=True,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failures.extend((row["rel_path"], error) for row in batch)
            print(
                f"FAILED BATCH ({len(batch)} documents): {error}",
                file=sys.stderr,
                flush=True,
            )

    finished_at = utc_now()
    try:
        write_conn.execute(
            """INSERT INTO runs
            (operation, model, source_count, details_json, created_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                "label",
                cfg["cerebras_model"],
                completed_count,
                json.dumps(
                    {
                        "failures": failures,
                        "prompt_version": LABEL_PROMPT_VERSION,
                        "batch_size": batch_size,
                        "min_interval_seconds": min_interval,
                        "token_usage": token_usage,
                    }
                ),
                finished_at,
            ),
        )
        write_conn.commit()
    finally:
        write_conn.close()
    print(
        f"Stored {completed_count} labels locally; failed documents: {len(failures)}; "
        f"tokens: {token_usage['total_tokens']}"
    )
    return 1 if failures else 0


def fts_query(text: str) -> str:
    terms = re.findall(r"[A-Za-z0-9_]{2,}", text.lower())
    unique = list(dict.fromkeys(terms))[:20]
    if not unique:
        raise ValueError("Query contains no searchable terms")
    return " OR ".join(f'"{term}"' for term in unique)


def decode_vector(blob: bytes) -> array:
    values = array("f")
    values.frombytes(blob)
    return values


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    denominator = math.sqrt(left_norm) * math.sqrt(right_norm)
    return dot / denominator if denominator else 0.0


def retrieve(
    cfg: dict[str, Any], query: str, limit: int = 12, cloud_only: bool = False
) -> list[dict[str, Any]]:
    db_path = Path(cfg["database_path"])
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cloud_filter = " AND d.cloud_allowed=1" if cloud_only else ""
    lexical = conn.execute(
        f"""SELECT c.chunk_id, c.doc_id, c.heading, c.content, d.title, d.specialist,
                  d.rel_path, d.labels_json, d.cloud_allowed,
                  bm25(chunks_fts, 0.0, 2.5, 2.0, 1.5, 1.0, 3.0) AS score
           FROM chunks_fts
           JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
           JOIN documents d ON d.doc_id = c.doc_id
           WHERE chunks_fts MATCH ?{cloud_filter}
           ORDER BY score
           LIMIT ?""",
        (fts_query(query), max(limit * 2, 20)),
    ).fetchall()

    model = embedding_model(cfg)
    query_vector = list(model.query_embed(query))[0]
    vector_rows = conn.execute(
        f"""SELECT c.chunk_id, c.doc_id, c.heading, c.content, c.embedding,
                  d.title, d.specialist, d.rel_path, d.labels_json, d.cloud_allowed
           FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
           WHERE 1=1{cloud_filter}"""
    ).fetchall()
    conn.close()
    vector_ranked = sorted(
        ((cosine(query_vector, decode_vector(row["embedding"])), row) for row in vector_rows),
        key=lambda item: item[0],
        reverse=True,
    )[: max(limit * 2, 20)]

    fused: dict[str, dict[str, Any]] = {}
    for lane, weight in ((lexical, 1.0), ([row for _, row in vector_ranked], 1.0)):
        for rank, row in enumerate(lane, start=1):
            chunk_id = row["chunk_id"]
            item = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "doc_id": row["doc_id"],
                    "heading": row["heading"],
                    "content": row["content"],
                    "title": row["title"],
                    "specialist": row["specialist"],
                    "rel_path": row["rel_path"],
                    "labels_json": row["labels_json"],
                    "cloud_allowed": bool(row["cloud_allowed"]),
                    "rrf_score": 0.0,
                },
            )
            item["rrf_score"] += weight / (60 + rank)
    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)[:limit]


def rerank(cfg: dict[str, Any], query: str, candidates: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    from cerebras.cloud.sdk import Cerebras

    key = read_keychain_key(cfg)
    ids = [item["chunk_id"] for item in candidates]
    evidence = "\n\n".join(
        f"ID={item['chunk_id']}\nTITLE={item['title']}\nHEADING={item['heading']}\n{item['content'][:1800]}"
        for item in candidates
    )
    schema = {
        "type": "object",
        "properties": {
            "ranked_ids": {
                "type": "array",
                "items": {"type": "string", "enum": ids},
            }
        },
        "required": ["ranked_ids"],
        "additionalProperties": False,
    }
    client = Cerebras(api_key=key)
    response = retry_cerebras_call(
        lambda: client.chat.completions.create(
            model=cfg["cerebras_model"],
            messages=[
                {
                    "role": "system",
                    "content": "Rank evidence for direct relevance. Return each useful ID once and do not invent IDs.",
                },
                {"role": "user", "content": f"QUESTION:\n{query}\n\nCANDIDATES:\n{evidence}"},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "evidence_ranking", "strict": True, "schema": schema},
            },
            reasoning_effort="low",
            temperature=0,
            max_completion_tokens=1200,
        )
    )
    content = response.choices[0].message.content
    ranked_ids = json.loads(content or "{}").get("ranked_ids", [])
    by_id = {item["chunk_id"]: item for item in candidates}
    ranked = [by_id[item_id] for item_id in ranked_ids if item_id in by_id]
    ranked.extend(item for item in candidates if item["chunk_id"] not in {x["chunk_id"] for x in ranked})
    return ranked[:keep]


def verify_answer(
    cfg: dict[str, Any],
    query: str,
    context: str,
    draft: str,
    valid_citations: set[str],
) -> dict[str, Any]:
    from cerebras.cloud.sdk import Cerebras

    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["pass", "review_required", "fail"]},
            "issues": {"type": "array", "items": {"type": "string"}},
            "final_answer": {"type": "string"},
        },
        "required": ["status", "issues", "final_answer"],
        "additionalProperties": False,
    }
    key = read_keychain_key(cfg)
    client = Cerebras(api_key=key)
    response = retry_cerebras_call(
        lambda: client.chat.completions.create(
            model=cfg["cerebras_model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Audit the draft against the supplied evidence. Every substantive claim must be "
                        "entailed by a cited source. Do not expand acronyms unless the evidence expands them. "
                        "Replace words such as guarantees, proves, or eliminates when the evidence is weaker. "
                        "Do not add outside knowledge. Return pass only if the draft needs no substantive "
                        "change. If you change, remove, or qualify any claim, return review_required with the "
                        "repaired answer. If evidence is insufficient, return fail and an insufficiency answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"QUESTION:\n{query}\n\nEVIDENCE:\n{context}\n\nDRAFT:\n{draft}",
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "grounding_verification", "strict": True, "schema": schema},
            },
            reasoning_effort="medium",
            temperature=0,
            max_completion_tokens=2200,
        )
    )
    content = response.choices[0].message.content
    verification = json.loads(content or "{}")
    if verification.get("issues") and verification.get("status") == "pass":
        verification["status"] = "review_required"
    final_answer = str(verification.get("final_answer", "")).strip()
    cited = {f"S{number}" for number in re.findall(r"S(\d+)", final_answer)}
    unknown = sorted(cited - valid_citations)
    if unknown:
        verification.setdefault("issues", []).append(
            f"Unknown citations in final answer: {', '.join(unknown)}"
        )
        verification["status"] = "fail"
        verification["final_answer"] = "The draft failed citation validation and was not promoted."
    elif not cited:
        verification.setdefault("issues", []).append("Final answer contains no source citations")
        verification["status"] = "fail"
        verification["final_answer"] = "The available answer could not be promoted because it lacked source citations."
    return verification


def command_search(cfg: dict[str, Any], query: str, limit: int) -> int:
    for index, item in enumerate(retrieve(cfg, query, limit=limit), start=1):
        print(f"[{index}] {item['title']} — {item['heading']}")
        outbound = "Cerebras-eligible" if item["cloud_allowed"] else "LOCAL-ONLY"
        print(f"    {item['rel_path']}  rrf={item['rrf_score']:.6f}  {outbound}")
        preview = re.sub(r"\s+", " ", item["content"])[:300]
        print(f"    {preview}")
    return 0


def command_ask(cfg: dict[str, Any], query: str, candidate_count: int, evidence_count: int) -> int:
    from langchain_cerebras import ChatCerebras
    from langchain_core.messages import HumanMessage, SystemMessage

    candidates = retrieve(cfg, query, limit=candidate_count, cloud_only=True)
    if not candidates:
        print("No allowlisted evidence was found.")
        return 2
    evidence = rerank(cfg, query, candidates, keep=evidence_count)
    context_blocks: list[str] = []
    for index, item in enumerate(evidence, start=1):
        item["citation"] = f"S{index}"
        context_blocks.append(
            f"<source id=\"S{index}\" path=\"{item['rel_path']}\" heading=\"{item['heading']}\">\n"
            f"{item['content']}\n</source>"
        )
    context = "\n\n".join(context_blocks)
    key = read_keychain_key(cfg)
    llm = ChatCerebras(
        model=cfg["cerebras_model"],
        api_key=key,
        temperature=0,
        max_tokens=1400,
    )
    messages = [
        SystemMessage(
                content=(
                    "Answer only from the supplied evidence. Cite every substantive claim with [S#]. "
                    "If evidence is incomplete, say so. Do not provide investment advice and do not claim "
                    "that a mathematical specialist is a human. Do not expand an acronym unless the evidence "
                    "explicitly defines it. Do not say a method guarantees, proves, or eliminates a risk unless "
                    "the evidence uses equally strong language."
                )
        ),
        HumanMessage(content=f"QUESTION:\n{query}\n\nEVIDENCE:\n{context}"),
    ]
    response = retry_cerebras_call(
        lambda: llm.invoke(messages)
    )
    draft_answer = str(response.content)
    valid_citations = {item["citation"] for item in evidence}
    verification = verify_answer(cfg, query, context, draft_answer, valid_citations)
    answer_text = str(verification["final_answer"])
    print(answer_text)
    print(f"\nVerification: {verification['status']}")
    if verification["status"] != "pass":
        print("This finding is stored for human review and is not eligible for the reward dataset.")
    for issue in verification.get("issues", []):
        print(f"- {issue}")
    print("\nSources:")
    for item in evidence:
        print(f"[{item['citation']}] {item['rel_path']} — {item['heading']}")

    conn = sqlite3.connect(cfg["database_path"])
    ensure_runtime_schema(conn)
    created_at = utc_now()
    evidence_record = [
        {
            "citation": item["citation"],
            "chunk_id": item["chunk_id"],
            "rel_path": item["rel_path"],
            "heading": item["heading"],
        }
        for item in evidence
    ]
    conn.execute(
        """INSERT INTO findings
        (query, answer, draft_answer, model, evidence_json, verification_status, verification_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            query,
            answer_text,
            draft_answer,
            cfg["cerebras_model"],
            json.dumps(evidence_record),
            verification["status"],
            json.dumps(verification),
            created_at,
        ),
    )
    conn.execute(
        """INSERT INTO runs (operation, model, source_count, details_json, created_at)
        VALUES (?, ?, ?, ?, ?)""",
        (
            "ask",
            cfg["cerebras_model"],
            len(evidence),
            json.dumps({"query_sha256": sha256_text(query), "chunk_ids": [x["chunk_id"] for x in evidence]}),
            created_at,
        ),
    )
    conn.commit()
    conn.close()
    return 0 if verification["status"] == "pass" else 3


def command_status(cfg: dict[str, Any]) -> int:
    db_path = Path(cfg["database_path"])
    if not db_path.exists():
        print("Index does not exist")
        return 1
    conn = sqlite3.connect(db_path)
    ensure_runtime_schema(conn)
    documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    cloud_allowed = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE cloud_allowed=1"
    ).fetchone()[0]
    labeled = conn.execute("SELECT COUNT(*) FROM documents WHERE labels_json IS NOT NULL").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    findings = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
    verified_findings = conn.execute(
        "SELECT COUNT(*) FROM findings WHERE verification_status='pass'"
    ).fetchone()[0]
    last_run = conn.execute(
        "SELECT operation, model, created_at FROM runs ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    print(f"Documents: {documents}")
    print(f"Cerebras outbound-eligible documents: {cloud_allowed}")
    print(f"Local-only documents: {documents - cloud_allowed}")
    print(f"Labeled documents: {labeled}")
    print(f"Chunks: {chunks}")
    print(f"Stored findings: {findings}")
    print(f"Verified findings: {verified_findings}")
    print(f"Embedding model: {cfg['embedding_model']}")
    print(f"Cerebras model: {cfg['cerebras_model']}")
    if last_run:
        print(f"Last run: {last_run[0]} / {last_run[1] or '-'} / {last_run[2]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Audit the allowlist without writing an index")
    sub.add_parser("build", help="Build a new local FTS and vector index")
    label = sub.add_parser("label", help="Generate local sidecar labels with Cerebras")
    label.add_argument("--workers", type=int, default=1)
    label.add_argument("--limit", type=int)
    label.add_argument("--batch-size", type=int, default=6)
    label.add_argument("--min-interval", type=float, default=36.0)
    search = sub.add_parser("search", help="Run local hybrid retrieval only")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)
    ask = sub.add_parser("ask", help="Retrieve, rerank, and synthesize with citations")
    ask.add_argument("query")
    ask.add_argument("--candidates", type=int, default=12)
    ask.add_argument("--evidence", type=int, default=6)
    sub.add_parser("status", help="Show index and labeling status")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "audit":
        return command_audit(cfg)
    if args.command == "build":
        return command_build(cfg)
    if args.command == "label":
        return command_label(
            cfg, args.workers, args.limit, args.batch_size, args.min_interval
        )
    if args.command == "search":
        return command_search(cfg, args.query, args.limit)
    if args.command == "ask":
        return command_ask(cfg, args.query, args.candidates, args.evidence)
    if args.command == "status":
        return command_status(cfg)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
