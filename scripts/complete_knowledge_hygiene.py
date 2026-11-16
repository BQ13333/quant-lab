#!/usr/bin/env python3
"""Build, verify, or search the local Steps 9--11 completion package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import tempfile

from quant_lab.research.knowledge_completion import (
    CompletionError,
    CompletionInputs,
    build_completion_package,
    search_canonical_artifacts,
    semantic_hash,
    sha256_file,
    verify_completion_package,
)
from quant_lab.research.control_plane import (
    ApprovalTrustStore,
    ValidationRunnerTrustStore,
)
from quant_lab.research.data_admission import (
    DataAdmissionTrustStore,
    DataReleaseTrustStore,
)


DEFAULT_WORKSPACE = Path(
    os.environ.get("QUANT_LAB_WORKSPACE", str(Path.cwd()))
).expanduser().resolve()
DEFAULT_RAG_HOME = Path(
    os.environ.get(
        "QUANT_LAB_RAG_HOME",
        str(Path.home() / ".local" / "share" / "quant-lab-rag"),
    )
).expanduser().resolve()
DEFAULT_KNOWLEDGE = DEFAULT_RAG_HOME / "knowledge_layers"
DEFAULT_PAPERS = DEFAULT_RAG_HOME / "rag" / "knowledge.sqlite3"
SEARCH_CACHE_NAME = "knowledge-completion-search-verification.json"


def _source_stat_fingerprint(configured: CompletionInputs) -> str:
    paths: set[Path] = set()
    for directory, _, filenames in os.walk(configured.workspace_root):
        for filename in filenames:
            if filename.endswith(".md") or filename.endswith(".py"):
                paths.add(Path(directory) / filename)
    try:
        overlay_manifest = Path(
            json.loads(configured.overlay_pointer.read_text(encoding="utf-8"))[
                "overlay_manifest"
            ]
        )
        paths.add(overlay_manifest)
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    paths.update(
        path
        for path in (
            configured.overlay_pointer,
            configured.source_findings_database,
            configured.paper_database,
            configured.books_database,
            configured.operations_database,
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[1] / "src/quant_lab/research/knowledge_completion.py",
            Path(__file__).resolve().parents[1] / "src/quant_lab/research/data_admission.py",
            Path(__file__).resolve().parents[1] / "src/quant_lab/data/lseg_pit_adapter.py",
            Path(__file__).resolve().parents[1] / "src/quant_lab/research/lseg_rerun.py",
        )
        if path is not None and path.is_file()
    )
    inventory = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        details = path.stat()
        inventory.append(
            {
                "path": str(path),
                "size": details.st_size,
                "mtime_ns": details.st_mtime_ns,
                "ctime_ns": details.st_ctime_ns,
                "inode": details.st_ino,
                # Same-size rewrites may share timestamps on coarse filesystems.
                "sha256": sha256_file(path),
            }
        )
    return semantic_hash(inventory)


def _package_stat_fingerprint(package: Path) -> str:
    inventory = []
    for path in sorted(package.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file():
            continue
        details = path.stat()
        inventory.append(
            {
                "path": path.relative_to(package).as_posix(),
                "size": details.st_size,
                "mtime_ns": details.st_mtime_ns,
                "ctime_ns": details.st_ctime_ns,
                "inode": details.st_ino,
                # Same-size rewrites may share timestamps on coarse filesystems.
                "sha256": sha256_file(path),
            }
        )
    return semantic_hash(inventory)


def _trust_policy_fingerprint(configured: CompletionInputs) -> str:
    policies = {
        "approval": sorted((configured.approval_trust_stores or {}).keys()),
        "validation_runner": sorted(
            (configured.validation_runner_trust_stores or {}).keys()
        ),
        "data_admission": sorted(
            (configured.data_admission_trust_stores or {}).keys()
        ),
        "data_release": sorted((configured.data_release_trust_stores or {}).keys()),
    }
    return semantic_hash(policies)


def _cache_payload(configured: CompletionInputs, package: Path) -> dict[str, object]:
    manifest_path = package / "completion-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload: dict[str, object] = {
        "schema_version": 1,
        "completion_id": manifest["completion_id"],
        "completion_manifest_sha256": sha256_file(manifest_path),
        "source_stat_fingerprint": _source_stat_fingerprint(configured),
        "package_stat_fingerprint": _package_stat_fingerprint(package),
        "trust_policy_fingerprint": _trust_policy_fingerprint(configured),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["receipt_semantic_sha256"] = semantic_hash(payload)
    return payload


def _write_search_cache(configured: CompletionInputs, package: Path) -> None:
    target = configured.output_root / SEARCH_CACHE_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _cache_payload(configured, package)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _search_cache_is_current(configured: CompletionInputs, package: Path) -> bool:
    path = configured.output_root / SEARCH_CACHE_NAME
    if path.is_symlink() or not path.is_file():
        return False
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        claimed = cached.pop("receipt_semantic_sha256")
        expected = _cache_payload(configured, package)
        expected.pop("verified_at")
        expected.pop("receipt_semantic_sha256")
        cached_without_time = dict(cached)
        cached_without_time.pop("verified_at")
        return semantic_hash(cached) == claimed and cached_without_time == expected
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def inputs(args: argparse.Namespace) -> CompletionInputs:
    knowledge = args.knowledge_root
    approval_stores = [ApprovalTrustStore(path) for path in args.approval_trust_store]
    runner_stores = [
        ValidationRunnerTrustStore(path) for path in args.validation_runner_trust_store
    ]
    admission_stores = [
        DataAdmissionTrustStore(path) for path in args.data_admission_trust_store
    ]
    release_stores = [DataReleaseTrustStore(path) for path in args.data_release_trust_store]
    return CompletionInputs(
        workspace_root=args.workspace_root,
        output_root=knowledge,
        overlay_pointer=knowledge / "findings-hygiene-current.json",
        source_findings_database=knowledge / "findings.sqlite3",
        paper_database=args.paper_database,
        books_database=knowledge / "books.sqlite3",
        operations_database=knowledge / "operations.sqlite3",
        embedding_model=args.embedding_model,
        approval_trust_stores={store.source_hash: store for store in approval_stores},
        validation_runner_trust_stores={store.source_hash: store for store in runner_stores},
        data_admission_trust_stores={store.source_hash: store for store in admission_stores},
        data_release_trust_stores={store.source_hash: store for store in release_stores},
    )


def current_package(knowledge_root: Path) -> Path:
    pointer_path = knowledge_root / "knowledge-completion-current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version", "completion_id", "completion_dir", "completion_manifest",
        "completion_manifest_sha256", "package_read_only_verified",
        "package_local_user_immutable_verified", "updated_at", "pointer_semantic_sha256",
    }
    if set(pointer) != expected_fields:
        raise CompletionError("completion pointer fields are not canonical")
    claimed = pointer["pointer_semantic_sha256"]
    unsigned = dict(pointer)
    unsigned.pop("pointer_semantic_sha256")
    if semantic_hash(unsigned) != claimed:
        raise CompletionError("completion pointer semantic hash mismatch")
    completion_id = pointer["completion_id"]
    package = Path(pointer["completion_dir"]).resolve()
    expected_package = (knowledge_root.resolve() / "completion" / completion_id).resolve()
    if package != expected_package or package.name != completion_id or package.is_symlink():
        raise CompletionError("completion pointer escaped its content-addressed package root")
    manifest_path = package / "completion-manifest.json"
    if Path(pointer["completion_manifest"]).resolve() != manifest_path.resolve():
        raise CompletionError("completion pointer manifest path mismatch")
    if not manifest_path.is_file() or sha256_file(manifest_path) != pointer["completion_manifest_sha256"]:
        raise CompletionError("completion pointer manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("completion_id") != completion_id:
        raise CompletionError("completion pointer and manifest identity disagree")
    read_only = not any(
        stat.S_IMODE(path.stat().st_mode) & 0o222 for path in [package, *package.rglob("*")]
    )
    if pointer["package_read_only_verified"] is not read_only:
        raise CompletionError("completion pointer package-custody claim is stale")
    immutable_flag = getattr(stat, "UF_IMMUTABLE", 0)
    immutable = bool(immutable_flag and package.stat().st_flags & immutable_flag)
    if pointer["package_local_user_immutable_verified"] is not immutable:
        raise CompletionError("completion pointer immutable-custody claim is stale")
    return package


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--workspace-root", type=Path, default=DEFAULT_WORKSPACE)
    result.add_argument("--knowledge-root", type=Path, default=DEFAULT_KNOWLEDGE)
    result.add_argument("--paper-database", type=Path, default=DEFAULT_PAPERS)
    result.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    result.add_argument(
        "--approval-trust-store", type=Path, action="append", default=[],
        help="historical root-anchored approval trust registry (repeatable)",
    )
    result.add_argument(
        "--validation-runner-trust-store", type=Path, action="append", default=[],
        help="historical root-anchored validation-runner registry (repeatable)",
    )
    result.add_argument(
        "--data-admission-trust-store", type=Path, action="append", default=[],
        help="historical root-anchored data-admission registry (repeatable)",
    )
    result.add_argument(
        "--data-release-trust-store", type=Path, action="append", default=[],
        help="historical root-anchored human data-release registry (repeatable)",
    )
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--embedding-cache", type=Path, default=DEFAULT_KNOWLEDGE.parent / "model-cache")
    build.add_argument("--no-protect", action="store_true")
    commands.add_parser("status")
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)
    return result


def main() -> int:
    args = parser().parse_args()
    configured = inputs(args)
    if args.command == "build":
        built = build_completion_package(
            configured,
            embedding_cache=args.embedding_cache,
            protect=not args.no_protect,
        )
        print(f"Completion: {built.completion_id}\nPackage: {built.completion_dir}")
        for name, passed in built.verification.items():
            print(f"{'PASS' if passed else 'FAIL'} {name}")
        if all(built.verification.values()):
            _write_search_cache(configured, built.completion_dir)
        return 0 if all(built.verification.values()) else 1
    package = current_package(args.knowledge_root)
    report: dict[str, bool] | None = None
    if args.command != "search" or not _search_cache_is_current(configured, package):
        report = verify_completion_package(configured, package)
        if not all(report.values()):
            failed = ", ".join(name for name, passed in report.items() if not passed)
            raise CompletionError(f"current completion package is stale or invalid: {failed}")
        _write_search_cache(configured, package)
    if args.command == "status":
        assert report is not None
        for name, passed in report.items():
            print(f"{'PASS' if passed else 'FAIL'} {name}")
        print(f"Overall: {'PASS' if all(report.values()) else 'FAIL'}")
        return 0 if all(report.values()) else 1
    manifest = json.loads((package / "completion-manifest.json").read_text(encoding="utf-8"))
    database = package / manifest["step_10_artifact_index"]["database"]
    rows = search_canonical_artifacts(database, args.query, limit=args.limit)
    for index, row in enumerate(rows, start=1):
        snippet = " ".join(row["content"].split())[:300]
        print(
            f"[{index}] {row['title']} — {row['heading']}\n"
            f"    {row['rel_path']} [{row['category']}]\n    {snippet}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
