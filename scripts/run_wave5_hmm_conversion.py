#!/usr/bin/env python3
"""Run Wave 5 Z.ai OCR stages without placing credentials in commands or artifacts."""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path


RUN_ID = "ZAI-OCR-WAVE5-HMM-REGIME-20260715"
BATCH_ID = "ACQ-WAVE5-HMM-REGIME-20260715"


def load_key(source: str, service: str, account: str) -> str:
    if source == "environment":
        key = os.environ.get("ZAI_API_KEY") or os.environ.get("Z_AI_API_KEY") or ""
        if not key:
            raise RuntimeError("ZAI_API_KEY is not available in this process environment")
        return key
    completed = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-s", service, "-a", account, "-w"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    key = completed.stdout.strip()
    if completed.returncode or not key:
        raise RuntimeError(
            f"No readable macOS Keychain item for service {service!r} and account {account!r}"
        )
    return key


def commands(workspace: Path, stage: str) -> list[tuple[str, list[str]]]:
    python = str(workspace / "quant_lab/.venv/bin/python")
    scripts = workspace / "quant_lab/scripts"
    run_dir = workspace / "quant_lab/artifacts/zai_ocr" / RUN_ID
    manifest = workspace / "PhD_Math_Agent_Library/acquisition_staging" / BATCH_ID / "conversion_queue.csv"
    production = [
        python, str(scripts / "run_zai_ocr_pipeline.py"),
        "--input-manifest", str(manifest),
        "--output-root", str(workspace / "quant_lab/artifacts/zai_ocr"),
        "--run-id", RUN_ID,
        "--quality-profile", "math-precision",
        "--model", "glm-ocr",
        "--transport-mode", "chunked",
        "--splitter", "hybrid",
        "--return-crop-images",
        "--need-layout-visualization",
    ]
    retries = [
        python, str(scripts / "run_zai_formula_retries.py"),
        "--run-dir", str(run_dir),
        "--dpi", "300",
        "--workers", "1",
        "--model", "glm-ocr",
    ]
    vision = [
        python, str(scripts / "run_zai_formula_vision_repairs.py"),
        "--run-dir", str(run_dir),
        "--dpi", "300",
        "--workers", "1",
        "--model", "glm-5v-turbo",
    ]
    available = {
        "production": [("production GLM-OCR", production)],
        "formula-retries": [("300-DPI GLM-OCR formula retries", retries)],
        "vision-repairs": [("GLM-5V-Turbo formula repairs", vision)],
        "all-api": [
            ("production GLM-OCR", production),
            ("300-DPI GLM-OCR formula retries", retries),
            ("GLM-5V-Turbo formula repairs", vision),
        ],
    }
    return available[stage]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--stage", choices=["production", "formula-retries", "vision-repairs", "all-api"],
        default="all-api",
    )
    parser.add_argument("--credential-source", choices=["keychain", "environment"], default="keychain")
    parser.add_argument("--keychain-service", default="quant-lab-zai-api")
    parser.add_argument("--keychain-account", default=getpass.getuser())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    workspace = Path(args.workspace_root).expanduser().resolve()
    planned = commands(workspace, args.stage)
    if args.dry_run:
        for name, command in planned:
            print(f"{name}: {' '.join(command)}")
        print("Credential is not read during --dry-run")
        return 0

    try:
        key = load_key(args.credential_source, args.keychain_service, args.keychain_account)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    env = os.environ.copy()
    env["ZAI_API_KEY"] = key
    env.pop("Z_AI_API_KEY", None)
    for name, command in planned:
        print(f"Starting {name}", flush=True)
        completed = subprocess.run(command, cwd=workspace, env=env, check=False)
        if completed.returncode:
            raise SystemExit(f"{name} failed with exit code {completed.returncode}")
    print("Wave 5 API stages completed; deterministic and visual acceptance audits are still required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
