from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_wave5_hmm_conversion.py"
WORKSPACE = Path(__file__).resolve().parents[2]


def test_dry_run_does_not_require_or_print_a_credential() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--workspace-root", str(WORKSPACE), "--stage", "all-api", "--dry-run"],
        check=False, capture_output=True, text=True, env={},
    )
    assert completed.returncode == 0
    assert "glm-ocr" in completed.stdout
    assert "glm-5v-turbo" in completed.stdout
    assert "Credential is not read" in completed.stdout
    assert "ZAI_API_KEY=" not in completed.stdout
