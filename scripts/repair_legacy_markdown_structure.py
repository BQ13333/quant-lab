#!/usr/bin/env python3
"""Apply source-verified structural repairs to three legacy Markdown books."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_zai_formula_vision_repairs as vision  # noqa: E402


REPAIRS = {
    "0b4f05a74037-minimizing-dynamic-regret-and-adaptive-regret-simultaneously": [
        (
            "$$ \\mathcal{A}_{t}=\\mathcal{A}_{t} \\setminus \\{E_{I}|t \\in I,(t+1) \\notin I\\} $",
            "$$ \\mathcal{A}_{t}=\\mathcal{A}_{t} \\setminus \\{E_{I}|t \\in I,(t+1) \\notin I\\} $$",
            "Page 7 algorithm display had one missing terminal dollar.",
        ),
        (
            "$$ C_{t,I}=C_{t-1,I}+|f_{t}(\\mathbf{w}_{t})-f_{t}(\\mathbf{w}_{t,I})| $",
            "$$ C_{t,I}=C_{t-1,I}+|f_{t}(\\mathbf{w}_{t})-f_{t}(\\mathbf{w}_{t,I})| $$",
            "Page 7 algorithm display had one missing terminal dollar.",
        ),
        (
            "$$= O\\left(\\sqrt{|I|(\\log s + P_{I})}\\right)$",
            "$$= O\\left(\\sqrt{|I|(\\log s + P_{I})}\\right)$$",
            "Page 7 theorem display had one missing terminal dollar.",
        ),
    ],
    "10084b27e2e3-monk-outlier-robust-mean-embedding-estimation-by-median-of-means": [
        (
            """$$\\max_{a, b \\in \\mathbb{R}^{2N}} c^* Kc \\leq 1 \\operatorname*{med}_{q \\in [Q]} \\left\\{ \\frac{1}{|S_q|} \\sum_{j, n \\in S_q} [a_n K(x_j, x_n)] + \\right.$$

$$b_n K(x_j, y_n)] - \\frac{\\sum_{j, n \\in S_q} [a_n K(y_j, x_n) + b_n K(y_j, y_n)]}{|S_q|} \\right\\}.$$""",
            """$$
\\begin{aligned}
&\\max_{c=[a,b] \\in \\mathbb{R}^{2N}:\\,c^*Kc\\leq1}
\\operatorname*{med}_{q\\in[Q]}\\left\\{
\\frac{1}{|S_q|}\\sum_{j,n\\in S_q}
[a_nK(x_j,x_n)+b_nK(x_j,y_n)] \\\\
&\\qquad-\\frac{\\sum_{j,n\\in S_q}
[a_nK(y_j,x_n)+b_nK(y_j,y_n)]}{|S_q|}\\right\\}.
\\end{aligned}
$$""",
            "Page 6 split display had an unpaired scalable delimiter and moved the quadratic constraint out of the maximization subscript; both were restored from the source render.",
        ),
    ],
    "a627adb7f73f-a-reality-check-for-data-snooping": [
        (
            "$$: $ S \\equiv \\left[ \\begin{array}{cc} S _ {f f} & S _ {f h} B ^ {\\prime} \\\\ B S _ {h f} & B S _ {h h} B ^ {\\prime} \\end{array} \\right] . $",
            """$$
S \\equiv \\left[ \\begin{array}{cc} S _ {f f} & S _ {f h} B ^ {\\prime} \\\\ B S _ {h f} & B S _ {h h} B ^ {\\prime} \\end{array} \\right].
$$""",
            "Page 21 covariance-block display was verified against the journal page and its damaged delimiters were restored.",
        ),
        (
            "$$ \\leq n ^ { 1 / 2 } \\left( E \\left( f _ { k } ^ { * } \\right) - E \\left( f _ { 1 } ^ { * } \\right) \\right) / S _ { k } ^ {\\prime} \\Omega S _ { k } ] .",
            """$$
\\begin{aligned}
P[A_k^c]
&= P[\\bar{f}_1-\\bar{f}_k \\leq 0] \\\\
&= P\\left[
\\frac{n^{1/2}\\left(\\bar{f}_1-E(f_1^*)-[\\bar{f}_k-E(f_k^*)]\\right)}{S_k^\\prime \\Omega S_k}
\\leq
\\frac{n^{1/2}\\left(E(f_k^*)-E(f_1^*)\\right)}{S_k^\\prime \\Omega S_k}
\\right].
\\end{aligned}
$$""",
            "Page 22 first proposition display was truncated by layout OCR; reconstructed from the rendered Econometrica source page.",
        ),
        (
            """so that uniformly in z

$$

$$
\\left| Q _ {\\omega}""",
            """so that uniformly in z

$$
\\left| Q _ {\\omega}""",
            "Page 24 contained a spurious empty display between the lead-in and its inequality.",
        ),
    ],
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    root = Path(args.markdown_dir).expanduser().resolve()
    records: list[dict[str, object]] = []
    for document_id, replacements in REPAIRS.items():
        path = root / f"{document_id}.md"
        text = path.read_text(encoding="utf-8")
        before_hash = sha256_text(text)
        notes: list[str] = []
        for old, new, note in replacements:
            count = text.count(old)
            if count != 1:
                raise RuntimeError(f"Expected one exact repair span in {path.name}; found {count}: {note}")
            text = text.replace(old, new, 1)
            notes.append(note)
        issues = vision.markdown_structure_issues(text)
        if issues:
            raise RuntimeError(f"Structural issues remain in {path.name}: {issues}")
        path.write_text(text, encoding="utf-8")
        records.append(
            {
                "document_id": document_id,
                "markdown_path": str(path),
                "sha256_before": before_hash,
                "sha256_after": sha256_text(text),
                "repair_count": len(replacements),
                "notes": notes,
                "validation": "pass",
            }
        )
    manifest = Path(args.manifest).expanduser().resolve()
    manifest.write_text(json.dumps({"status": "pass", "repairs": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Repaired {len(records)} legacy Markdown documents; manifest: {manifest}")


if __name__ == "__main__":
    main()
