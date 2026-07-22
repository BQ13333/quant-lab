from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g2_price_union import build_japan_g2_price_union


def test_builds_deterministic_checksummed_union_without_promotion(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    result = build_japan_g2_price_union(
        **inputs,
        artifact_root=tmp_path / "artifacts",
        run_id="JPN-G2-PRICE-UNION",
        batch_size=1,
        copy_chunk_size=17,
    )

    assert result.manifest["status"] == "complete"
    assert result.manifest["output_rows"] == 4
    assert result.manifest["role_rows"] == {"active": 2, "reference": 2}
    assert result.manifest["exact_rows_unique"] is True
    assert result.manifest["keys_unique"] is True
    assert result.manifest["source_file_hashes_verified"] is True
    assert result.manifest["promotion"] is False
    assert result.manifest["promotion_allowed"] is False
    assert not any(result.manifest["promotion_policy"].values())
    assert sorted(path.relative_to(result.dataset_root).as_posix() for path in result.dataset_root.rglob("*.parquet")) == [
        "active/year_partition=2020/part-000.parquet",
        "reference/year_partition=2020/part-000.parquet",
    ]
    union = pd.read_parquet(result.dataset_root).sort_values(["ticker", "date"]).reset_index(drop=True)
    assert union["ticker"].tolist() == ["AAA.T", "AAA.T", "BBB.T", "BBB.T"]
    for line in result.checksums_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert _sha256(result.artifact_dir / relative) == digest

    with pytest.raises(FileExistsError, match="immutable"):
        build_japan_g2_price_union(
            **inputs,
            artifact_root=tmp_path / "artifacts",
            run_id="JPN-G2-PRICE-UNION",
        )


@pytest.mark.parametrize(
    ("manifest_name", "change", "message"),
    [
        ("active_compaction_manifest", {"status": "partial"}, "compaction manifest is incomplete"),
        (
            "reference_row_equality_manifest",
            {"row_equality_passed": False},
            "row-equality manifest is incomplete",
        ),
    ],
)
def test_rejects_incomplete_or_non_equal_inputs(
    tmp_path: Path, manifest_name: str, change: dict[str, object], message: str
) -> None:
    inputs = _inputs(tmp_path)
    path = inputs[manifest_name]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(change)
    _write_json(path, manifest)

    with pytest.raises(ValueError, match=message):
        build_japan_g2_price_union(
            **inputs,
            artifact_root=tmp_path / "artifacts",
            run_id="REJECTED",
        )
    assert not (tmp_path / "artifacts" / "REJECTED").exists()


def test_rejects_duplicate_keys_and_leaves_no_partial_artifact(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, reference_ticker="AAA.T")

    with pytest.raises(ValueError, match="duplicate key rows"):
        build_japan_g2_price_union(
            **inputs,
            artifact_root=tmp_path / "artifacts",
            run_id="DUPLICATE-KEYS",
            batch_size=1,
        )

    assert not (tmp_path / "artifacts" / "DUPLICATE-KEYS").exists()
    assert not list((tmp_path / "artifacts").glob(".DUPLICATE-KEYS.*"))


def test_rejects_manifest_row_count_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    path = inputs["active_compaction_manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["source_rows"] = manifest["output_rows"] = 3
    _write_json(path, manifest)
    equality_path = inputs["active_row_equality_manifest"]
    equality = json.loads(equality_path.read_text(encoding="utf-8"))
    equality["source_rows"] = equality["compacted_rows"] = 3
    _write_json(equality_path, equality)

    with pytest.raises(ValueError, match="do not reconcile"):
        build_japan_g2_price_union(
            **inputs,
            artifact_root=tmp_path / "artifacts",
            run_id="ROW-DRIFT",
        )


def _inputs(tmp_path: Path, reference_ticker: str = "BBB.T") -> dict[str, Path]:
    active = _lane(tmp_path, "active", "AAA.T", 100.0)
    reference = _lane(tmp_path, "reference", reference_ticker, 200.0)
    return {
        "active_compacted_root": active[0],
        "active_compaction_manifest": active[1],
        "active_row_equality_manifest": active[2],
        "reference_compacted_root": reference[0],
        "reference_compaction_manifest": reference[1],
        "reference_row_equality_manifest": reference[2],
    }


def _lane(root: Path, role: str, ticker: str, close: float) -> tuple[Path, Path, Path]:
    compacted = root / role / "compacted"
    partition = compacted / "year_partition=2020"
    partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "ticker": [ticker, ticker],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "close": [close, close + 1.0],
            "volume": [1_000.0, 1_100.0],
        }
    ).to_parquet(partition / "part-000.parquet", index=False)
    compaction_path = root / role / "compaction.json"
    equality_path = root / role / "row_equality.json"
    _write_json(
        compaction_path,
        {
            "status": "complete",
            "source_rows": 2,
            "output_rows": 2,
            "failed_groups": 0,
            "output_root": str(compacted),
        },
    )
    _write_json(
        equality_path,
        {
            "status": "pass",
            "row_equality_passed": True,
            "source_rows": 2,
            "compacted_rows": 2,
            "failed_years": [],
            "compacted_root": str(compacted),
        },
    )
    return compacted, compaction_path, equality_path


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
