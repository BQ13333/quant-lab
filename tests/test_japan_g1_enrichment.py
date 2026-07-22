from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g1_enrichment import (
    ACTIVE_UNIVERSE_FILE,
    JAPAN_G1_ENRICHMENT_FIELDS,
    REFERENCE_UNIVERSE_FILE,
    _acquire_run_lock,
    write_japan_g1_enrichment_artifacts,
)
NOW = datetime(2026, 7, 11, 16, 0, tzinfo=timezone.utc)


class FakeGetData:
    def __init__(self, empty: bool = False):
        self.empty = empty
        self.calls: list[dict[str, object]] = []

    def get_data(self, universe, fields):
        rics = list(universe)
        self.calls.append({"universe": rics, "fields": list(fields)})
        if self.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "Instrument": ric,
                    "CF_CURR": "JPY",
                    "First Trade Date": "2000-01-01",
                    "Retire Date": pd.NA,
                    "ISIN": f"ISIN-{ric}",
                    "SEDOL": f"SEDOL-{ric}",
                    "PermID": f"PERM-{ric}",
                }
                for ric in rics
            ]
        )


class OneBadRicGetData(FakeGetData):
    def get_data(self, universe, fields):
        rics = list(universe)
        self.calls.append({"universe": rics, "fields": list(fields)})
        if "6758.T" in rics:
            raise RuntimeError("400 Bad Request")
        return pd.DataFrame(
            [{"Instrument": ric, "CF_CURR": "JPY", "First Trade Date": "2000-01-01"} for ric in rics]
        )


def _sources(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    pd.DataFrame({"refinitiv_ric": ["7203.T", "6758.T"]}).to_csv(source / ACTIVE_UNIVERSE_FILE, index=False)
    pd.DataFrame({"ric": ["7203.T", "6758.T", "OLD.T"]}).to_csv(source / REFERENCE_UNIVERSE_FILE, index=False)
    return source


def test_bounded_resume_uses_discovery_fields_and_never_promotes(tmp_path):
    source = _sources(tmp_path)
    client = FakeGetData()
    first = write_japan_g1_enrichment_artifacts(
        client,
        source_dir=source,
        artifact_root=tmp_path / "out",
        run_id="G1",
        max_rics=2,
        request_chunk_size=1,
        sleep_seconds=0,
        session_name="desktop.workspace",
        clock=lambda: NOW,
    )
    assert [call["universe"] for call in client.calls] == [["7203.T"], ["6758.T"]]
    assert all(call["fields"] == list(JAPAN_G1_ENRICHMENT_FIELDS) for call in client.calls)
    assert first.manifest["status"] == "bounded_in_progress"
    assert first.manifest["does_not_assert_historical_pit"] is True
    assert first.manifest["promotion_policy"] == {
        "alpha": False, "production": False, "ibkr": False, "live_trading": False
    }

    second = write_japan_g1_enrichment_artifacts(
        client,
        source_dir=source,
        artifact_root=tmp_path / "out",
        run_id="G1",
        max_rics=2,
        request_chunk_size=2,
        sleep_seconds=0,
        clock=lambda: NOW,
    )
    assert client.calls[-1]["universe"] == ["OLD.T"]
    assert second.manifest["status"] == "complete_current_observation"
    normalized = pd.read_csv(second.normalized_responses_path)
    assert normalized["observed_at_utc"].notna().all()
    assert not normalized["historical_pit_claimed"].astype(bool).any()
    for line in second.checksums_path.read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((second.artifact_dir / name).read_bytes()).hexdigest() == expected


def test_empty_response_fails_closed_and_persists_per_ric_error(tmp_path):
    source = _sources(tmp_path)
    result = write_japan_g1_enrichment_artifacts(
        FakeGetData(empty=True),
        source_dir=source,
        artifact_root=tmp_path / "out",
        run_id="EMPTY",
        max_rics=2,
        request_chunk_size=2,
        clock=lambda: NOW,
    )
    assert result.manifest["status"] == "bounded_in_progress_with_retryable_errors"
    assert result.manifest["fail_closed_on_empty_response"] is True
    assert result.status["status"].tolist()[:2] == ["failed_empty", "failed_empty"]
    assert result.status["error_message"].astype(str).str.contains("no normalized").iloc[:2].all()
    event = json.loads(result.raw_responses_path.read_text().splitlines()[0])
    assert event["raw_row_count"] == 0
    assert event["historical_pit_claimed"] is False


def test_single_writer_lock_and_source_checksum_drift_are_rejected(tmp_path):
    source = _sources(tmp_path)
    artifact_dir = tmp_path / "out" / "LOCKED"
    artifact_dir.mkdir(parents=True)
    lock = _acquire_run_lock(artifact_dir)
    try:
        with pytest.raises(RuntimeError, match="owns the lock"):
            write_japan_g1_enrichment_artifacts(
                FakeGetData(), source_dir=source, artifact_root=tmp_path / "out", run_id="LOCKED", max_rics=1
            )
    finally:
        lock.close()

    write_japan_g1_enrichment_artifacts(
        FakeGetData(), source_dir=source, artifact_root=tmp_path / "out", run_id="DRIFT", max_rics=1, clock=lambda: NOW
    )
    frame = pd.read_csv(source / ACTIVE_UNIVERSE_FILE)
    frame.loc[len(frame)] = ["9984.T"]
    frame.to_csv(source / ACTIVE_UNIVERSE_FILE, index=False)
    with pytest.raises(RuntimeError, match="checksum changed"):
        write_japan_g1_enrichment_artifacts(
            FakeGetData(), source_dir=source, artifact_root=tmp_path / "out", run_id="DRIFT", max_rics=1
        )


def test_runner_defaults_to_desktop_workspace_and_supports_max_rics():
    path = Path(__file__).resolve().parents[1] / "examples" / "run_japan_g1_enrichment.py"
    spec = importlib.util.spec_from_file_location("run_japan_g1_enrichment", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["--max-rics", "7"])
    assert args.max_rics == 7
    assert args.session == "desktop.workspace"


def test_vendor_400_is_bisected_to_single_bad_ric_without_stopping_progress(tmp_path):
    source = _sources(tmp_path)
    client = OneBadRicGetData()

    result = write_japan_g1_enrichment_artifacts(
        client,
        source_dir=source,
        artifact_root=tmp_path / "out",
        run_id="BISECT",
        max_rics=3,
        request_chunk_size=3,
        clock=lambda: NOW,
    )

    status = result.status.set_index("ric")["status"].to_dict()
    assert status == {"7203.T": "success", "6758.T": "error", "OLD.T": "success"}
    assert result.manifest["status"] == "bounded_in_progress_with_retryable_errors"
    assert any(call["universe"] == ["6758.T"] for call in client.calls)
