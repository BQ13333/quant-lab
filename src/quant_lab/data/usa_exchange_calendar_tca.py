from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd


NEW_YORK = ZoneInfo("America/New_York")
UTC = timezone.utc
EXCHANGES = ("NYSE", "NASDAQ")
HALT_EXCHANGES = (
    "NASDAQ",
    "NYSE",
    "NYSE_AMERICAN",
    "NYSE_NATIONAL",
    "NYSE_ARCA",
    "IEX",
    "CBOE_BZX",
)
OFFICIAL_HALT_SOURCE_QUALITY = "official_exchange_or_regulator"
VALIDATED_TCA_SOURCE_QUALITIES = {
    "broker_fee_schedule_and_empirical_calibration",
    "empirically_calibrated_execution_sample",
}
RESEARCH_TCA_SOURCE_QUALITY = "market_data_calibrated_research"
MIN_RESEARCH_CALIBRATION_ROWS = 1_000_000
MIN_RESEARCH_CALIBRATION_ENTITIES = 500
REQUIRED_CALIBRATION_ARTIFACTS = {
    "tca_assumptions_json",
    "tca_assumptions_csv",
    "calibration_evidence",
    "stress_grid",
}
SPECIAL_CLOSURE_NAMES = {
    "national_day_of_mourning_nixon",
    "september_11_market_closure",
    "national_day_of_mourning_reagan",
    "national_day_of_mourning_ford",
    "hurricane_sandy_market_closure",
    "national_day_of_mourning_bush",
    "national_day_of_mourning_carter",
}

CALENDAR_COLUMNS = (
    "exchange",
    "calendar_code",
    "date",
    "day_name",
    "is_session",
    "session_open_local",
    "session_close_local",
    "session_open_utc",
    "session_close_utc",
    "scheduled_minutes",
    "closure_type",
    "event_name",
    "is_early_close",
    "source_provider",
    "source_version",
    "source_quality",
)

HALT_COLUMNS = (
    "event_id",
    "exchange",
    "symbol",
    "halt_start_utc",
    "halt_end_utc",
    "event_type",
    "reason_code",
    "source_name",
    "source_record_id",
    "observed_at_utc",
)


@dataclass(frozen=True)
class TCAAssumptions:
    commission_bps: float = 0.0
    exchange_fee_bps: float = 0.30
    regulatory_sell_fee_bps: float = 0.0
    half_spread_bps: float = 2.50
    impact_bps_at_reference_participation: float = 5.0
    reference_participation_rate: float = 0.10
    max_participation_rate: float = 0.10
    source_quality: str = "model_assumption_unvalidated"
    source_name: str = "user_configured_research_assumptions"

    def validate(self) -> None:
        numeric_bps = {
            "commission_bps": self.commission_bps,
            "exchange_fee_bps": self.exchange_fee_bps,
            "regulatory_sell_fee_bps": self.regulatory_sell_fee_bps,
            "half_spread_bps": self.half_spread_bps,
            "impact_bps_at_reference_participation": self.impact_bps_at_reference_participation,
        }
        for field, value in numeric_bps.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be a finite non-negative number")
        if not 0 < self.reference_participation_rate <= 1:
            raise ValueError("reference_participation_rate must be in (0, 1]")
        if not 0 < self.max_participation_rate <= 1:
            raise ValueError("max_participation_rate must be in (0, 1]")
        if self.max_participation_rate < self.reference_participation_rate:
            raise ValueError("max_participation_rate cannot be below reference_participation_rate")
        if not self.source_name.strip():
            raise ValueError("source_name cannot be empty")


@dataclass(frozen=True)
class CalendarSchedule:
    exchange: str
    calendar_code: str
    source_provider: str
    source_version: str
    source_quality: str
    authoritative: bool
    sessions: pd.DataFrame


class CalendarProvider(Protocol):
    def schedule(self, exchange: str, start_date: date, end_date: date) -> CalendarSchedule: ...


@dataclass(frozen=True)
class USAExchangeCalendarTCAArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    calendar_path: Path
    tca_assumptions_path: Path
    halt_suspension_path: Path
    evidence_quality_path: Path
    summary_path: Path
    calendar: pd.DataFrame
    tca_assumptions: pd.DataFrame
    halt_suspensions: pd.DataFrame
    evidence_quality: pd.DataFrame


class SpecialistCalendarProvider:
    """Adapter for locally installed specialist exchange-calendar packages."""

    def __init__(self, backend: str = "auto") -> None:
        supported = {"auto", "exchange_calendars", "pandas_market_calendars"}
        if backend not in supported:
            raise ValueError(f"calendar backend must be one of {sorted(supported)}")
        self.backend = self._resolve_backend(backend)

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        candidates = (
            (backend,) if backend != "auto" else ("exchange_calendars", "pandas_market_calendars")
        )
        for candidate in candidates:
            try:
                importlib.import_module(candidate)
                return candidate
            except ImportError:
                continue
        requested = backend if backend != "auto" else "exchange_calendars or pandas_market_calendars"
        raise RuntimeError(f"no local specialist calendar provider available ({requested})")

    def schedule(self, exchange: str, start_date: date, end_date: date) -> CalendarSchedule:
        if exchange not in EXCHANGES:
            raise ValueError(f"unsupported exchange: {exchange}")
        if self.backend == "exchange_calendars":
            sessions, code, authoritative = self._exchange_calendars_schedule(exchange, start_date, end_date)
        else:
            sessions, code = self._pandas_market_calendars_schedule(exchange, start_date, end_date)
            authoritative = True
        return CalendarSchedule(
            exchange=exchange,
            calendar_code=code,
            source_provider=self.backend,
            source_version=_package_version(self.backend),
            source_quality=(
                "authoritative_calendar_library"
                if authoritative
                else "specialist_library_shared_us_equity_proxy"
            ),
            authoritative=authoritative,
            sessions=sessions,
        )

    @staticmethod
    def _exchange_calendars_schedule(
        exchange: str, start_date: date, end_date: date
    ) -> tuple[pd.DataFrame, str, bool]:
        module = importlib.import_module("exchange_calendars")
        requested_code = "XNYS" if exchange == "NYSE" else "XNAS"
        available = set(module.get_calendar_names())
        code = requested_code if requested_code in available else "XNYS"
        if code not in available:
            raise RuntimeError("exchange_calendars does not provide XNYS")
        calendar = module.get_calendar(code)
        labels = calendar.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
        rows = []
        for label in labels:
            rows.append(
                {
                    "date": pd.Timestamp(label).date(),
                    "open_utc": _as_utc_timestamp(calendar.session_open(label)),
                    "close_utc": _as_utc_timestamp(calendar.session_close(label)),
                }
            )
        suffix = "" if code == requested_code else "_shared_for_nasdaq"
        return (
            pd.DataFrame(rows, columns=["date", "open_utc", "close_utc"]),
            f"{code}{suffix}",
            code == requested_code,
        )

    @staticmethod
    def _pandas_market_calendars_schedule(
        exchange: str, start_date: date, end_date: date
    ) -> tuple[pd.DataFrame, str]:
        module = importlib.import_module("pandas_market_calendars")
        code = "NYSE" if exchange == "NYSE" else "NASDAQ"
        calendar = module.get_calendar(code)
        raw = calendar.schedule(start_date=start_date.isoformat(), end_date=end_date.isoformat())
        rows = [
            {
                "date": pd.Timestamp(label).date(),
                "open_utc": _as_utc_timestamp(row["market_open"]),
                "close_utc": _as_utc_timestamp(row["market_close"]),
            }
            for label, row in raw.iterrows()
        ]
        return pd.DataFrame(rows, columns=["date", "open_utc", "close_utc"]), code


class BuiltinUSMarketRulesProvider:
    """Offline modern-market fallback; complete rows, but not authoritative evidence."""

    minimum_date = date(1990, 1, 1)

    def schedule(self, exchange: str, start_date: date, end_date: date) -> CalendarSchedule:
        if exchange not in EXCHANGES:
            raise ValueError(f"unsupported exchange: {exchange}")
        if start_date < self.minimum_date:
            raise ValueError("built-in US market rules support dates from 1990-01-01 onward")
        closures = _closure_events(start_date, end_date)
        early_closes = _early_close_events(start_date, end_date)
        rows: list[dict[str, Any]] = []
        for day in _date_range(start_date, end_date):
            if day.weekday() >= 5 or day in closures:
                continue
            close_at = time(13, 0) if day in early_closes else time(16, 0)
            open_local = datetime.combine(day, time(9, 30), tzinfo=NEW_YORK)
            close_local = datetime.combine(day, close_at, tzinfo=NEW_YORK)
            rows.append(
                {
                    "date": day,
                    "open_utc": pd.Timestamp(open_local.astimezone(UTC)),
                    "close_utc": pd.Timestamp(close_local.astimezone(UTC)),
                }
            )
        return CalendarSchedule(
            exchange=exchange,
            calendar_code=f"{exchange}_BUILTIN_MODERN_RULES",
            source_provider="quant_lab_builtin_us_market_rules",
            source_version="1",
            source_quality="rules_based_non_authoritative",
            authoritative=False,
            sessions=pd.DataFrame(rows, columns=["date", "open_utc", "close_utc"]),
        )


def resolve_calendar_provider(backend: str = "auto") -> CalendarProvider:
    if backend == "builtin":
        return BuiltinUSMarketRulesProvider()
    try:
        return SpecialistCalendarProvider(backend=backend)
    except RuntimeError:
        if backend != "auto":
            raise
        return BuiltinUSMarketRulesProvider()


def build_exchange_calendar(
    start_date: str | date,
    end_date: str | date,
    provider: CalendarProvider | None = None,
) -> tuple[pd.DataFrame, list[CalendarSchedule]]:
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date cannot be before start_date")
    provider = provider or resolve_calendar_provider()
    schedules = [provider.schedule(exchange, start, end) for exchange in EXCHANGES]
    rows: list[dict[str, Any]] = []
    closures = _closure_events(start, end)
    expected_early_closes = _early_close_events(start, end)

    for schedule in schedules:
        session_by_date = _normalize_sessions(schedule.sessions, start, end)
        for day in _date_range(start, end):
            session = session_by_date.get(day)
            if session is None:
                event_name = "weekend" if day.weekday() >= 5 else closures.get(day, "provider_reported_closure")
                if day.weekday() >= 5:
                    closure_type = "weekend"
                elif event_name in SPECIAL_CLOSURE_NAMES:
                    closure_type = "special_closure"
                elif event_name == "provider_reported_closure":
                    closure_type = "unexpected_weekday_closure"
                else:
                    closure_type = "exchange_holiday"
                rows.append(
                    _calendar_row(
                        schedule=schedule,
                        day=day,
                        is_session=False,
                        closure_type=closure_type,
                        event_name=event_name,
                    )
                )
                continue

            open_utc, close_utc = session
            close_local = close_utc.tz_convert(NEW_YORK)
            is_early = close_local.time() < time(16, 0)
            event_name = expected_early_closes.get(day, "provider_reported_early_close") if is_early else ""
            rows.append(
                _calendar_row(
                    schedule=schedule,
                    day=day,
                    is_session=True,
                    open_utc=open_utc,
                    close_utc=close_utc,
                    event_name=event_name,
                    is_early_close=is_early,
                )
            )

    calendar = pd.DataFrame(rows, columns=CALENDAR_COLUMNS)
    calendar = calendar.sort_values(["exchange", "date"]).reset_index(drop=True)
    return calendar, schedules


def build_tca_assumption_grid(assumptions: TCAAssumptions) -> pd.DataFrame:
    assumptions.validate()
    rows = []
    impact_at_cap = assumptions.impact_bps_at_reference_participation * math.sqrt(
        assumptions.max_participation_rate / assumptions.reference_participation_rate
    )
    for exchange in EXCHANGES:
        for side in ("buy", "sell"):
            regulatory_fee = assumptions.regulatory_sell_fee_bps if side == "sell" else 0.0
            explicit_fees = assumptions.commission_bps + assumptions.exchange_fee_bps + regulatory_fee
            modeled_slippage = assumptions.half_spread_bps + assumptions.impact_bps_at_reference_participation
            rows.append(
                {
                    "exchange": exchange,
                    "side": side,
                    "commission_bps": assumptions.commission_bps,
                    "exchange_fee_bps": assumptions.exchange_fee_bps,
                    "regulatory_fee_bps": regulatory_fee,
                    "half_spread_bps": assumptions.half_spread_bps,
                    "impact_model": "square_root_participation",
                    "impact_bps_at_reference_participation": assumptions.impact_bps_at_reference_participation,
                    "reference_participation_rate": assumptions.reference_participation_rate,
                    "max_participation_rate": assumptions.max_participation_rate,
                    "impact_bps_at_max_participation": impact_at_cap,
                    "estimated_cost_bps_at_reference": explicit_fees + modeled_slippage,
                    "source_quality": assumptions.source_quality,
                    "source_name": assumptions.source_name,
                }
            )
    return pd.DataFrame(rows)


def load_halt_suspension_evidence(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=HALT_COLUMNS)
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"halt/suspension input does not exist: {source}")
    if source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    elif source.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(source, keep_default_na=False)
    else:
        raise ValueError("halt/suspension input must be CSV or Parquet")
    missing = [column for column in HALT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"halt/suspension input missing required columns: {missing}")
    result = frame.loc[:, HALT_COLUMNS].copy()
    required_text = (
        "event_id",
        "exchange",
        "symbol",
        "event_type",
        "source_name",
        "source_record_id",
    )
    for column in required_text:
        blank = result[column].isna() | result[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(f"halt/suspension input has blank {column} values")
    result["exchange"] = result["exchange"].astype(str).str.upper()
    invalid_exchanges = sorted(
        set(result.loc[~result["exchange"].isin(HALT_EXCHANGES), "exchange"])
    )
    if invalid_exchanges:
        raise ValueError(f"halt/suspension input has unsupported exchanges: {invalid_exchanges}")
    for column in ("halt_start_utc", "observed_at_utc"):
        parsed = pd.to_datetime(result[column], errors="coerce", utc=True)
        if parsed.isna().any():
            raise ValueError(f"halt/suspension input has invalid {column} values")
        result[column] = parsed
    raw_end = result["halt_end_utc"]
    parsed_end = pd.to_datetime(raw_end, errors="coerce", utc=True)
    invalid_end = raw_end.notna() & raw_end.astype(str).str.strip().ne("") & parsed_end.isna()
    if invalid_end.any():
        raise ValueError("halt/suspension input has invalid halt_end_utc values")
    result["halt_end_utc"] = parsed_end
    ended = result["halt_end_utc"].notna()
    if (result.loc[ended, "halt_end_utc"] < result.loc[ended, "halt_start_utc"]).any():
        raise ValueError("halt_end_utc cannot precede halt_start_utc")
    if result["event_id"].astype(str).duplicated().any():
        raise ValueError("halt/suspension event_id values must be unique")
    return result.sort_values(["halt_start_utc", "exchange", "symbol"]).reset_index(drop=True)


def load_tca_assumptions_from_calibration_manifest(path: str | Path) -> TCAAssumptions:
    manifest_path, manifest = _load_calibration_manifest(path)
    artifacts = manifest.get("artifacts", {})
    assumptions_path = _calibration_artifact_path(
        manifest_path,
        artifacts.get("tca_assumptions_json") if isinstance(artifacts, dict) else None,
    )
    if assumptions_path is None or not assumptions_path.is_file():
        raise ValueError("calibration manifest has no readable tca_assumptions_json artifact")
    try:
        payload = json.loads(assumptions_path.read_text(encoding="utf-8"))
        assumptions = TCAAssumptions(**payload)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("calibration tca_assumptions_json artifact is invalid") from exc
    assumptions.validate()
    return assumptions


def _load_calibration_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"calibration manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("calibration manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("calibration manifest must contain a JSON object")
    return manifest_path, manifest


def _calibration_artifact_path(manifest_path: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    artifact_path = Path(value).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = manifest_path.parent / artifact_path
    return artifact_path.resolve()


def _validate_research_calibration_manifest(
    path: str | Path | None,
    assumptions: TCAAssumptions,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "provided": path is not None,
        "valid": False,
        "manifest_path": str(Path(path).expanduser().resolve()) if path is not None else None,
        "filtered_rows": 0,
        "entity_count": 0,
        "bounded_scanner": False,
        "artifacts_and_checksums_valid": False,
        "assumptions_match": False,
        "errors": [],
    }
    errors: list[str] = result["errors"]
    if path is None:
        if assumptions.source_quality == RESEARCH_TCA_SOURCE_QUALITY:
            errors.append("market-data research TCA requires a calibration manifest")
        return result

    manifest_path, manifest = _load_calibration_manifest(path)
    if manifest.get("calibration_type") != RESEARCH_TCA_SOURCE_QUALITY:
        errors.append("calibration_type must be market_data_calibrated_research")
    if manifest.get("status") != RESEARCH_TCA_SOURCE_QUALITY:
        errors.append("calibration status must be market_data_calibrated_research")
    if manifest.get("research_only") is not True:
        errors.append("calibration manifest must set research_only=true")
    for field in ("live_trading_ready", "ibkr_ready", "production_ready"):
        if manifest.get(field) is not False:
            errors.append(f"calibration manifest must set {field}=false")

    sample_count = manifest.get("sample_count", {})
    filtered_rows = sample_count.get("filtered_rows", 0) if isinstance(sample_count, dict) else 0
    if not isinstance(filtered_rows, int) or isinstance(filtered_rows, bool):
        filtered_rows = 0
    result["filtered_rows"] = filtered_rows
    if filtered_rows < MIN_RESEARCH_CALIBRATION_ROWS:
        errors.append(
            f"calibration filtered_rows must be at least {MIN_RESEARCH_CALIBRATION_ROWS}"
        )

    entities = manifest.get("entities", {})
    entity_count = entities.get("count", 0) if isinstance(entities, dict) else 0
    entity_values = entities.get("values", []) if isinstance(entities, dict) else []
    if not isinstance(entity_count, int) or isinstance(entity_count, bool):
        entity_count = 0
    unique_entities = {
        value.strip() for value in entity_values if isinstance(value, str) and value.strip()
    } if isinstance(entity_values, list) else set()
    result["entity_count"] = entity_count
    if entity_count < MIN_RESEARCH_CALIBRATION_ENTITIES:
        errors.append(
            f"calibration entity count must be at least {MIN_RESEARCH_CALIBRATION_ENTITIES}"
        )
    if entity_count != len(unique_entities):
        errors.append("calibration entity count does not match unique entity values")

    scanner = manifest.get("scanner", {})
    configured_batch_size = scanner.get("configured_batch_size", 0) if isinstance(scanner, dict) else 0
    max_batch_rows = scanner.get("max_batch_rows_observed", 0) if isinstance(scanner, dict) else 0
    batches_processed = scanner.get("batches_processed", 0) if isinstance(scanner, dict) else 0
    bounded_scanner = bool(
        isinstance(scanner, dict)
        and scanner.get("engine") == "pyarrow.dataset.Scanner"
        and scanner.get("memory_bounded") is True
        and scanner.get("use_threads") is False
        and isinstance(configured_batch_size, int)
        and not isinstance(configured_batch_size, bool)
        and configured_batch_size > 0
        and isinstance(max_batch_rows, int)
        and not isinstance(max_batch_rows, bool)
        and 0 < max_batch_rows <= configured_batch_size
        and isinstance(batches_processed, int)
        and not isinstance(batches_processed, bool)
        and batches_processed > 0
    )
    result["bounded_scanner"] = bounded_scanner
    if not bounded_scanner:
        errors.append("calibration scanner is not demonstrably batch bounded")

    artifacts = manifest.get("artifacts", {})
    checksums = manifest.get("artifact_sha256", {})
    artifacts_valid = isinstance(artifacts, dict) and isinstance(checksums, dict)
    if artifacts_valid and not REQUIRED_CALIBRATION_ARTIFACTS.issubset(artifacts):
        artifacts_valid = False
    if artifacts_valid and not REQUIRED_CALIBRATION_ARTIFACTS.issubset(checksums):
        artifacts_valid = False
    if artifacts_valid:
        for name in REQUIRED_CALIBRATION_ARTIFACTS:
            artifact_path = _calibration_artifact_path(manifest_path, artifacts.get(name))
            expected_digest = checksums.get(name)
            if (
                artifact_path is None
                or not artifact_path.is_file()
                or not isinstance(expected_digest, str)
                or len(expected_digest) != 64
                or _sha256(artifact_path) != expected_digest.lower()
            ):
                artifacts_valid = False
                break
    result["artifacts_and_checksums_valid"] = artifacts_valid
    if not artifacts_valid:
        errors.append("calibration artifacts or SHA-256 checksums are invalid")

    assumptions_match = False
    if isinstance(artifacts, dict):
        assumptions_path = _calibration_artifact_path(
            manifest_path, artifacts.get("tca_assumptions_json")
        )
        if assumptions_path is not None and assumptions_path.is_file():
            try:
                calibrated_payload = json.loads(assumptions_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                calibrated_payload = None
            if isinstance(calibrated_payload, dict):
                expected = asdict(assumptions)
                assumptions_match = set(calibrated_payload) == set(expected) and all(
                    _calibration_value_matches(expected[key], calibrated_payload[key])
                    for key in expected
                )
    result["assumptions_match"] = assumptions_match
    if not assumptions_match:
        errors.append("TCA assumptions do not match the calibrated assumptions artifact")

    result["valid"] = not errors
    return result


def _calibration_value_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(expected, float(actual), rel_tol=1e-12, abs_tol=1e-12)
        )
    return expected == actual


def build_evidence_quality(
    calendar: pd.DataFrame,
    schedules: list[CalendarSchedule],
    halt_suspensions: pd.DataFrame,
    halt_source_quality: str,
    tca_assumptions: TCAAssumptions,
    calibration_validation: dict[str, Any] | None = None,
) -> pd.DataFrame:
    expected_calendar_rows = int(calendar["date"].nunique() * len(EXCHANGES))
    calendar_row_coverage = len(calendar) / expected_calendar_rows if expected_calendar_rows else 0.0
    authoritative_exchanges = sum(schedule.authoritative for schedule in schedules)
    calendar_complete = calendar_row_coverage == 1.0 and authoritative_exchanges == len(EXCHANGES)
    halt_complete = not halt_suspensions.empty and halt_source_quality == OFFICIAL_HALT_SOURCE_QUALITY
    research_calibration_ready = bool(
        tca_assumptions.source_quality == RESEARCH_TCA_SOURCE_QUALITY
        and calibration_validation
        and calibration_validation.get("valid") is True
    )
    tca_backtest_ready = bool(
        research_calibration_ready
        or tca_assumptions.source_quality in VALIDATED_TCA_SOURCE_QUALITIES
    )
    tca_production_ready = bool(
        tca_assumptions.source_quality in VALIDATED_TCA_SOURCE_QUALITIES
    )
    if research_calibration_ready:
        tca_status = "complete_research_backtest_only"
        tca_notes = "Validated market-data calibration for research backtests; no execution fills"
    elif tca_assumptions.source_quality == RESEARCH_TCA_SOURCE_QUALITY:
        tca_status = "incomplete_invalid_research_calibration"
        errors = (calibration_validation or {}).get("errors", [])
        tca_notes = "; ".join(str(value) for value in errors) or "Calibration manifest not supplied"
    elif tca_production_ready:
        tca_status = "complete_execution_calibrated"
        tca_notes = "TCA assumptions are backed by an empirically calibrated execution sample"
    else:
        tca_status = "incomplete_uncalibrated_assumptions"
        tca_notes = "Fees, half-spread, square-root impact, and participation caps are unvalidated"
    return pd.DataFrame(
        [
            {
                "component": "exchange_calendars",
                "status": "complete" if calendar_complete else "incomplete_non_authoritative_source",
                "row_count": len(calendar),
                "coverage": calendar_row_coverage,
                "source_quality": ";".join(sorted(set(calendar["source_quality"].astype(str)))),
                "authoritative": authoritative_exchanges == len(EXCHANGES),
                "backtest_ready": calendar_complete,
                "production_ready": calendar_complete,
                "notes": "NYSE and Nasdaq daily sessions, closures, and early closes",
            },
            {
                "component": "halt_suspension_events",
                "status": "complete" if halt_complete else "incomplete_missing_official_source",
                "row_count": len(halt_suspensions),
                "coverage": 1.0 if halt_complete else 0.0,
                "source_quality": halt_source_quality,
                "authoritative": halt_source_quality == OFFICIAL_HALT_SOURCE_QUALITY,
                "backtest_ready": halt_complete,
                "production_ready": halt_complete,
                "notes": (
                    f"{len(halt_suspensions)} official halt/suspension rows retained"
                    if halt_complete
                    else "Empty schema is a placeholder, not evidence of no halts"
                ),
            },
            {
                "component": "tca_assumptions",
                "status": tca_status,
                "row_count": len(EXCHANGES) * 2,
                "coverage": 1.0,
                "source_quality": tca_assumptions.source_quality,
                "authoritative": tca_production_ready,
                "backtest_ready": tca_backtest_ready,
                "production_ready": tca_production_ready,
                "notes": tca_notes,
            },
        ]
    )


def write_usa_exchange_calendar_tca_artifacts(
    artifact_root: str | Path,
    run_id: str,
    start_date: str | date,
    end_date: str | date,
    *,
    calendar_backend: str = "auto",
    calendar_provider: CalendarProvider | None = None,
    tca_assumptions: TCAAssumptions | None = None,
    calibration_manifest_input: str | Path | None = None,
    halt_suspension_input: str | Path | None = None,
    halt_source_quality: str = "missing_official_source",
    halt_source_name: str = "not_provided",
    overwrite: bool = False,
) -> USAExchangeCalendarTCAArtifacts:
    if not run_id.strip() or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path-safe name")
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    assumptions = tca_assumptions
    if assumptions is None and calibration_manifest_input is not None:
        assumptions = load_tca_assumptions_from_calibration_manifest(calibration_manifest_input)
    assumptions = assumptions or TCAAssumptions()
    assumptions.validate()
    provider = calendar_provider or resolve_calendar_provider(calendar_backend)
    calendar, schedules = build_exchange_calendar(start, end, provider=provider)
    tca_grid = build_tca_assumption_grid(assumptions)
    halts = load_halt_suspension_evidence(halt_suspension_input)
    calibration_validation = _validate_research_calibration_manifest(
        calibration_manifest_input,
        assumptions,
    )
    quality = build_evidence_quality(
        calendar,
        schedules,
        halts,
        halt_source_quality,
        assumptions,
        calibration_validation,
    )

    artifact_dir = Path(artifact_root) / run_id
    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"artifact directory is not empty: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    calendar_path = artifact_dir / "exchange_calendar.csv"
    tca_path = artifact_dir / "tca_assumptions.csv"
    halt_path = artifact_dir / "halt_suspension_events.csv"
    quality_path = artifact_dir / "evidence_quality.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"

    _atomic_write_csv(calendar, calendar_path)
    _atomic_write_csv(tca_grid, tca_path)
    _atomic_write_csv(halts, halt_path)
    _atomic_write_csv(quality, quality_path)

    backtest_ready = bool(quality["backtest_ready"].all())
    production_ready = bool(quality["production_ready"].all())
    live_execution_ready = production_ready
    status = "complete" if backtest_ready else "incomplete"
    summary = _summary_markdown(
        run_id=run_id,
        start=start,
        end=end,
        calendar=calendar,
        quality=quality,
        assumptions=assumptions,
        halt_source_name=halt_source_name,
        backtest_ready=backtest_ready,
        production_ready=production_ready,
    )
    _atomic_write_text(summary_path, summary)

    artifact_paths = {
        "exchange_calendar": calendar_path,
        "tca_assumptions": tca_path,
        "halt_suspension_events": halt_path,
        "evidence_quality": quality_path,
        "summary": summary_path,
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": status,
        "backtest_ready": backtest_ready,
        "backtest_scorecard_eligible": backtest_ready,
        "production_ready": production_ready,
        "live_execution_ready": live_execution_ready,
        "live_trading_ready": live_execution_ready,
        "ibkr_ready": live_execution_ready,
        "scorecard_eligible": production_ready,
        "does_not_assert_5_of_5": not backtest_ready,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "exchanges": list(EXCHANGES),
        "calendar": {
            "status": quality.loc[quality["component"].eq("exchange_calendars"), "status"].iloc[0],
            "providers": [
                {
                    "exchange": item.exchange,
                    "calendar_code": item.calendar_code,
                    "provider": item.source_provider,
                    "version": item.source_version,
                    "source_quality": item.source_quality,
                    "authoritative": item.authoritative,
                }
                for item in schedules
            ],
            "row_count": len(calendar),
            "session_count": int(calendar["is_session"].sum()),
            "early_close_count": int(calendar["is_early_close"].sum()),
        },
        "halt_suspensions": {
            "status": quality.loc[quality["component"].eq("halt_suspension_events"), "status"].iloc[0],
            "row_count": len(halts),
            "source_name": halt_source_name,
            "source_quality": halt_source_quality,
            "placeholder_only": halts.empty,
            "no_events_claimed": False if halts.empty else None,
        },
        "tca": {
            "status": quality.loc[quality["component"].eq("tca_assumptions"), "status"].iloc[0],
            "assumptions": asdict(assumptions),
            "model": "explicit_fees_plus_half_spread_plus_square_root_participation_impact",
            "calibration_manifest": calibration_validation,
            "execution_fills_supplied": bool(
                quality.loc[
                    quality["component"].eq("tca_assumptions"), "production_ready"
                ].iloc[0]
            ),
        },
        "quality_fields": {
            "required_components": int(len(quality)),
            "complete_components": int(quality["backtest_ready"].sum()),
            "incomplete_components": quality.loc[~quality["backtest_ready"], "component"].tolist(),
            "backtest_complete_components": int(quality["backtest_ready"].sum()),
            "backtest_incomplete_components": quality.loc[
                ~quality["backtest_ready"], "component"
            ].tolist(),
            "production_complete_components": int(quality["production_ready"].sum()),
            "production_incomplete_components": quality.loc[
                ~quality["production_ready"], "component"
            ].tolist(),
            "calendar_row_coverage": float(
                quality.loc[quality["component"].eq("exchange_calendars"), "coverage"].iloc[0]
            ),
            "official_halt_source_present": bool(
                quality.loc[quality["component"].eq("halt_suspension_events"), "production_ready"].iloc[0]
            ),
            "tca_empirically_validated": bool(
                quality.loc[quality["component"].eq("tca_assumptions"), "backtest_ready"].iloc[0]
            ),
            "tca_research_calibration_validated": calibration_validation.get("valid") is True,
            "execution_fill_evidence_present": bool(
                quality.loc[
                    quality["component"].eq("tca_assumptions"), "production_ready"
                ].iloc[0]
            ),
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "artifacts": {name: str(path.resolve()) for name, path in artifact_paths.items()},
        "sha256": {name: _sha256(path) for name, path in artifact_paths.items()},
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return USAExchangeCalendarTCAArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        calendar_path=calendar_path,
        tca_assumptions_path=tca_path,
        halt_suspension_path=halt_path,
        evidence_quality_path=quality_path,
        summary_path=summary_path,
        calendar=calendar,
        tca_assumptions=tca_grid,
        halt_suspensions=halts,
        evidence_quality=quality,
    )


def _calendar_row(
    schedule: CalendarSchedule,
    day: date,
    is_session: bool,
    open_utc: pd.Timestamp | None = None,
    close_utc: pd.Timestamp | None = None,
    closure_type: str = "",
    event_name: str = "",
    is_early_close: bool = False,
) -> dict[str, Any]:
    if is_session:
        assert open_utc is not None and close_utc is not None
        open_local = open_utc.tz_convert(NEW_YORK)
        close_local = close_utc.tz_convert(NEW_YORK)
        minutes = int((close_utc - open_utc).total_seconds() / 60)
    else:
        open_local = close_local = open_utc = close_utc = None
        minutes = 0
    return {
        "exchange": schedule.exchange,
        "calendar_code": schedule.calendar_code,
        "date": day.isoformat(),
        "day_name": day.strftime("%A"),
        "is_session": is_session,
        "session_open_local": open_local.isoformat() if open_local is not None else "",
        "session_close_local": close_local.isoformat() if close_local is not None else "",
        "session_open_utc": open_utc.isoformat() if open_utc is not None else "",
        "session_close_utc": close_utc.isoformat() if close_utc is not None else "",
        "scheduled_minutes": minutes,
        "closure_type": closure_type,
        "event_name": event_name,
        "is_early_close": is_early_close,
        "source_provider": schedule.source_provider,
        "source_version": schedule.source_version,
        "source_quality": schedule.source_quality,
    }


def _normalize_sessions(
    sessions: pd.DataFrame, start_date: date, end_date: date
) -> dict[date, tuple[pd.Timestamp, pd.Timestamp]]:
    required = {"date", "open_utc", "close_utc"}
    if not required.issubset(sessions.columns):
        raise ValueError(f"calendar provider schedule missing columns: {sorted(required - set(sessions.columns))}")
    result: dict[date, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for row in sessions.itertuples(index=False):
        session_date = _parse_date(row.date, "provider session date")
        if not start_date <= session_date <= end_date:
            continue
        if session_date in result:
            raise ValueError(f"calendar provider returned duplicate session date: {session_date}")
        open_utc = _as_utc_timestamp(row.open_utc)
        close_utc = _as_utc_timestamp(row.close_utc)
        if close_utc <= open_utc:
            raise ValueError(f"calendar provider returned non-positive session: {session_date}")
        result[session_date] = (open_utc, close_utc)
    return result


def _closure_events(start_date: date, end_date: date) -> dict[date, str]:
    events: dict[date, str] = {}
    for year in range(start_date.year - 1, end_date.year + 2):
        _add_event(events, _new_year_observed(year), "new_years_day")
        if year >= 1998:
            _add_event(events, _nth_weekday(year, 1, 0, 3), "martin_luther_king_jr_day")
        _add_event(events, _nth_weekday(year, 2, 0, 3), "washingtons_birthday")
        _add_event(events, _easter_date(year) - timedelta(days=2), "good_friday")
        _add_event(events, _last_weekday(year, 5, 0), "memorial_day")
        if year >= 2022:
            _add_event(events, _observed_fixed_holiday(year, 6, 19), "juneteenth")
        _add_event(events, _observed_fixed_holiday(year, 7, 4), "independence_day")
        _add_event(events, _nth_weekday(year, 9, 0, 1), "labor_day")
        _add_event(events, _nth_weekday(year, 11, 3, 4), "thanksgiving_day")
        _add_event(events, _observed_fixed_holiday(year, 12, 25), "christmas_day")

    special = {
        date(1994, 4, 27): "national_day_of_mourning_nixon",
        date(2001, 9, 11): "september_11_market_closure",
        date(2001, 9, 12): "september_11_market_closure",
        date(2001, 9, 13): "september_11_market_closure",
        date(2001, 9, 14): "september_11_market_closure",
        date(2004, 6, 11): "national_day_of_mourning_reagan",
        date(2007, 1, 2): "national_day_of_mourning_ford",
        date(2012, 10, 29): "hurricane_sandy_market_closure",
        date(2012, 10, 30): "hurricane_sandy_market_closure",
        date(2018, 12, 5): "national_day_of_mourning_bush",
        date(2025, 1, 9): "national_day_of_mourning_carter",
    }
    events.update(special)
    return {day: name for day, name in events.items() if start_date <= day <= end_date}


def _early_close_events(start_date: date, end_date: date) -> dict[date, str]:
    closures = _closure_events(start_date, end_date)
    events: dict[date, str] = {}
    for year in range(start_date.year, end_date.year + 1):
        thanksgiving = _nth_weekday(year, 11, 3, 4)
        candidates = {
            thanksgiving + timedelta(days=1): "day_after_thanksgiving_early_close",
            date(year, 7, 3): "pre_independence_day_early_close",
            date(year, 12, 24): "christmas_eve_early_close",
        }
        for day, name in candidates.items():
            if day.weekday() < 5 and day not in closures:
                events[day] = name
    return {day: name for day, name in events.items() if start_date <= day <= end_date}


def _summary_markdown(
    run_id: str,
    start: date,
    end: date,
    calendar: pd.DataFrame,
    quality: pd.DataFrame,
    assumptions: TCAAssumptions,
    halt_source_name: str,
    backtest_ready: bool,
    production_ready: bool,
) -> str:
    status = "BACKTEST READY" if backtest_ready else "INCOMPLETE"
    lines = [
        f"# USA Exchange Calendar and TCA Evidence: {run_id}",
        "",
        f"**Evidence status:** {status}",
        "",
        f"Coverage window: {start.isoformat()} through {end.isoformat()} (inclusive).",
        f"Calendar rows: {len(calendar):,}; sessions: {int(calendar['is_session'].sum()):,}; early closes: {int(calendar['is_early_close'].sum()):,}.",
        f"Halt/suspension source: {halt_source_name}.",
        f"TCA source quality: {assumptions.source_quality}.",
        f"Research backtest ready: {str(backtest_ready).lower()}.",
        f"Live production ready: {str(production_ready).lower()}.",
        "",
        "## Quality Gates",
        "",
        "| Component | Status | Backtest ready | Production ready | Notes |",
        "|---|---|---:|---:|---|",
    ]
    for row in quality.itertuples(index=False):
        lines.append(
            f"| {row.component} | {row.status} | {str(bool(row.backtest_ready)).lower()} | "
            f"{str(bool(row.production_ready)).lower()} | {row.notes} |"
        )
    if not backtest_ready:
        lines.extend(
            [
                "",
                "This artifact does not assert a 5/5 research-backtest scorecard result. Empty halt/suspension rows are an input placeholder and are not evidence that no events occurred.",
            ]
        )
    lines.extend(
        [
            "",
            "Production, IBKR, and live execution remain false until execution-fill evidence is supplied and validated.",
        ]
    )
    return "\n".join(lines) + "\n"


def _date_range(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _parse_date(value: str | date | datetime | pd.Timestamp, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field} is not a valid date: {value!r}")
    return pd.Timestamp(parsed).date()


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _package_version(module_name: str) -> str:
    package_name = module_name.replace("_", "-")
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _add_event(events: dict[date, str], day: date, name: str) -> None:
    events[day] = name


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def _new_year_observed(year: int) -> date:
    holiday = date(year, 1, 1)
    return holiday + timedelta(days=1) if holiday.weekday() == 6 else holiday


def _nth_weekday(year: int, month: int, weekday: int, ordinal: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (ordinal - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter_date(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
