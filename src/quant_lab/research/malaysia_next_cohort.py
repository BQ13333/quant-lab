from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_lab.research.bounded_loop import (
    BASE_COST_BPS_ONE_WAY,
    STRESS_COST_BPS_ONE_WAY,
    MalaysiaUniverseContract,
)
from quant_lab.research.institutional_workflow import MalaysiaAuditInputs
from quant_lab.research.malaysia_pilot import MalaysiaHistoricalBacktester
from quant_lab.research.next_cohort_protocol import (
    MATERIALIZED_PROTOCOL_ID,
    materialized_next_cohort_protocol_hash,
    validate_materialized_next_cohort_protocol,
)


PRICE_VOLUME_ID = "MYS-S-NEXT-PVCONF-21-60"
ILLIQUIDITY_ID = "MYS-S-NEXT-ILLIQ-60"
TURN_OF_MONTH_ID = "MYS-S-NEXT-TOM-L1-F3"
OPENING_GAP_ID = "MYS-S-NEXT-GAPREV-X20"
EXPECTED_STRATEGY_IDS = (
    PRICE_VOLUME_ID,
    ILLIQUIDITY_ID,
    TURN_OF_MONTH_ID,
    OPENING_GAP_ID,
)


@dataclass(frozen=True)
class SelectionDiagnostics:
    signal_dates: int
    evaluated_signal_dates: int
    zero_exposure_dates: int
    minimum_valid_count: int
    minimum_selected_count: int
    maximum_selected_count: int
    selected_bad_current_ohlc_rows: int = 0


@dataclass
class NextCohortComputation:
    hypothesis_id: str
    strategy_id: str
    family: str
    u3_gross_returns: pd.Series
    u3_base_returns: pd.Series
    u3_stress_returns: pd.Series
    u3_weights: pd.DataFrame
    u3_turnover: pd.Series
    u3_trade_weights: pd.DataFrame
    u3_asset_net_contributions: pd.DataFrame
    u2_base_returns: pd.Series
    u2_stress_returns: pd.Series
    selection_diagnostics: SelectionDiagnostics
    lineage_receipt: dict[str, Any]
    calendar_receipt: dict[str, Any] | None
    primary_comparison_series: pd.Series | None

    def validate(self, universe: MalaysiaUniverseContract) -> None:
        if self.strategy_id not in EXPECTED_STRATEGY_IDS:
            raise ValueError("next-cohort strategy identity is invalid")
        if tuple(self.u3_weights.columns) != tuple(universe.u3):
            raise ValueError("next-cohort U3 columns drifted")
        if not self.u3_trade_weights.index.equals(self.u3_weights.index):
            raise ValueError("next-cohort trade and position calendars differ")
        if not self.u3_trade_weights.columns.equals(self.u3_weights.columns):
            raise ValueError("next-cohort trade and position universes differ")
        if not self.u3_asset_net_contributions.index.equals(self.u3_weights.index):
            raise ValueError("next-cohort contribution calendar drifted")
        aligned = (
            self.u3_gross_returns.index,
            self.u3_base_returns.index,
            self.u3_stress_returns.index,
            self.u3_weights.index,
            self.u3_turnover.index,
            self.u2_base_returns.index,
            self.u2_stress_returns.index,
        )
        if any(not aligned[0].equals(index) for index in aligned[1:]):
            raise ValueError("next-cohort return calendars are not identical")
        numeric = pd.concat(
            [
                self.u3_gross_returns,
                self.u3_base_returns,
                self.u3_stress_returns,
                self.u3_turnover,
                self.u2_base_returns,
                self.u2_stress_returns,
            ],
            axis=1,
        )
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("next-cohort return evidence must be finite")
        for frame in (self.u3_weights, self.u3_trade_weights, self.u3_asset_net_contributions):
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError("next-cohort portfolio evidence must be finite")
        if float(self.u3_weights.abs().sum(axis=1).max()) > 1.0 + 1e-12:
            raise ValueError("next-cohort gross exposure exceeded one")
        if (self.u3_turnover < -1e-12).any():
            raise ValueError("next-cohort turnover cannot be negative")
        if self.strategy_id == OPENING_GAP_ID:
            expected = self.u3_weights.abs().sum(axis=1) * 2.0
            if not np.allclose(self.u3_turnover, expected, atol=1e-12, rtol=0.0):
                raise ValueError("opening-gap turnover must include entry and exit")


class MalaysiaNextCohortEngine:
    """One-shot deterministic engine for the human-frozen Malaysia Draft 02 wave."""

    REQUIRED_COLUMNS = (
        "date",
        "ticker",
        "country",
        "open",
        "close",
        "adj_close",
        "simple_return",
        "overnight_return",
        "intraday_return",
        "volume",
        "adv_60_local_lag1",
        "vol_63_lag1",
        "amihud_60_lag1",
        "stale_close_flag",
        "bad_ohlc_flag",
    )

    def __init__(
        self,
        *,
        daily_features_path: str | Path,
        universe: MalaysiaUniverseContract,
        protocol: Mapping[str, Any],
        research_end: str = "2026-07-06",
    ) -> None:
        validate_materialized_next_cohort_protocol(protocol)
        if protocol.get("protocol_id") != MATERIALIZED_PROTOCOL_ID:
            raise ValueError("next-cohort engine requires materialized Draft 02")
        observed = tuple(item["strategy_id"] for item in protocol["hypotheses"])
        if observed != EXPECTED_STRATEGY_IDS:
            raise ValueError("next-cohort strategy order drifted")
        self.daily_features_path = Path(daily_features_path)
        self.universe = universe
        self.protocol = dict(protocol)
        self.protocol_hash = materialized_next_cohort_protocol_hash(protocol)
        self.research_end = pd.Timestamp(research_end)
        self._matrices: dict[str, pd.DataFrame] | None = None
        self._calendar_receipt: dict[str, Any] | None = None

    def load_matrices(self) -> dict[str, pd.DataFrame]:
        if self._matrices is not None:
            return {key: value.copy() for key, value in self._matrices.items()}
        raw = pd.read_parquet(self.daily_features_path, columns=list(self.REQUIRED_COLUMNS))
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.loc[
            raw["country"].eq("Malaysia")
            & raw["ticker"].isin(self.universe.u2)
            & raw["date"].le(self.research_end)
        ].sort_values(["date", "ticker"])
        if raw.empty or raw["date"].nunique() < 504:
            raise ValueError("insufficient Malaysia next-cohort history")
        if raw.duplicated(["date", "ticker"]).any():
            raise ValueError("Malaysia next-cohort input has duplicate date-ticker rows")
        matrices: dict[str, pd.DataFrame] = {}
        for column in self.REQUIRED_COLUMNS:
            if column in {"date", "ticker", "country"}:
                continue
            matrices[column] = raw.pivot(index="date", columns="ticker", values=column).reindex(
                columns=self.universe.u2
            )
        index = matrices["adj_close"].index
        if any(not index.equals(frame.index) for frame in matrices.values()):
            raise ValueError("Malaysia next-cohort feature calendars differ")
        self._matrices = matrices
        return {key: value.copy() for key, value in matrices.items()}

    def compute_all(self) -> tuple[NextCohortComputation, ...]:
        matrices = self.load_matrices()
        output = (
            self._compute_price_volume(matrices),
            self._compute_illiquidity(matrices),
            self._compute_turn_of_month(matrices),
            self._compute_opening_gap(matrices),
        )
        for item in output:
            item.validate(self.universe)
        return output

    def audit_inputs(self, computation: NextCohortComputation) -> MalaysiaAuditInputs:
        matrices = self.load_matrices()
        u3_returns = matrices["simple_return"].reindex(columns=self.universe.u3).fillna(0.0)
        prices = matrices["adj_close"].reindex(columns=self.universe.u3)
        volatility = matrices["vol_63_lag1"].reindex(columns=self.universe.u3)
        adv = matrices["adv_60_local_lag1"].reindex(columns=self.universe.u3)
        market = u3_returns.mean(axis=1).fillna(0.0)
        factors = pd.DataFrame(
            {
                "market": market,
                "momentum": MalaysiaHistoricalBacktester._cross_sectional_factor_return(
                    prices.pct_change(126, fill_method=None).shift(1), u3_returns
                ),
                "low_volatility": MalaysiaHistoricalBacktester._cross_sectional_factor_return(
                    -volatility.shift(1), u3_returns
                ),
                "lagged_liquidity": MalaysiaHistoricalBacktester._cross_sectional_factor_return(
                    np.log1p(adv.shift(1)), u3_returns
                ),
            }
        ).fillna(0.0)
        return MalaysiaAuditInputs(
            strategy_returns=computation.u3_base_returns.copy(),
            market_returns=market,
            factor_returns=factors,
            regime_labels=MalaysiaHistoricalBacktester._causal_regime_labels(market),
        )

    def capacity_metrics(
        self,
        computation: NextCohortComputation,
        notionals_myr: tuple[int, ...] = (1_000_000, 10_000_000, 100_000_000),
    ) -> dict[str, Any]:
        adv = self.load_matrices()["adv_60_local_lag1"].reindex(columns=self.universe.u3)
        valid_adv = adv.where(adv > 0.0)
        maximums: dict[str, float] = {}
        for notional in notionals_myr:
            participation = computation.u3_trade_weights.mul(float(notional)).div(valid_adv)
            maximums[str(notional)] = _finite_max(participation)
        pnl = computation.u3_asset_net_contributions.sum(axis=0)
        positive = pnl.clip(lower=0.0)
        positive_total = float(positive.sum())
        maximum_positive_share = (
            float(positive.max() / positive_total) if positive_total > 0.0 else 1.0
        )
        return {
            "maximum_trade_participation_by_notional_myr": maximums,
            "reference_notional_myr": 10_000_000,
            "reference_maximum_trade_participation": maximums["10000000"],
            "maximum_positive_pnl_concentration": maximum_positive_share,
        }

    def registered_momentum_factor(self) -> pd.Series:
        matrices = self.load_matrices()
        returns = matrices["simple_return"].reindex(columns=self.universe.u3).fillna(0.0)
        prices = matrices["adj_close"].reindex(columns=self.universe.u3)
        return MalaysiaHistoricalBacktester._cross_sectional_factor_return(
            prices.pct_change(126, fill_method=None).shift(1), returns
        )

    def _compute_price_volume(
        self, matrices: Mapping[str, pd.DataFrame]
    ) -> NextCohortComputation:
        price = matrices["adj_close"]
        volume_lag1 = matrices["volume"].shift(1)
        adv = matrices["adv_60_local_lag1"]
        r21 = price.pct_change(21, fill_method=None).shift(1)
        abnormal_volume = np.log(volume_lag1.where(volume_lag1 > 0.0) / adv.where(adv > 0.0))
        score = _cross_sectional_z(_cross_sectional_winsor(r21, 0.025)) * _cross_sectional_z(
            _cross_sectional_winsor(abnormal_volume, 0.025)
        ).clip(lower=0.0)
        quality, lineage = self._common_quality_mask(matrices, (r21, abnormal_volume))
        score = score.where(quality)
        return self._sampled_long_only(
            hypothesis_id="MYS-H-NEXT-PVCONF-21-60",
            strategy_id=PRICE_VOLUME_ID,
            family="price_volume_interaction",
            score=score,
            matrices=matrices,
            frequency="weekly",
            lineage=lineage,
        )

    def _compute_illiquidity(
        self, matrices: Mapping[str, pd.DataFrame]
    ) -> NextCohortComputation:
        raw_score = np.log(matrices["amihud_60_lag1"].where(matrices["amihud_60_lag1"] > 0.0))
        score = _cross_sectional_winsor(raw_score, 0.025)
        quality, lineage = self._common_quality_mask(matrices, (raw_score,))
        score = score.where(quality)
        return self._sampled_long_only(
            hypothesis_id="MYS-H-NEXT-ILLIQ-60",
            strategy_id=ILLIQUIDITY_ID,
            family="liquidity_premium",
            score=score,
            matrices=matrices,
            frequency="monthly",
            lineage=lineage,
        )

    def _compute_turn_of_month(
        self, matrices: Mapping[str, pd.DataFrame]
    ) -> NextCohortComputation:
        index = matrices["simple_return"].index
        sessions, calendar_receipt = self._frozen_calendar_sessions(index)
        analysis_index = index.intersection(sessions).sort_values()
        if len(analysis_index) < 504:
            raise ValueError("turn-of-month intersection calendar has insufficient history")
        inside = turn_of_month_session_mask(analysis_index, sessions)
        quality, lineage = self._common_quality_mask(matrices, ())
        u3_quality = quality.reindex(index=analysis_index, columns=self.universe.u3)
        u2_quality = quality.reindex(index=analysis_index, columns=self.universe.u2)
        u3_weights = _equal_weight_valid(u3_quality).mul(inside.astype(float), axis=0)
        u2_weights = _equal_weight_valid(u2_quality).mul(inside.astype(float), axis=0)
        u3 = _close_to_close_portfolio(
            u3_weights,
            matrices["simple_return"].reindex(index=analysis_index, columns=self.universe.u3),
            BASE_COST_BPS_ONE_WAY,
            STRESS_COST_BPS_ONE_WAY,
        )
        u2 = _close_to_close_portfolio(
            u2_weights,
            matrices["simple_return"].reindex(index=analysis_index, columns=self.universe.u2),
            BASE_COST_BPS_ONE_WAY,
            STRESS_COST_BPS_ONE_WAY,
        )
        market = (
            _equal_weight_valid(u3_quality)
            * matrices["simple_return"]
            .reindex(index=analysis_index, columns=self.universe.u3)
            .fillna(0.0)
        ).sum(axis=1)
        contrast = calendar_contrast_series(market, inside)
        selected = u3_weights.gt(0.0).sum(axis=1)
        active_selected = selected.loc[inside]
        diagnostics = SelectionDiagnostics(
            signal_dates=int(inside.sum()),
            evaluated_signal_dates=int((active_selected > 0).sum()),
            zero_exposure_dates=int((inside & selected.eq(0)).sum()),
            minimum_valid_count=int(u3_quality.sum(axis=1).min()),
            minimum_selected_count=int(active_selected.min()) if len(active_selected) else 0,
            maximum_selected_count=int(active_selected.max()) if len(active_selected) else 0,
        )
        return NextCohortComputation(
            hypothesis_id="MYS-H-NEXT-TOM-L1-F3",
            strategy_id=TURN_OF_MONTH_ID,
            family="calendar_seasonality",
            u3_gross_returns=u3[0],
            u3_base_returns=u3[1],
            u3_stress_returns=u3[2],
            u3_weights=u3_weights,
            u3_turnover=u3[3],
            u3_trade_weights=u3[4],
            u3_asset_net_contributions=u3[5],
            u2_base_returns=u2[1],
            u2_stress_returns=u2[2],
            selection_diagnostics=diagnostics,
            lineage_receipt=lineage,
            calendar_receipt=calendar_receipt,
            primary_comparison_series=contrast,
        )

    def _compute_opening_gap(
        self, matrices: Mapping[str, pd.DataFrame]
    ) -> NextCohortComputation:
        gap = matrices["overnight_return"]
        intraday = matrices["intraday_return"]
        common, lineage = self._common_quality_mask(matrices, (gap,))
        current_bad = matrices["bad_ohlc_flag"].fillna(1.0).ne(0.0)
        outcome_valid = intraday.replace([np.inf, -np.inf], np.nan).notna()
        valid = common & ~current_bad & outcome_valid
        u3 = _opening_gap_portfolio(
            gap.reindex(columns=self.universe.u3),
            intraday.reindex(columns=self.universe.u3),
            valid.reindex(columns=self.universe.u3),
            base_cost_bps_one_way=45.0,
            stress_cost_bps_one_way=65.0,
        )
        u2 = _opening_gap_portfolio(
            gap.reindex(columns=self.universe.u2),
            intraday.reindex(columns=self.universe.u2),
            valid.reindex(columns=self.universe.u2),
            base_cost_bps_one_way=45.0,
            stress_cost_bps_one_way=65.0,
        )
        lineage = {
            **lineage,
            "current_bad_ohlc_excluded_as_execution_data_defect": int(current_bad.sum().sum()),
            "current_nonfinite_intraday_outcome_excluded": int((~outcome_valid).sum().sum()),
        }
        diagnostics = u3[6]
        return NextCohortComputation(
            hypothesis_id="MYS-H-NEXT-GAPREV-X20",
            strategy_id=OPENING_GAP_ID,
            family="overnight_intraday_decomposition",
            u3_gross_returns=u3[0],
            u3_base_returns=u3[1],
            u3_stress_returns=u3[2],
            u3_weights=u3[3],
            u3_turnover=u3[4],
            u3_trade_weights=u3[3].abs(),
            u3_asset_net_contributions=u3[5],
            u2_base_returns=u2[1],
            u2_stress_returns=u2[2],
            selection_diagnostics=diagnostics,
            lineage_receipt=lineage,
            calendar_receipt=None,
            primary_comparison_series=None,
        )

    def _sampled_long_only(
        self,
        *,
        hypothesis_id: str,
        strategy_id: str,
        family: str,
        score: pd.DataFrame,
        matrices: Mapping[str, pd.DataFrame],
        frequency: str,
        lineage: dict[str, Any],
    ) -> NextCohortComputation:
        u3_weights, u3_diagnostics = sampled_top_fraction_weights(
            score.reindex(columns=self.universe.u3), frequency=frequency
        )
        u2_weights, _ = sampled_top_fraction_weights(
            score.reindex(columns=self.universe.u2), frequency=frequency
        )
        u3 = _close_to_close_portfolio(
            u3_weights,
            matrices["simple_return"].reindex(columns=self.universe.u3),
            BASE_COST_BPS_ONE_WAY,
            STRESS_COST_BPS_ONE_WAY,
        )
        u2 = _close_to_close_portfolio(
            u2_weights,
            matrices["simple_return"].reindex(columns=self.universe.u2),
            BASE_COST_BPS_ONE_WAY,
            STRESS_COST_BPS_ONE_WAY,
        )
        return NextCohortComputation(
            hypothesis_id=hypothesis_id,
            strategy_id=strategy_id,
            family=family,
            u3_gross_returns=u3[0],
            u3_base_returns=u3[1],
            u3_stress_returns=u3[2],
            u3_weights=u3_weights,
            u3_turnover=u3[3],
            u3_trade_weights=u3[4],
            u3_asset_net_contributions=u3[5],
            u2_base_returns=u2[1],
            u2_stress_returns=u2[2],
            selection_diagnostics=u3_diagnostics,
            lineage_receipt=lineage,
            calendar_receipt=None,
            primary_comparison_series=None,
        )

    def _common_quality_mask(
        self,
        matrices: Mapping[str, pd.DataFrame],
        signal_inputs: tuple[pd.DataFrame, ...],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        volume_lag1 = matrices["volume"].shift(1)
        adv = matrices["adv_60_local_lag1"]
        stale_lag1 = matrices["stale_close_flag"].shift(1)
        bad_lag1 = matrices["bad_ohlc_flag"].shift(1)
        masks = {
            "zero_or_missing_lagged_volume": volume_lag1.isna() | volume_lag1.le(0.0),
            "nonpositive_lagged_ADV": adv.isna() | adv.le(0.0),
            "lagged_stale_close": stale_lag1.isna() | stale_lag1.ne(0.0),
            "lagged_bad_OHLC": bad_lag1.isna() | bad_lag1.ne(0.0),
        }
        if signal_inputs:
            nonfinite = pd.DataFrame(False, index=adv.index, columns=adv.columns)
            for value in signal_inputs:
                nonfinite |= value.replace([np.inf, -np.inf], np.nan).isna()
        else:
            nonfinite = pd.DataFrame(False, index=adv.index, columns=adv.columns)
        masks["nonfinite_input"] = nonfinite
        excluded = pd.DataFrame(False, index=adv.index, columns=adv.columns)
        for value in masks.values():
            excluded |= value
        receipt = {
            "protocol_hash": self.protocol_hash,
            "information_set": "signals use t-1 data or fields explicitly precomputed through t-1",
            "exclusion_counts": {key: int(value.sum().sum()) for key, value in masks.items()},
            "valid_observation_count": int((~excluded).sum().sum()),
            "total_observation_cells": int(excluded.size),
        }
        return ~excluded, receipt

    def _frozen_calendar_sessions(
        self, observation_index: pd.DatetimeIndex
    ) -> tuple[pd.DatetimeIndex, dict[str, Any]]:
        if self._calendar_receipt is not None:
            sessions = pd.DatetimeIndex(self._calendar_receipt["intersection_sessions"])
            receipt = dict(self._calendar_receipt)
            receipt.pop("intersection_sessions", None)
            return sessions, receipt
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XKLS")
        observations = pd.DatetimeIndex(observation_index).tz_localize(None).normalize().unique().sort_values()
        start = max(observations.min(), pd.Timestamp(calendar.first_session).tz_localize(None))
        end = min(observations.max(), pd.Timestamp(calendar.last_session).tz_localize(None))
        exchange = calendar.sessions_in_range(start, end).tz_localize(None).normalize()
        bounded_observations = observations[(observations >= start) & (observations <= end)]
        intersection = bounded_observations.intersection(exchange).sort_values()
        receipt = {
            "calendar_name": "XKLS",
            "calendar_library": "exchange_calendars",
            "calendar_library_version": str(xcals.__version__),
            "calendar_start": start.isoformat(),
            "calendar_end": end.isoformat(),
            "observation_session_count": len(bounded_observations),
            "exchange_session_count": len(exchange),
            "intersection_session_count": len(intersection),
            "observation_only_count": len(bounded_observations.difference(exchange)),
            "exchange_only_count": len(exchange.difference(bounded_observations)),
            "observation_only_sessions": [
                item.isoformat() for item in bounded_observations.difference(exchange)
            ],
            "exchange_only_sessions": [item.isoformat() for item in exchange.difference(bounded_observations)],
            "receipt_sha256": _digest(
                {
                    "calendar": "XKLS",
                    "version": str(xcals.__version__),
                    "intersection": [item.isoformat() for item in intersection],
                    "observation_only": [
                        item.isoformat() for item in bounded_observations.difference(exchange)
                    ],
                    "exchange_only": [item.isoformat() for item in exchange.difference(bounded_observations)],
                }
            ),
        }
        self._calendar_receipt = {
            **receipt,
            "intersection_sessions": [item.isoformat() for item in intersection],
        }
        return intersection, receipt


def sampled_top_fraction_weights(
    score: pd.DataFrame,
    *,
    frequency: str,
    top_fraction: float = 0.20,
    minimum_selected: int = 5,
) -> tuple[pd.DataFrame, SelectionDiagnostics]:
    if frequency not in {"weekly", "monthly"}:
        raise ValueError("sample frequency must be weekly or monthly")
    if not 0.0 < top_fraction < 1.0 or minimum_selected < 1:
        raise ValueError("sample selection parameters are invalid")
    period = score.index.to_period("W-FRI" if frequency == "weekly" else "M")
    signal_dates = pd.Series(score.index, index=score.index).groupby(period).max().to_list()
    sampled = score.loc[pd.DatetimeIndex(signal_dates)]
    targets = pd.DataFrame(0.0, index=sampled.index, columns=score.columns)
    valid_counts: list[int] = []
    selected_counts: list[int] = []
    zero = 0
    for date, raw_row in sampled.iterrows():
        row = raw_row.replace([np.inf, -np.inf], np.nan).dropna()
        if len(row) == 0:
            selected_counts.append(0)
            continue
        valid_counts.append(len(row))
        selected_count = int(math.ceil(top_fraction * len(row)))
        if selected_count < minimum_selected:
            zero += 1
            selected_counts.append(0)
            continue
        ordered = _ordered_tickers(row, descending=True)[:selected_count]
        targets.loc[date, ordered] = 1.0 / len(ordered)
        selected_counts.append(len(ordered))
    expanded = targets.reindex(score.index).ffill().shift(1).fillna(0.0)
    diagnostics = SelectionDiagnostics(
        signal_dates=len(sampled),
        evaluated_signal_dates=sum(value > 0 for value in selected_counts),
        zero_exposure_dates=zero,
        minimum_valid_count=min(valid_counts) if valid_counts else 0,
        minimum_selected_count=min((value for value in selected_counts if value > 0), default=0),
        maximum_selected_count=max(selected_counts) if selected_counts else 0,
    )
    return expanded, diagnostics


def turn_of_month_session_mask(
    full_index: pd.DatetimeIndex, valid_exchange_sessions: pd.DatetimeIndex
) -> pd.Series:
    valid = pd.DatetimeIndex(valid_exchange_sessions).tz_localize(None).normalize().unique().sort_values()
    inside: set[pd.Timestamp] = set()
    grouped = pd.Series(valid, index=valid).groupby(valid.to_period("M"))
    for _, values in grouped:
        dates = pd.DatetimeIndex(values.to_numpy())
        if dates[0].day <= 7:
            inside.update(pd.Timestamp(item) for item in dates[:3])
        if dates[-1].day >= 20:
            inside.add(pd.Timestamp(dates[-1]))
    normalized = pd.DatetimeIndex(full_index).tz_localize(None).normalize()
    return pd.Series(normalized.isin(inside), index=full_index, dtype=bool, name="turn_of_month")


def calendar_contrast_series(market_returns: pd.Series, inside: pd.Series) -> pd.Series:
    aligned = pd.concat([market_returns.rename("return"), inside.rename("inside")], axis=1).dropna()
    flag = aligned["inside"].astype(bool)
    inside_count = int(flag.sum())
    outside_count = int((~flag).sum())
    if inside_count < 16 or outside_count < 16:
        raise ValueError("calendar contrast requires at least sixteen inside and outside sessions")
    contrast = pd.Series(0.0, index=aligned.index, dtype=float, name="inside_minus_outside")
    contrast.loc[flag] = aligned.loc[flag, "return"] / (inside_count / len(aligned))
    contrast.loc[~flag] = -aligned.loc[~flag, "return"] / (outside_count / len(aligned))
    return contrast


def _opening_gap_portfolio(
    gap: pd.DataFrame,
    intraday_return: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    base_cost_bps_one_way: float,
    stress_cost_bps_one_way: float,
    bottom_fraction: float = 0.20,
    minimum_selected: int = 5,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, SelectionDiagnostics]:
    weights = pd.DataFrame(0.0, index=gap.index, columns=gap.columns)
    valid_counts: list[int] = []
    selected_counts: list[int] = []
    zero = 0
    selected_bad = 0
    for date in gap.index:
        row = gap.loc[date].where(valid.loc[date]).replace([np.inf, -np.inf], np.nan).dropna()
        valid_counts.append(len(row))
        capacity = int(math.ceil(bottom_fraction * len(row)))
        candidates = row.loc[row < 0.0]
        ordered = _ordered_tickers(candidates, descending=False)[:capacity]
        if len(ordered) < minimum_selected:
            zero += 1
            selected_counts.append(0)
            continue
        weights.loc[date, ordered] = 1.0 / len(ordered)
        selected_counts.append(len(ordered))
    gross_contributions = weights * intraday_return.reindex_like(weights).fillna(0.0)
    gross = gross_contributions.sum(axis=1)
    round_trip_trade_weights = weights.abs() * 2.0
    turnover = round_trip_trade_weights.sum(axis=1)
    base_asset = gross_contributions - round_trip_trade_weights * base_cost_bps_one_way / 10_000.0
    stress_asset = gross_contributions - round_trip_trade_weights * stress_cost_bps_one_way / 10_000.0
    diagnostics = SelectionDiagnostics(
        signal_dates=len(gap),
        evaluated_signal_dates=sum(value > 0 for value in selected_counts),
        zero_exposure_dates=zero,
        minimum_valid_count=min(valid_counts) if valid_counts else 0,
        minimum_selected_count=min(selected_counts) if selected_counts else 0,
        maximum_selected_count=max(selected_counts) if selected_counts else 0,
        selected_bad_current_ohlc_rows=selected_bad,
    )
    return (
        gross,
        base_asset.sum(axis=1),
        stress_asset.sum(axis=1),
        weights,
        turnover,
        base_asset,
        diagnostics,
    )


def _close_to_close_portfolio(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    base_cost_bps_one_way: float,
    stress_cost_bps_one_way: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    aligned_returns = returns.reindex_like(weights).fillna(0.0)
    trade_weights = weights.diff().abs().fillna(weights.abs())
    turnover = trade_weights.sum(axis=1)
    gross_asset = weights * aligned_returns
    gross = gross_asset.sum(axis=1)
    base_asset = gross_asset - trade_weights * base_cost_bps_one_way / 10_000.0
    stress_asset = gross_asset - trade_weights * stress_cost_bps_one_way / 10_000.0
    return gross, base_asset.sum(axis=1), stress_asset.sum(axis=1), turnover, trade_weights, base_asset


def _equal_weight_valid(valid: pd.DataFrame) -> pd.DataFrame:
    weights = valid.astype(float)
    return weights.div(weights.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


def _cross_sectional_winsor(frame: pd.DataFrame, tail: float) -> pd.DataFrame:
    lower = frame.quantile(tail, axis=1)
    upper = frame.quantile(1.0 - tail, axis=1)
    return frame.clip(lower=lower, upper=upper, axis=0)


def _cross_sectional_z(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0)


def _ordered_tickers(row: pd.Series, *, descending: bool) -> list[str]:
    protocol_tie_salt = MATERIALIZED_PROTOCOL_ID
    return sorted(
        (str(ticker) for ticker in row.index),
        key=lambda ticker: (
            -float(row[ticker]) if descending else float(row[ticker]),
            hashlib.sha256(f"{protocol_tie_salt}::{ticker}".encode("utf-8")).hexdigest(),
        ),
    )


def _finite_max(frame: pd.DataFrame) -> float:
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.max()) if len(finite) else float("inf")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def return_series_hash(series: pd.Series) -> str:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("next-cohort return hash requires a DatetimeIndex")
    values = series.astype(float).sort_index()
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("next-cohort return hash requires finite values")
    return _digest([[timestamp.isoformat(), float(value)] for timestamp, value in values.items()])
