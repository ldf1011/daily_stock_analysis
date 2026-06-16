# -*- coding: utf-8 -*-
"""Backtest orchestration service."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, select

from src.config import get_config
from src.core.backtest_engine import OVERALL_SENTINEL_CODE, BacktestEngine, EvaluationConfig
from src.market_phase_summary import extract_market_phase_summary, normalize_analysis_phase_bucket
from src.repositories.backtest_repo import BacktestRepository
from src.repositories.stock_repo import StockRepository
from src.schemas.decision_action import build_action_fields
from src.storage import BacktestResult, BacktestSummary, DatabaseManager
from src.utils.data_processing import parse_json_field

logger = logging.getLogger(__name__)


class BacktestService:
    """Service layer to run and query backtests."""

    MAX_DYNAMIC_SUMMARY_ROWS = 2000

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager.get_instance()
        self.repo = BacktestRepository(self.db)
        self.stock_repo = StockRepository(self.db)

    def run_backtest(
        self,
        *,
        code: Optional[str] = None,
        force: bool = False,
        eval_window_days: Optional[int] = None,
        min_age_days: Optional[int] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        config = get_config()

        if eval_window_days is None:
            eval_window_days = getattr(config, "backtest_eval_window_days", 10)
        if min_age_days is None:
            min_age_days = getattr(config, "backtest_min_age_days", 14)

        engine_version = getattr(config, "backtest_engine_version", "v1")
        neutral_band_pct = float(getattr(config, "backtest_neutral_band_pct", 2.0))

        eval_config = EvaluationConfig(
            eval_window_days=int(eval_window_days),
            neutral_band_pct=neutral_band_pct,
            engine_version=str(engine_version),
        )

        candidates = self.repo.get_candidates(
            code=code,
            min_age_days=int(min_age_days),
            limit=int(limit),
            eval_window_days=int(eval_window_days),
            engine_version=str(engine_version),
            force=force,
        )

        processed = 0
        completed = 0
        insufficient = 0
        errors = 0
        touched_codes: set[str] = set()

        results_to_save: List[BacktestResult] = []

        for analysis in candidates:
            processed += 1
            touched_codes.add(analysis.code)

            try:
                analysis_date = self._resolve_analysis_date(analysis)
                if analysis_date is None:
                    errors += 1
                    results_to_save.append(
                        BacktestResult(
                            analysis_history_id=analysis.id,
                            code=analysis.code,
                            eval_window_days=int(eval_window_days),
                            engine_version=str(engine_version),
                            eval_status="error",
                            evaluated_at=datetime.now(),
                            operation_advice=analysis.operation_advice,
                        )
                    )
                    continue
                start_daily = self.stock_repo.get_start_daily(code=analysis.code, analysis_date=analysis_date)

                if start_daily is None or start_daily.close is None:
                    self._try_fill_daily_data(code=analysis.code, analysis_date=analysis_date, eval_window_days=eval_window_days)
                    start_daily = self.stock_repo.get_start_daily(code=analysis.code, analysis_date=analysis_date)

                if start_daily is None or start_daily.close is None:
                    insufficient += 1
                    results_to_save.append(
                        BacktestResult(
                            analysis_history_id=analysis.id,
                            code=analysis.code,
                            analysis_date=analysis_date,
                            eval_window_days=int(eval_window_days),
                            engine_version=str(engine_version),
                            eval_status="insufficient_data",
                            evaluated_at=datetime.now(),
                            operation_advice=analysis.operation_advice,
                        )
                    )
                    continue

                forward_bars = self.stock_repo.get_forward_bars(
                    code=analysis.code,
                    analysis_date=start_daily.date,
                    eval_window_days=int(eval_window_days),
                )

                if len(forward_bars) < int(eval_window_days):
                    self._try_fill_daily_data(code=analysis.code, analysis_date=start_daily.date, eval_window_days=eval_window_days)
                    forward_bars = self.stock_repo.get_forward_bars(
                        code=analysis.code,
                        analysis_date=start_daily.date,
                        eval_window_days=int(eval_window_days),
                    )

                evaluation = BacktestEngine.evaluate_single(
                    operation_advice=analysis.operation_advice,
                    analysis_date=start_daily.date,
                    start_price=float(start_daily.close),
                    forward_bars=forward_bars,
                    stop_loss=analysis.stop_loss,
                    take_profit=analysis.take_profit,
                    config=eval_config,
                )

                status = evaluation.get("eval_status")
                if status == "insufficient_data":
                    insufficient += 1
                elif status == "completed":
                    completed += 1
                else:
                    errors += 1

                results_to_save.append(
                    BacktestResult(
                        analysis_history_id=analysis.id,
                        code=analysis.code,
                        analysis_date=evaluation.get("analysis_date"),
                        eval_window_days=int(evaluation.get("eval_window_days") or eval_window_days),
                        engine_version=str(evaluation.get("engine_version") or engine_version),
                        eval_status=str(evaluation.get("eval_status") or "error"),
                        evaluated_at=datetime.now(),
                        operation_advice=evaluation.get("operation_advice"),
                        position_recommendation=evaluation.get("position_recommendation"),
                        start_price=evaluation.get("start_price"),
                        end_close=evaluation.get("end_close"),
                        max_high=evaluation.get("max_high"),
                        min_low=evaluation.get("min_low"),
                        stock_return_pct=evaluation.get("stock_return_pct"),
                        direction_expected=evaluation.get("direction_expected"),
                        direction_correct=evaluation.get("direction_correct"),
                        outcome=evaluation.get("outcome"),
                        stop_loss=evaluation.get("stop_loss"),
                        take_profit=evaluation.get("take_profit"),
                        hit_stop_loss=evaluation.get("hit_stop_loss"),
                        hit_take_profit=evaluation.get("hit_take_profit"),
                        first_hit=evaluation.get("first_hit"),
                        first_hit_date=evaluation.get("first_hit_date"),
                        first_hit_trading_days=evaluation.get("first_hit_trading_days"),
                        simulated_entry_price=evaluation.get("simulated_entry_price"),
                        simulated_exit_price=evaluation.get("simulated_exit_price"),
                        simulated_exit_reason=evaluation.get("simulated_exit_reason"),
                        simulated_return_pct=evaluation.get("simulated_return_pct"),
                    )
                )

            except Exception as exc:
                errors += 1
                logger.error(f"回测失败: {analysis.code}#{analysis.id}: {exc}")
                results_to_save.append(
                    BacktestResult(
                        analysis_history_id=analysis.id,
                        code=analysis.code,
                        analysis_date=self._resolve_analysis_date(analysis),
                        eval_window_days=int(eval_window_days),
                        engine_version=str(engine_version),
                        eval_status="error",
                        evaluated_at=datetime.now(),
                        operation_advice=analysis.operation_advice,
                    )
                )

        saved = 0
        if results_to_save:
            saved = self.repo.save_results_batch(results_to_save, replace_existing=force)

        if saved:
            self._recompute_summaries(
                touched_codes=sorted(touched_codes),
                eval_window_days=int(eval_window_days),
                engine_version=str(engine_version),
            )

        return {
            "processed": processed,
            "saved": saved,
            "completed": completed,
            "insufficient": insufficient,
            "errors": errors,
        }

    def simulate_strategy(
        self,
        *,
        code: str,
        strategy: str = "momentum_quality",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        initial_cash: float = 100000.0,
        stop_loss_pct: float = 8.0,
        take_profit_pct: float = 20.0,
    ) -> Dict[str, Any]:
        """Run a close-to-close historical strategy simulation from daily bars."""
        if not code or not str(code).strip():
            raise ValueError("code is required")

        from data_provider.base import canonical_stock_code, normalize_stock_code

        raw_code = canonical_stock_code(str(code))
        normalized_code = canonical_stock_code(normalize_stock_code(raw_code))
        code_candidates = self._strategy_code_candidates(raw_code, normalized_code)
        effective_end = end_date or date.today()
        effective_start = start_date or (effective_end - timedelta(days=365))
        if effective_start > effective_end:
            raise ValueError("start_date cannot be after end_date")

        requested_days = max(80, int((effective_end - effective_start).days * 1.8) + 60)
        data_source = "db_cache"
        bars = self._load_strategy_bars(
            code_candidates=code_candidates,
            start_date=effective_start,
            end_date=effective_end,
            requested_days=requested_days,
        )
        if len(bars) < 30:
            try:
                from data_provider.base import DataFetcherManager

                manager = DataFetcherManager()
                df, source = manager.get_daily_data(
                    normalized_code,
                    start_date=effective_start.isoformat(),
                    end_date=effective_end.isoformat(),
                    days=requested_days,
                )
                if df is not None and not df.empty:
                    for candidate in code_candidates:
                        self.db.save_daily_data(df, code=candidate, data_source=source)
                    data_source = source
            except Exception as exc:
                logger.warning("历史策略模拟补行情失败: %s %s", raw_code, exc)
            bars = self._load_strategy_bars(
                code_candidates=code_candidates,
                start_date=effective_start,
                end_date=effective_end,
                requested_days=requested_days,
            )

        if len(bars) < 30:
            return {
                "code": raw_code,
                "strategy": strategy,
                "start_date": effective_start.isoformat(),
                "end_date": effective_end.isoformat(),
                "data_start_date": bars[0]["date"].isoformat() if bars else None,
                "data_end_date": bars[-1]["date"].isoformat() if bars else None,
                "bars_count": len(bars),
                "data_source": data_source,
                "trades": [],
                "trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate_pct": None,
                "total_return_pct": None,
                "buy_hold_return_pct": None,
                "max_drawdown_pct": None,
                "final_equity": float(initial_cash),
                "message": "可用日线少于 30 根，无法进行历史策略模拟。",
            }

        simulation = self._simulate_strategy_from_bars(
            bars=bars,
            strategy=strategy,
            initial_cash=float(initial_cash),
            stop_loss_pct=float(stop_loss_pct),
            take_profit_pct=float(take_profit_pct),
        )
        simulation.update(
            {
                "code": raw_code,
                "strategy": strategy,
                "start_date": effective_start.isoformat(),
                "end_date": effective_end.isoformat(),
                "data_start_date": bars[0]["date"].isoformat(),
                "data_end_date": bars[-1]["date"].isoformat(),
                "bars_count": len(bars),
                "data_source": data_source,
                "message": simulation.get("message") or "历史策略模拟完成。",
            }
        )
        return simulation

    @staticmethod
    def _strategy_code_candidates(raw_code: str, normalized_code: str) -> List[str]:
        candidates: List[str] = []
        for candidate in (raw_code, normalized_code):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        if normalized_code.isdigit() and len(normalized_code) == 6:
            if normalized_code.startswith(("5", "6", "9")):
                suffixed = f"{normalized_code}.SH"
            elif normalized_code.startswith(("0", "1", "2", "3")):
                suffixed = f"{normalized_code}.SZ"
            elif normalized_code.startswith(("4", "8")):
                suffixed = f"{normalized_code}.BJ"
            else:
                suffixed = ""
            if suffixed and suffixed not in candidates:
                candidates.append(suffixed)
        return candidates

    def _load_strategy_bars(
        self,
        *,
        code_candidates: List[str],
        start_date: date,
        end_date: date,
        requested_days: int,
    ) -> List[Dict[str, Any]]:
        best: List[Any] = []
        for candidate in code_candidates:
            rows = self.stock_repo.get_range(candidate, start_date, end_date)
            if len(rows) > len(best):
                best = rows

        if len(best) < 30:
            try:
                from src.services.history_loader import load_history_df

                df, _source = load_history_df(code_candidates[0], days=requested_days, target_date=end_date)
                if df is not None and not df.empty:
                    rows = []
                    for item in df.to_dict(orient="records"):
                        row_date = self._coerce_date(item.get("date"))
                        if row_date is None or row_date < start_date or row_date > end_date:
                            continue
                        rows.append(item)
                    return self._normalize_strategy_bars(rows)
            except Exception as exc:
                logger.debug("历史策略模拟 DB fallback 失败: %s", exc)

        return self._normalize_strategy_bars([row.to_dict() for row in best])

    @staticmethod
    def _coerce_date(value: Any) -> Optional[date]:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return datetime.strptime(value[:10], "%Y-%m-%d").date()
            except ValueError:
                return None
        return None

    @classmethod
    def _normalize_strategy_bars(cls, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        bars: List[Dict[str, Any]] = []
        for row in rows:
            row_date = cls._coerce_date(row.get("date"))
            close = cls._to_float(row.get("close"))
            if row_date is None or close is None or close <= 0:
                continue
            bars.append(
                {
                    "date": row_date,
                    "open": cls._to_float(row.get("open")) or close,
                    "high": cls._to_float(row.get("high")) or close,
                    "low": cls._to_float(row.get("low")) or close,
                    "close": close,
                    "volume": cls._to_float(row.get("volume")),
                    "pct_chg": cls._to_float(row.get("pct_chg")),
                    "volume_ratio": cls._to_float(row.get("volume_ratio")),
                }
            )
        bars.sort(key=lambda item: item["date"])
        cls._attach_strategy_indicators(bars)
        return bars

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            result = float(value)
            if result != result:
                return None
            return result
        except Exception:
            return None

    @staticmethod
    def _attach_strategy_indicators(bars: List[Dict[str, Any]]) -> None:
        for index, bar in enumerate(bars):
            closes = [item["close"] for item in bars[: index + 1]]
            highs = [item["high"] for item in bars[: index + 1]]
            lows = [item["low"] for item in bars[: index + 1]]
            volumes = [item.get("volume") or 0.0 for item in bars[: index + 1]]
            for window in (5, 10, 20, 60, 120):
                if len(closes) >= window:
                    bar[f"ma{window}"] = sum(closes[-window:]) / window
                else:
                    bar[f"ma{window}"] = None
            if len(volumes) >= 20:
                bar["volume_ma20"] = sum(volumes[-20:]) / 20
            else:
                bar["volume_ma20"] = None
            if index >= 1:
                prev_close = bars[index - 1]["close"]
                bar["pct_chg_calc"] = (bar["close"] - prev_close) / prev_close * 100 if prev_close else 0.0
                true_range = max(
                    bar["high"] - bar["low"],
                    abs(bar["high"] - prev_close),
                    abs(bar["low"] - prev_close),
                )
            else:
                bar["pct_chg_calc"] = 0.0
                true_range = bar["high"] - bar["low"]

            bar["close_up"] = index >= 1 and bar["close"] > bars[index - 1]["close"]

            if len(highs) > 20:
                prev_highs = highs[-21:-1]
                prev_lows = lows[-21:-1]
                bar["donchian_high20_prev"] = max(prev_highs)
                bar["donchian_low20_prev"] = min(prev_lows)
            else:
                bar["donchian_high20_prev"] = None
                bar["donchian_low20_prev"] = None

            tr_values = []
            for lookback in range(max(0, index - 13), index + 1):
                if lookback == 0:
                    tr_values.append(bars[lookback]["high"] - bars[lookback]["low"])
                    continue
                prev_close = bars[lookback - 1]["close"]
                tr_values.append(
                    max(
                        bars[lookback]["high"] - bars[lookback]["low"],
                        abs(bars[lookback]["high"] - prev_close),
                        abs(bars[lookback]["low"] - prev_close),
                    )
                )
            bar["atr14"] = sum(tr_values) / 14 if len(tr_values) == 14 else None

            gains: List[float] = []
            losses: List[float] = []
            for lookback in range(max(1, index - 13), index + 1):
                change = bars[lookback]["close"] - bars[lookback - 1]["close"]
                gains.append(max(change, 0.0))
                losses.append(abs(min(change, 0.0)))
            if len(gains) == 14 and len(losses) == 14:
                avg_gain = sum(gains) / 14
                avg_loss = sum(losses) / 14
                if avg_loss == 0:
                    bar["rsi14"] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    bar["rsi14"] = 100 - (100 / (1 + rs))
            else:
                bar["rsi14"] = None

            if len(highs) > 100:
                range_highs = highs[-101:-1]
                range_lows = lows[-101:-1]
                range_high = max(range_highs)
                range_low = min(range_lows)
                bar["range_high100_prev"] = range_high
                bar["range_low100_prev"] = range_low
                bar["range_width100_pct"] = ((range_high - range_low) / range_low * 100) if range_low > 0 else None
            else:
                bar["range_high100_prev"] = None
                bar["range_low100_prev"] = None
                bar["range_width100_pct"] = None

            if len(lows) >= 5:
                bar["low5"] = min(lows[-5:])
            else:
                bar["low5"] = min(lows)

            spread = bar["high"] - bar["low"]
            if spread > 0:
                bar["candle_close_pos"] = (bar["close"] - bar["low"]) / spread
                bar["lower_shadow_ratio"] = (min(bar["open"], bar["close"]) - bar["low"]) / spread
                bar["upper_shadow_ratio"] = (bar["high"] - max(bar["open"], bar["close"])) / spread
            else:
                bar["candle_close_pos"] = 0.5
                bar["lower_shadow_ratio"] = 0.0
                bar["upper_shadow_ratio"] = 0.0

    @classmethod
    def _strategy_entry_signal(cls, bars: List[Dict[str, Any]], index: int, strategy: str) -> bool:
        bar = bars[index]
        close = bar["close"]
        ma5 = bar.get("ma5")
        ma10 = bar.get("ma10")
        ma20 = bar.get("ma20")
        if not all(value is not None for value in (ma5, ma10, ma20)):
            return False
        volume = bar.get("volume") or 0.0
        volume_ma20 = bar.get("volume_ma20") or 0.0
        pct = bar.get("pct_chg") if bar.get("pct_chg") is not None else bar.get("pct_chg_calc") or 0.0
        prev_closes = [item["close"] for item in bars[max(0, index - 20) : index]]
        strategy_id = str(strategy or "momentum_quality")

        if strategy_id == "momentum_quality":
            return close > ma5 > ma10 > ma20 and pct > -2.5
        if strategy_id == "shrink_pullback":
            return ma5 > ma10 > ma20 and bar["low"] <= ma10 * 1.03 and close >= ma10 and pct > -4
        if strategy_id == "volume_breakout":
            return bool(prev_closes) and close >= max(prev_closes) * 0.995 and volume_ma20 > 0 and volume >= volume_ma20 * 1.2
        if strategy_id == "capital_heat":
            return close > ma10 and volume_ma20 > 0 and volume >= volume_ma20 * 1.15 and 0 <= pct <= 8
        if strategy_id in ("dual_low", "quality_value"):
            return close > ma20 and abs(close / ma20 - 1) <= 0.12 and pct > -3
        if strategy_id == "oversold_reversal":
            last_10 = [item["close"] for item in bars[max(0, index - 10) : index + 1]]
            drawdown = (close - max(last_10)) / max(last_10) * 100 if last_10 else 0.0
            return drawdown <= -8 and pct > 0
        if strategy_id == "donchian_breakout":
            upper = bar.get("donchian_high20_prev")
            return bool(upper and close > upper)
        if strategy_id == "rsi_composite":
            rsi = bar.get("rsi14")
            prev_rsi = bars[index - 1].get("rsi14") if index >= 1 else None
            low5 = bar.get("low5")
            ma120 = bar.get("ma120")
            recent_rsis = [item.get("rsi14") for item in bars[max(0, index - 10) : index]]
            had_oversold = any(value is not None and value < 30 for value in recent_rsis)
            had_deep_oversold = any(value is not None and value < 25 for value in recent_rsis)
            rebound_cross = bool(prev_rsi is not None and rsi is not None and prev_rsi <= 36 < rsi)
            deep_rebound = bool(prev_rsi is not None and rsi is not None and had_deep_oversold and rsi >= 30 and rsi > prev_rsi)
            price_rebound = bool(bar.get("close_up") and low5 and close >= low5 * 1.02)
            trend_ok = bool(ma120 and close >= ma120 * 0.60)
            volume_ok = bool(volume_ma20 > 0 and volume >= volume_ma20 * 0.45)
            return bool((had_oversold and rebound_cross) or deep_rebound) and price_rebound and trend_ok and volume_ok
        if strategy_id == "wyckoff_enhanced":
            range_high = bar.get("range_high100_prev")
            range_low = bar.get("range_low100_prev")
            width_pct = bar.get("range_width100_pct")
            if not range_high or not range_low or width_pct is None or width_pct > 35:
                return False
            spring = (
                bar["low"] < range_low
                and close > range_low
                and (bar.get("candle_close_pos") or 0.0) >= 0.55
                and (bar.get("lower_shadow_ratio") or 0.0) >= 0.35
            )
            sos = close > range_high and volume_ma20 > 0 and volume > volume_ma20 * 1.05
            recent_sos = any(
                (item.get("range_high100_prev") is not None)
                and item["close"] > item["range_high100_prev"]
                and (item.get("volume_ma20") or 0.0) > 0
                and (item.get("volume") or 0.0) > (item.get("volume_ma20") or 0.0) * 1.05
                for item in bars[max(0, index - 20) : index]
            )
            lps = (
                recent_sos
                and range_high * 0.97 <= close <= range_high * 1.03
                and volume_ma20 > 0
                and volume < volume_ma20 * 0.85
                and close >= bars[index - 1]["close"]
            )
            return bool(spring or sos or lps)
        return close > ma20 and ma5 >= ma10 and pct > -3

    @classmethod
    def _strategy_exit_signal(cls, bars: List[Dict[str, Any]], index: int, strategy: str) -> bool:
        bar = bars[index]
        close = bar["close"]
        ma10 = bar.get("ma10")
        ma20 = bar.get("ma20")
        ma5 = bar.get("ma5")
        strategy_id = str(strategy or "momentum_quality")
        if strategy_id in ("dual_low", "quality_value"):
            return bool(ma20 and close < ma20 * 0.97)
        if strategy_id == "oversold_reversal":
            return bool(ma10 and close < ma10 * 0.97)
        if strategy_id == "donchian_breakout":
            lower = bar.get("donchian_low20_prev")
            return bool(lower and close < lower)
        return bool((ma10 and close < ma10) or (ma5 and ma10 and ma5 < ma10))

    @classmethod
    def _strategy_custom_exit_reason(
        cls,
        *,
        bars: List[Dict[str, Any]],
        index: int,
        strategy: str,
        entry_index: int,
        entry_price: float,
        peak_close: float,
    ) -> str:
        bar = bars[index]
        close = float(bar["close"])
        ma60 = bar.get("ma60")
        atr14 = bar.get("atr14")
        trade_return_pct = (close - entry_price) / entry_price * 100 if entry_price else 0.0
        strategy_id = str(strategy or "momentum_quality")

        if strategy_id == "rsi_composite":
            entry_atr = bars[entry_index].get("atr14")
            if trade_return_pct <= -13:
                return "stop_loss"
            if entry_atr and close <= entry_price - entry_atr * 4:
                return "atr_stop"
            if peak_close >= entry_price * 1.12:
                trailing_floor = entry_price + (peak_close - entry_price) * 0.80
                if close <= trailing_floor:
                    return "trailing_take_profit"
            if (bar.get("rsi14") or 0.0) >= 75:
                return "rsi_overheat"
            if trade_return_pct > 0 and ma60 and close < ma60:
                return "signal_exit"
            if index - entry_index >= 120 and trade_return_pct < 0 and ma60 and close < ma60:
                return "signal_exit"
            return ""

        if strategy_id == "wyckoff_enhanced":
            range_high = bar.get("range_high100_prev")
            range_low = bar.get("range_low100_prev")
            if range_high and bar["high"] > range_high and close < range_high and (bar.get("upper_shadow_ratio") or 0.0) >= 0.30:
                return "upthrust"
            if range_high and close < range_high:
                return "signal_exit"
            if range_low and close < range_low:
                return "signal_exit"
            if ma60 and close < ma60:
                return "signal_exit"
            return ""

        if strategy_id == "donchian_breakout":
            lower = bar.get("donchian_low20_prev")
            if lower and close < lower:
                return "signal_exit"
            return ""

        return ""

    @classmethod
    def _simulate_strategy_from_bars(
        cls,
        *,
        bars: List[Dict[str, Any]],
        strategy: str,
        initial_cash: float,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> Dict[str, Any]:
        equity = float(initial_cash)
        peak = equity
        max_drawdown = 0.0
        in_position = False
        entry_price = 0.0
        entry_index = -1
        entry_date: Optional[date] = None
        entry_equity = equity
        peak_close_since_entry = 0.0
        trades: List[Dict[str, Any]] = []
        custom_exit_strategies = {"donchian_breakout", "rsi_composite", "wyckoff_enhanced"}

        for index, bar in enumerate(bars):
            close = float(bar["close"])
            marked_equity = equity
            if in_position and entry_price > 0:
                marked_equity = entry_equity * (close / entry_price)
            peak = max(peak, marked_equity)
            if peak > 0:
                max_drawdown = min(max_drawdown, (marked_equity - peak) / peak * 100)

            if index < 20:
                continue

            if not in_position:
                if cls._strategy_entry_signal(bars, index, strategy):
                    in_position = True
                    entry_price = close
                    entry_index = index
                    entry_date = bar["date"]
                    entry_equity = equity
                    peak_close_since_entry = close
                continue

            trade_return_pct = (close - entry_price) / entry_price * 100 if entry_price else 0.0
            peak_close_since_entry = max(peak_close_since_entry, close)
            exit_reason = ""
            strategy_id = str(strategy or "momentum_quality")
            if strategy_id in custom_exit_strategies:
                exit_reason = cls._strategy_custom_exit_reason(
                    bars=bars,
                    index=index,
                    strategy=strategy,
                    entry_index=entry_index,
                    entry_price=entry_price,
                    peak_close=peak_close_since_entry,
                )
                if not exit_reason and cls._strategy_exit_signal(bars, index, strategy):
                    exit_reason = "signal_exit"
            else:
                if stop_loss_pct > 0 and trade_return_pct <= -abs(stop_loss_pct):
                    exit_reason = "stop_loss"
                elif take_profit_pct > 0 and trade_return_pct >= abs(take_profit_pct):
                    exit_reason = "take_profit"
                elif cls._strategy_exit_signal(bars, index, strategy):
                    exit_reason = "signal_exit"

            if exit_reason:
                equity = entry_equity * (1 + trade_return_pct / 100)
                trades.append(
                    {
                        "entry_date": entry_date.isoformat() if entry_date else bar["date"].isoformat(),
                        "exit_date": bar["date"].isoformat(),
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(close, 4),
                        "return_pct": round(trade_return_pct, 2),
                        "holding_days": max(1, index - entry_index),
                        "exit_reason": exit_reason,
                    }
                )
                in_position = False
                entry_price = 0.0
                entry_index = -1
                entry_date = None
                entry_equity = equity
                peak_close_since_entry = 0.0

        if in_position and entry_price > 0:
            last = bars[-1]
            close = float(last["close"])
            trade_return_pct = (close - entry_price) / entry_price * 100
            equity = entry_equity * (1 + trade_return_pct / 100)
            trades.append(
                {
                    "entry_date": entry_date.isoformat() if entry_date else last["date"].isoformat(),
                    "exit_date": last["date"].isoformat(),
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(close, 4),
                    "return_pct": round(trade_return_pct, 2),
                    "holding_days": max(1, len(bars) - 1 - entry_index),
                    "exit_reason": "period_end",
                }
            )

        win_count = sum(1 for trade in trades if float(trade["return_pct"]) > 0)
        loss_count = sum(1 for trade in trades if float(trade["return_pct"]) < 0)
        trade_count = len(trades)
        buy_hold_return = (bars[-1]["close"] - bars[0]["close"]) / bars[0]["close"] * 100 if bars[0]["close"] else None
        total_return = (equity - initial_cash) / initial_cash * 100 if initial_cash else None
        return {
            "trades": trades,
            "trade_count": trade_count,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate_pct": round(win_count / trade_count * 100, 2) if trade_count else None,
            "total_return_pct": round(total_return, 2) if total_return is not None else None,
            "buy_hold_return_pct": round(buy_hold_return, 2) if buy_hold_return is not None else None,
            "max_drawdown_pct": round(abs(max_drawdown), 2),
            "final_equity": round(equity, 2),
            "message": "历史策略模拟完成。" if trade_count else "区间内没有触发策略入场信号。",
        }

    def get_recent_evaluations(
        self,
        *,
        code: Optional[str],
        eval_window_days: Optional[int] = None,
        limit: int = 50,
        page: int = 1,
        analysis_date_from: Optional[date] = None,
        analysis_date_to: Optional[date] = None,
        analysis_phase: Optional[str] = None,
    ) -> Dict[str, Any]:
        config = get_config()
        engine_version = str(getattr(config, "backtest_engine_version", "v1"))

        phase_bucket = self._normalize_phase_filter(analysis_phase)
        if eval_window_days is None and (analysis_date_from is not None or analysis_date_to is not None or phase_bucket is not None):
            eval_window_days = self._infer_eval_window_for_query(
                code=code,
                engine_version=engine_version,
                analysis_date_from=analysis_date_from,
                analysis_date_to=analysis_date_to,
            )
        if phase_bucket is not None:
            return self._get_recent_evaluations_by_phase(
                code=code,
                eval_window_days=eval_window_days,
                engine_version=engine_version,
                limit=limit,
                page=page,
                analysis_date_from=analysis_date_from,
                analysis_date_to=analysis_date_to,
                phase_bucket=phase_bucket,
            )

        offset = max(page - 1, 0) * limit
        rows, total = self.repo.get_results_paginated(
            code=code,
            eval_window_days=eval_window_days,
            engine_version=engine_version,
            analysis_date_from=analysis_date_from,
            analysis_date_to=analysis_date_to,
            days=None,
            offset=offset,
            limit=limit,
        )
        items = []
        for result, stock_name, trend_prediction, _created_at, context_snapshot, raw_result, report_type in rows:
            summary = extract_market_phase_summary(context_snapshot)
            items.append(
                self._result_to_dict(
                    result,
                    stock_name,
                    trend_prediction,
                    market_phase_summary=summary,
                    market_phase=self._phase_bucket_from_summary(summary),
                    raw_result=raw_result,
                    report_type=report_type,
                )
            )
        return {"total": total, "page": page, "limit": limit, "items": items}

    def get_summary(
        self,
        *,
        scope: str,
        code: Optional[str],
        eval_window_days: Optional[int] = None,
        analysis_date_from: Optional[date] = None,
        analysis_date_to: Optional[date] = None,
        analysis_phase: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        config = get_config()
        engine_version = str(getattr(config, "backtest_engine_version", "v1"))
        lookup_code = OVERALL_SENTINEL_CODE if scope == "overall" else code

        phase_bucket = self._normalize_phase_filter(analysis_phase)
        if analysis_date_from is not None or analysis_date_to is not None or phase_bucket is not None:
            if eval_window_days is None:
                eval_window_days = self._infer_eval_window_for_query(
                    code=code,
                    engine_version=engine_version,
                    analysis_date_from=analysis_date_from,
                    analysis_date_to=analysis_date_to,
                )
            ew = int(eval_window_days) if eval_window_days is not None else None
            count = self.repo.count_results(
                code=code,
                eval_window_days=ew,
                engine_version=engine_version,
                analysis_date_from=analysis_date_from,
                analysis_date_to=analysis_date_to,
            )
            if count > self.MAX_DYNAMIC_SUMMARY_ROWS:
                if phase_bucket is not None:
                    raise ValueError(
                        "Phase-filtered summary candidate set matches too many rows; "
                        "narrow the analysis date range, stock code, or evaluation window."
                    )
                raise ValueError("Date-filtered summary matches too many rows; narrow the analysis date range or stock code.")
            if phase_bucket is not None:
                rows_with_context = self.repo.list_results_with_context(
                    code=code,
                    eval_window_days=ew,
                    engine_version=engine_version,
                    analysis_date_from=analysis_date_from,
                    analysis_date_to=analysis_date_to,
                    limit=self.MAX_DYNAMIC_SUMMARY_ROWS + 1,
                )
                if len(rows_with_context) > self.MAX_DYNAMIC_SUMMARY_ROWS:
                    raise ValueError(
                        "Phase-filtered summary matches too many rows; narrow the analysis date range or stock code."
                    )
                filtered_pairs = [
                    (row, snapshot)
                    for row, snapshot in rows_with_context
                    if self._phase_bucket_from_snapshot(snapshot) == phase_bucket
                ]
                phase_counts = self._phase_counts_from_contexts([snapshot for _, snapshot in filtered_pairs])
                filtered_rows = [row for row, _ in filtered_pairs]
                return self._build_dynamic_summary(
                    rows=filtered_rows,
                    scope=scope,
                    code=lookup_code,
                    eval_window_days=int(eval_window_days) if eval_window_days is not None else None,
                    engine_version=engine_version,
                    max_rows=self.MAX_DYNAMIC_SUMMARY_ROWS,
                    phase_breakdown=phase_counts["phase_breakdown"],
                    raw_phase_counts=phase_counts["raw_phase_counts"],
                )
            rows = self.repo.list_results(
                code=code,
                eval_window_days=ew,
                engine_version=engine_version,
                analysis_date_from=analysis_date_from,
                analysis_date_to=analysis_date_to,
            )
            return self._build_dynamic_summary(
                rows=rows,
                scope=scope,
                code=lookup_code,
                eval_window_days=int(eval_window_days) if eval_window_days is not None else None,
                engine_version=engine_version,
                max_rows=self.MAX_DYNAMIC_SUMMARY_ROWS,
            )

        summary = self.repo.get_summary(
            scope=scope,
            code=lookup_code,
            eval_window_days=eval_window_days,
            engine_version=engine_version,
        )
        if summary is None:
            return None
        return self._summary_to_dict(summary)

    def get_global_summary(self, *, eval_window_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return overall backtest metrics normalized for Agent memory consumers."""
        return self._normalize_learning_summary(
            self.get_summary(scope="overall", code=None, eval_window_days=eval_window_days)
        )

    def get_stock_summary(self, code: str, *, eval_window_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return per-stock backtest metrics normalized for Agent memory consumers."""
        return self._normalize_learning_summary(
            self.get_summary(scope="stock", code=code, eval_window_days=eval_window_days)
        )

    def get_skill_summary(self, skill_id: str, *, eval_window_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return skill-like summary metrics for Agent memory consumers.

        The current backtest storage layer only persists overall / per-stock rollups.
        Re-using the overall rollup here would fabricate skill-specific performance
        and mislead auto-weighting. Until real skill-tagged summaries exist, return
        ``None`` so downstream callers fall back to neutral weighting.
        """
        return None

    def get_strategy_summary(self, strategy_id: str, *, eval_window_days: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Compatibility wrapper for legacy strategy-based callers."""
        summary = self.get_skill_summary(strategy_id, eval_window_days=eval_window_days)
        if summary is None:
            return None
        normalized = dict(summary)
        normalized["strategy_id"] = strategy_id
        return normalized

    def _infer_eval_window_for_query(
        self,
        *,
        code: Optional[str],
        engine_version: str,
        analysis_date_from: Optional[date],
        analysis_date_to: Optional[date],
    ) -> Optional[int]:
        windows = self.repo.get_distinct_eval_windows(
            code=code,
            engine_version=engine_version,
            analysis_date_from=analysis_date_from,
            analysis_date_to=analysis_date_to,
        )
        return windows[0] if windows else None

    def _get_recent_evaluations_by_phase(
        self,
        *,
        code: Optional[str],
        eval_window_days: Optional[int],
        engine_version: str,
        limit: int,
        page: int,
        analysis_date_from: Optional[date],
        analysis_date_to: Optional[date],
        phase_bucket: str,
    ) -> Dict[str, Any]:
        page_offset = max(page - 1, 0) * limit
        batch_size = max(100, min(500, limit * 4))
        sql_offset = 0
        scanned = 0
        matched_total = 0
        page_rows: List[
            Tuple[
                BacktestResult,
                Optional[str],
                Optional[str],
                Optional[Dict[str, Any]],
                str,
                Optional[str],
                Optional[str],
            ]
        ] = []

        while True:
            remaining_probe_rows = self.MAX_DYNAMIC_SUMMARY_ROWS + 1 - scanned
            if remaining_probe_rows <= 0:
                raise ValueError("Phase-filtered results match too many rows; narrow the analysis date range or stock code.")
            batch_limit = min(batch_size, remaining_probe_rows)
            batch = self.repo.get_results_with_context_batch(
                code=code,
                eval_window_days=eval_window_days,
                engine_version=engine_version,
                analysis_date_from=analysis_date_from,
                analysis_date_to=analysis_date_to,
                days=None,
                offset=sql_offset,
                limit=batch_limit,
            )
            if not batch:
                break
            scanned += len(batch)
            if scanned > self.MAX_DYNAMIC_SUMMARY_ROWS:
                raise ValueError("Phase-filtered results match too many rows; narrow the analysis date range or stock code.")
            sql_offset += len(batch)
            for (
                result,
                stock_name,
                trend_prediction,
                _created_at,
                context_snapshot,
                raw_result,
                report_type,
            ) in batch:
                summary = extract_market_phase_summary(context_snapshot)
                bucket = self._phase_bucket_from_summary(summary)
                if bucket != phase_bucket:
                    continue
                if matched_total >= page_offset and len(page_rows) < limit:
                    page_rows.append((result, stock_name, trend_prediction, summary, bucket, raw_result, report_type))
                matched_total += 1
            if len(batch) < batch_limit:
                break

        items = [
            self._result_to_dict(
                result,
                stock_name,
                trend_prediction,
                market_phase_summary=summary,
                market_phase=bucket,
                raw_result=raw_result,
                report_type=report_type,
            )
            for result, stock_name, trend_prediction, summary, bucket, raw_result, report_type in page_rows
        ]
        return {"total": matched_total, "page": page, "limit": limit, "items": items}

    @staticmethod
    def _normalize_phase_filter(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value or "").strip().lower()
        if not text or text == "all":
            return None
        allowed = {"premarket", "intraday", "postmarket", "unknown"}
        if text not in allowed:
            raise ValueError("analysis_phase must be one of premarket, intraday, postmarket, unknown")
        return text

    @staticmethod
    def _phase_bucket_from_summary(summary: Optional[Dict[str, Any]]) -> str:
        if not isinstance(summary, dict):
            return "unknown"
        return normalize_analysis_phase_bucket(summary.get("phase"))

    @classmethod
    def _phase_bucket_from_snapshot(cls, context_snapshot: Optional[str]) -> str:
        return cls._phase_bucket_from_summary(extract_market_phase_summary(context_snapshot))

    @classmethod
    def _phase_counts_from_contexts(cls, snapshots: List[Optional[str]]) -> Dict[str, Dict[str, int]]:
        phase_breakdown = {"premarket": 0, "intraday": 0, "postmarket": 0, "unknown": 0}
        raw_phase_counts: Dict[str, int] = {}
        for snapshot in snapshots:
            summary = extract_market_phase_summary(snapshot)
            raw_phase = str(summary.get("phase")) if isinstance(summary, dict) and summary.get("phase") else "unknown"
            raw_phase_counts[raw_phase] = raw_phase_counts.get(raw_phase, 0) + 1
            bucket = cls._phase_bucket_from_summary(summary)
            phase_breakdown[bucket] = phase_breakdown.get(bucket, 0) + 1
        return {"phase_breakdown": phase_breakdown, "raw_phase_counts": raw_phase_counts}

    def _resolve_analysis_date(self, analysis) -> Optional[date]:
        parsed = self.repo.parse_analysis_date_from_snapshot(analysis.context_snapshot)
        if parsed:
            return parsed
        if getattr(analysis, "created_at", None):
            return analysis.created_at.date()
        logger.warning(f"无法确定分析日期，跳过记录: {analysis.code}#{getattr(analysis, 'id', '?')}")
        return None

    def _try_fill_daily_data(self, *, code: str, analysis_date: date, eval_window_days: int) -> None:
        try:
            from data_provider.base import DataFetcherManager

            # fetch a window that covers start + forward bars
            end_date = analysis_date + timedelta(days=max(eval_window_days * 2, 30))
            manager = DataFetcherManager()
            df, source = manager.get_daily_data(
                stock_code=code,
                start_date=analysis_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                days=eval_window_days * 2,
            )
            if df is None or df.empty:
                return
            self.db.save_daily_data(df, code=code, data_source=source)
        except Exception as exc:
            logger.warning(f"补全日线数据失败({code}): {exc}")

    def _recompute_summaries(self, *, touched_codes: List[str], eval_window_days: int, engine_version: str) -> None:
        with self.db.get_session() as session:
            # overall
            overall_rows = session.execute(
                select(BacktestResult).where(
                    and_(
                        BacktestResult.eval_window_days == eval_window_days,
                        BacktestResult.engine_version == engine_version,
                    )
                )
            ).scalars().all()
            overall_data = BacktestEngine.compute_summary(
                results=overall_rows,
                scope="overall",
                code=OVERALL_SENTINEL_CODE,
                eval_window_days=eval_window_days,
                engine_version=engine_version,
            )
            overall_summary = self._build_summary_model(overall_data)
            self.repo.upsert_summary(overall_summary)

            for code in touched_codes:
                rows = session.execute(
                    select(BacktestResult).where(
                        and_(
                            BacktestResult.code == code,
                            BacktestResult.eval_window_days == eval_window_days,
                            BacktestResult.engine_version == engine_version,
                        )
                    )
                ).scalars().all()
                data = BacktestEngine.compute_summary(
                    results=rows,
                    scope="stock",
                    code=code,
                    eval_window_days=eval_window_days,
                    engine_version=engine_version,
                )
                summary = self._build_summary_model(data)
                self.repo.upsert_summary(summary)

    @staticmethod
    def _build_summary_model(summary_data: Dict[str, Any]) -> BacktestSummary:
        return BacktestSummary(
            scope=summary_data.get("scope"),
            code=summary_data.get("code"),
            eval_window_days=summary_data.get("eval_window_days"),
            engine_version=summary_data.get("engine_version"),
            computed_at=datetime.now(),
            total_evaluations=summary_data.get("total_evaluations") or 0,
            completed_count=summary_data.get("completed_count") or 0,
            insufficient_count=summary_data.get("insufficient_count") or 0,
            long_count=summary_data.get("long_count") or 0,
            cash_count=summary_data.get("cash_count") or 0,
            win_count=summary_data.get("win_count") or 0,
            loss_count=summary_data.get("loss_count") or 0,
            neutral_count=summary_data.get("neutral_count") or 0,
            direction_accuracy_pct=summary_data.get("direction_accuracy_pct"),
            win_rate_pct=summary_data.get("win_rate_pct"),
            neutral_rate_pct=summary_data.get("neutral_rate_pct"),
            avg_stock_return_pct=summary_data.get("avg_stock_return_pct"),
            avg_simulated_return_pct=summary_data.get("avg_simulated_return_pct"),
            stop_loss_trigger_rate=summary_data.get("stop_loss_trigger_rate"),
            take_profit_trigger_rate=summary_data.get("take_profit_trigger_rate"),
            ambiguous_rate=summary_data.get("ambiguous_rate"),
            avg_days_to_first_hit=summary_data.get("avg_days_to_first_hit"),
            advice_breakdown_json=json.dumps(summary_data.get("advice_breakdown") or {}, ensure_ascii=False),
            diagnostics_json=json.dumps(summary_data.get("diagnostics") or {}, ensure_ascii=False),
        )

    @staticmethod
    def _result_to_dict(
        row: BacktestResult,
        stock_name: Optional[str] = None,
        trend_prediction: Optional[str] = None,
        market_phase_summary: Optional[Dict[str, Any]] = None,
        market_phase: Optional[str] = None,
        raw_result: Optional[Any] = None,
        report_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        parsed_raw_result = parse_json_field(raw_result)
        raw = parsed_raw_result if isinstance(parsed_raw_result, dict) else {}
        action_fields = build_action_fields(
            operation_advice=raw.get("operation_advice") or row.operation_advice,
            explicit_action=raw.get("action"),
            report_type=report_type or ("market_review" if row.code == "market_review" else None),
            report_language=raw.get("report_language"),
        )
        return {
            "analysis_history_id": row.analysis_history_id,
            "code": row.code,
            "stock_name": stock_name,
            "analysis_date": row.analysis_date.isoformat() if row.analysis_date else None,
            "eval_window_days": row.eval_window_days,
            "engine_version": row.engine_version,
            "eval_status": row.eval_status,
            "evaluated_at": row.evaluated_at.isoformat() if row.evaluated_at else None,
            "operation_advice": row.operation_advice,
            "action": action_fields["action"],
            "action_label": action_fields["action_label"],
            "trend_prediction": trend_prediction,
            "market_phase": market_phase,
            "market_phase_summary": market_phase_summary,
            "position_recommendation": row.position_recommendation,
            "start_price": row.start_price,
            "end_close": row.end_close,
            "max_high": row.max_high,
            "min_low": row.min_low,
            "stock_return_pct": row.stock_return_pct,
            "actual_return_pct": row.stock_return_pct,
            "actual_movement": BacktestService._actual_movement_from_return(row.stock_return_pct),
            "direction_expected": row.direction_expected,
            "direction_correct": row.direction_correct,
            "outcome": row.outcome,
            "stop_loss": row.stop_loss,
            "take_profit": row.take_profit,
            "hit_stop_loss": row.hit_stop_loss,
            "hit_take_profit": row.hit_take_profit,
            "first_hit": row.first_hit,
            "first_hit_date": row.first_hit_date.isoformat() if row.first_hit_date else None,
            "first_hit_trading_days": row.first_hit_trading_days,
            "simulated_entry_price": row.simulated_entry_price,
            "simulated_exit_price": row.simulated_exit_price,
            "simulated_exit_reason": row.simulated_exit_reason,
            "simulated_return_pct": row.simulated_return_pct,
        }

    @staticmethod
    def _summary_to_dict(row: BacktestSummary) -> Dict[str, Any]:
        return {
            "scope": row.scope,
            "code": None if row.code == OVERALL_SENTINEL_CODE else row.code,
            "eval_window_days": row.eval_window_days,
            "engine_version": row.engine_version,
            "computed_at": row.computed_at.isoformat() if row.computed_at else None,
            "total_evaluations": row.total_evaluations,
            "completed_count": row.completed_count,
            "insufficient_count": row.insufficient_count,
            "long_count": row.long_count,
            "cash_count": row.cash_count,
            "win_count": row.win_count,
            "loss_count": row.loss_count,
            "neutral_count": row.neutral_count,
            "direction_accuracy_pct": row.direction_accuracy_pct,
            "win_rate_pct": row.win_rate_pct,
            "neutral_rate_pct": row.neutral_rate_pct,
            "avg_stock_return_pct": row.avg_stock_return_pct,
            "avg_simulated_return_pct": row.avg_simulated_return_pct,
            "stop_loss_trigger_rate": row.stop_loss_trigger_rate,
            "take_profit_trigger_rate": row.take_profit_trigger_rate,
            "ambiguous_rate": row.ambiguous_rate,
            "avg_days_to_first_hit": row.avg_days_to_first_hit,
            "advice_breakdown": json.loads(row.advice_breakdown_json) if row.advice_breakdown_json else {},
            "diagnostics": json.loads(row.diagnostics_json) if row.diagnostics_json else {},
        }

    @staticmethod
    def _normalize_learning_summary(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Normalize summary metrics to the ratio-based shape expected by Agent memory."""
        if summary is None:
            return None

        normalized = dict(summary)
        normalized["win_rate"] = BacktestService._pct_to_ratio(summary.get("win_rate_pct"), default=0.5)
        normalized["direction_accuracy"] = BacktestService._pct_to_ratio(
            summary.get("direction_accuracy_pct"),
            default=0.5,
        )

        avg_return_pct = summary.get("avg_simulated_return_pct")
        if avg_return_pct is None:
            avg_return_pct = summary.get("avg_stock_return_pct")
        normalized["avg_return"] = BacktestService._pct_to_ratio(avg_return_pct, default=0.0)
        return normalized

    @staticmethod
    def _pct_to_ratio(value: Optional[float], default: float = 0.0) -> float:
        try:
            return float(value) / 100.0
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _actual_movement_from_return(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        try:
            actual_return = float(value)
        except (TypeError, ValueError):
            return None
        if actual_return > 0:
            return "up"
        if actual_return < 0:
            return "down"
        return "flat"

    @staticmethod
    def _build_dynamic_summary(
        *,
        rows: List[BacktestResult],
        scope: str,
        code: Optional[str],
        eval_window_days: Optional[int],
        engine_version: str,
        max_rows: Optional[int] = None,
        phase_breakdown: Optional[Dict[str, int]] = None,
        raw_phase_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        filtered_rows = [row for row in rows if getattr(row, "engine_version", None) == engine_version]
        if eval_window_days is not None:
            summary_window_days = int(eval_window_days)
        else:
            window_values = sorted({
                int(row.eval_window_days)
                for row in filtered_rows
                if getattr(row, "eval_window_days", None) is not None
            })
            if len(window_values) > 1:
                logger.warning(
                    "Multiple eval_window_days values found for dynamic summary; using %s for engine_version=%s, scope=%s, code=%s",
                    window_values[0],
                    engine_version,
                    scope,
                    code,
                )
            if window_values:
                summary_window_days = window_values[0]
            else:
                summary_window_days = int(getattr(get_config(), "backtest_eval_window_days", 10))

        filtered_rows = [
            row for row in filtered_rows if getattr(row, "eval_window_days", None) == summary_window_days
        ]

        if max_rows is not None and len(filtered_rows) > max_rows:
            raise ValueError(
                "Date-filtered summary matches too many rows; narrow the analysis date range or stock code."
            )

        summary = BacktestEngine.compute_summary(
            results=filtered_rows,
            scope=scope,
            code=code,
            eval_window_days=summary_window_days,
            engine_version=engine_version,
        )
        diagnostics = summary.get("diagnostics")
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        if phase_breakdown is not None:
            diagnostics["phase_breakdown"] = phase_breakdown
        if raw_phase_counts is not None:
            diagnostics["raw_phase_counts"] = raw_phase_counts
        summary["diagnostics"] = diagnostics
        summary["code"] = None if summary.get("code") == OVERALL_SENTINEL_CODE else summary.get("code")
        summary["computed_at"] = datetime.now().isoformat()
        return summary
