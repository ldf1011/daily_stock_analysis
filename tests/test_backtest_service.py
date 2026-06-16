# -*- coding: utf-8 -*-
"""Integration tests for backtest service and repository.

These tests run against a temporary SQLite DB (same approach as other tests)
and validate idempotency/force semantics, result field correctness,
summary creation, and query methods.
"""

import json
import os
import tempfile
import unittest
from datetime import date, datetime
from unittest.mock import patch

from src.config import Config
from src.core.backtest_engine import OVERALL_SENTINEL_CODE
from src.services.backtest_service import BacktestService
from src.storage import AnalysisHistory, BacktestResult, BacktestSummary, DatabaseManager, StockDaily


class BacktestServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._db_path = os.path.join(self._temp_dir.name, "test_backtest_service.db")
        os.environ["DATABASE_PATH"] = self._db_path
        os.environ["BACKTEST_EVAL_WINDOW_DAYS"] = "3"

        Config._instance = None
        DatabaseManager.reset_instance()
        self.db = DatabaseManager.get_instance()

        # Ensure analysis is old enough for default min_age_days=14
        old_created_at = datetime(2024, 1, 1, 0, 0, 0)

        with self.db.get_session() as session:
            session.add(
                AnalysisHistory(
                    query_id="q1",
                    code="600519",
                    name="贵州茅台",
                    report_type="simple",
                    sentiment_score=80,
                    operation_advice="买入",
                    trend_prediction="看多",
                    analysis_summary="test",
                    stop_loss=95.0,
                    take_profit=110.0,
                    created_at=old_created_at,
                    context_snapshot=json.dumps(
                        {
                            "enhanced_context": {"date": "2024-01-01"},
                            "market_phase_summary": {
                                "phase": "premarket",
                                "market": "cn",
                                "trigger_source": "api",
                            },
                        }
                    ),
                )
            )

            # Analysis day close
            session.add(
                StockDaily(
                    code="600519",
                    date=date(2024, 1, 1),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                )
            )

            # Forward bars (3 days) that hit take-profit on day1
            session.add_all(
                [
                    StockDaily(code="600519", date=date(2024, 1, 2), high=111.0, low=100.0, close=105.0),
                    StockDaily(code="600519", date=date(2024, 1, 3), high=108.0, low=103.0, close=106.0),
                    StockDaily(code="600519", date=date(2024, 1, 4), high=109.0, low=104.0, close=107.0),
                ]
            )
            session.commit()

    def _seed_analysis(
        self,
        *,
        query_id: str,
        analysis_date: date,
        created_at: datetime,
        operation_advice: str,
        trend_prediction: str,
        start_close: float,
        forward_bars: list[StockDaily],
        phase: str = "intraday",
    ) -> None:
        with self.db.get_session() as session:
            session.add(
                AnalysisHistory(
                    query_id=query_id,
                    code="600519",
                    name="贵州茅台",
                    report_type="simple",
                    sentiment_score=60,
                    operation_advice=operation_advice,
                    trend_prediction=trend_prediction,
                    analysis_summary="extra-test",
                    stop_loss=None,
                    take_profit=None,
                    created_at=created_at,
                    context_snapshot=json.dumps(
                        {
                            "enhanced_context": {"date": analysis_date.isoformat()},
                            "market_phase_summary": {
                                "phase": phase,
                                "market": "cn",
                                "trigger_source": "api",
                            },
                        }
                    ),
                )
            )
            session.add(
                StockDaily(
                    code="600519",
                    date=analysis_date,
                    open=start_close,
                    high=start_close,
                    low=start_close,
                    close=start_close,
                )
            )
            session.add_all(forward_bars)
            session.commit()

    def tearDown(self) -> None:
        DatabaseManager.reset_instance()
        self._temp_dir.cleanup()

    def _count_results(self) -> int:
        with self.db.get_session() as session:
            return session.query(BacktestResult).count()

    def _make_backtest_result(
        self,
        *,
        analysis_history_id: int,
        analysis_date: date,
        eval_window_days: int = 1,
        engine_version: str = "v1",
    ) -> BacktestResult:
        return BacktestResult(
            analysis_history_id=analysis_history_id,
            code="600519",
            analysis_date=analysis_date,
            eval_window_days=eval_window_days,
            engine_version=engine_version,
            eval_status="completed",
            evaluated_at=datetime(2024, 1, 20, 0, 0, 0),
            operation_advice="买入",
            position_recommendation="long",
            start_price=100.0,
            end_close=101.0,
            stock_return_pct=1.0,
            direction_expected="up",
            direction_correct=True,
            outcome="win",
            simulated_return_pct=1.0,
        )

    def test_force_semantics(self) -> None:
        service = BacktestService(self.db)

        stats1 = service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)
        self.assertEqual(stats1["saved"], 1)
        self.assertEqual(self._count_results(), 1)

        # Non-force should be idempotent
        stats2 = service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)
        self.assertEqual(stats2["saved"], 0)
        self.assertEqual(self._count_results(), 1)

        # Force should replace existing result without unique constraint errors
        stats3 = service.run_backtest(code="600519", force=True, eval_window_days=3, min_age_days=0, limit=10)
        self.assertEqual(stats3["saved"], 1)
        self.assertEqual(self._count_results(), 1)

    def _run_and_get_result(self) -> BacktestResult:
        """Helper: run backtest and return the single BacktestResult row."""
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)
        with self.db.get_session() as session:
            return session.query(BacktestResult).one()

    def test_query_accepts_bare_and_suffixed_codes(self) -> None:
        """Backtest queries should match 600519 and 600519.SH interchangeably."""
        with self.db.get_session() as session:
            history = AnalysisHistory(
                query_id="q-suffix",
                code="600519.SH",
                name="贵州茅台",
                report_type="simple",
                sentiment_score=75,
                operation_advice="买入",
                trend_prediction="看多",
                analysis_summary="suffix-test",
                stop_loss=95.0,
                take_profit=110.0,
                created_at=datetime(2024, 1, 1, 0, 0, 0),
                context_snapshot=json.dumps(
                    {
                        "enhanced_context": {"date": "2024-01-01"},
                        "market_phase_summary": {
                            "phase": "premarket",
                            "market": "cn",
                            "trigger_source": "api",
                        },
                    }
                ),
            )
            session.add(history)
            session.flush()
            session.add(
                BacktestResult(
                    analysis_history_id=history.id,
                    code="600519.SH",
                    analysis_date=date(2024, 1, 1),
                    eval_window_days=3,
                    engine_version="v1",
                    eval_status="completed",
                    evaluated_at=datetime(2024, 1, 2, 0, 0, 0),
                    operation_advice="买入",
                    position_recommendation="long",
                    start_price=100.0,
                    end_close=101.0,
                    stock_return_pct=1.0,
                    direction_expected="up",
                    direction_correct=True,
                    outcome="win",
                    simulated_return_pct=1.0,
                )
            )
            session.commit()

        service = BacktestService(self.db)
        data = service.get_recent_evaluations(code="600519", eval_window_days=3, limit=10)
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["items"][0]["code"], "600519.SH")

    def test_result_fields_correct(self) -> None:
        """Verify BacktestResult row contains correct evaluation values."""
        result = self._run_and_get_result()

        self.assertEqual(result.eval_status, "completed")
        self.assertEqual(result.code, "600519")
        self.assertEqual(result.analysis_date, date(2024, 1, 1))
        self.assertEqual(result.operation_advice, "买入")
        self.assertEqual(result.position_recommendation, "long")
        self.assertEqual(result.direction_expected, "up")

        # Prices
        self.assertAlmostEqual(result.start_price, 100.0)
        self.assertAlmostEqual(result.end_close, 107.0)
        self.assertAlmostEqual(result.stock_return_pct, 7.0)

        # Direction & outcome
        self.assertEqual(result.outcome, "win")
        self.assertTrue(result.direction_correct)

        # Target hits -- day2 high=111 >= take_profit=110
        self.assertTrue(result.hit_take_profit)
        self.assertFalse(result.hit_stop_loss)
        self.assertEqual(result.first_hit, "take_profit")
        self.assertEqual(result.first_hit_trading_days, 1)
        self.assertEqual(result.first_hit_date, date(2024, 1, 2))

        # Simulated execution
        self.assertAlmostEqual(result.simulated_entry_price, 100.0)
        self.assertAlmostEqual(result.simulated_exit_price, 110.0)
        self.assertEqual(result.simulated_exit_reason, "take_profit")
        self.assertAlmostEqual(result.simulated_return_pct, 10.0)

    def test_summaries_created_after_run(self) -> None:
        """Verify both overall and per-stock BacktestSummary rows are created."""
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        with self.db.get_session() as session:
            # Overall summary uses sentinel code
            overall = session.query(BacktestSummary).filter(
                BacktestSummary.scope == "overall",
                BacktestSummary.code == OVERALL_SENTINEL_CODE,
            ).first()
            self.assertIsNotNone(overall)
            self.assertEqual(overall.total_evaluations, 1)
            self.assertEqual(overall.completed_count, 1)
            self.assertEqual(overall.win_count, 1)
            self.assertEqual(overall.loss_count, 0)
            self.assertAlmostEqual(overall.win_rate_pct, 100.0)

            # Stock-level summary
            stock = session.query(BacktestSummary).filter(
                BacktestSummary.scope == "stock",
                BacktestSummary.code == "600519",
            ).first()
            self.assertIsNotNone(stock)
            self.assertEqual(stock.total_evaluations, 1)
            self.assertEqual(stock.completed_count, 1)
            self.assertEqual(stock.win_count, 1)

    def test_get_summary_overall_returns_sentinel_as_none(self) -> None:
        """Verify get_summary translates __overall__ sentinel back to None."""
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        summary = service.get_summary(scope="overall", code=None)
        self.assertIsNotNone(summary)
        self.assertIsNone(summary["code"])
        self.assertEqual(summary["scope"], "overall")
        self.assertEqual(summary["win_count"], 1)

    def test_agent_learning_summary_helpers_keep_skill_rollups_neutral_until_supported(self) -> None:
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        global_summary = service.get_global_summary(eval_window_days=3)
        stock_summary = service.get_stock_summary("600519", eval_window_days=3)
        skill_summary = service.get_skill_summary("bull_trend", eval_window_days=3)
        strategy_summary = service.get_strategy_summary("bull_trend", eval_window_days=3)

        self.assertIsNotNone(global_summary)
        self.assertEqual(global_summary["total_evaluations"], 1)
        self.assertAlmostEqual(global_summary["win_rate"], 1.0)
        self.assertAlmostEqual(global_summary["direction_accuracy"], 1.0)
        self.assertAlmostEqual(global_summary["avg_return"], 0.10)

        self.assertIsNotNone(stock_summary)
        self.assertEqual(stock_summary["code"], "600519")
        self.assertAlmostEqual(stock_summary["win_rate"], 1.0)

        self.assertIsNone(skill_summary)
        self.assertIsNone(strategy_summary)

    def test_get_recent_evaluations(self) -> None:
        """Verify get_recent_evaluations returns correct paginated results."""
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        data = service.get_recent_evaluations(code="600519", limit=10, page=1)
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["page"], 1)
        self.assertEqual(data["limit"], 10)
        self.assertEqual(len(data["items"]), 1)

        item = data["items"][0]
        self.assertEqual(item["code"], "600519")
        self.assertEqual(item["outcome"], "win")
        self.assertEqual(item["direction_expected"], "up")
        self.assertTrue(item["direction_correct"])

    def test_get_recent_evaluations_prefers_persisted_raw_action(self) -> None:
        service = BacktestService(self.db)

        with self.db.get_session() as session:
            history = session.query(AnalysisHistory).filter(AnalysisHistory.query_id == "q1").one()
            history.operation_advice = "持有观察"
            history.raw_result = json.dumps(
                {
                    "operation_advice": "持有观察",
                    "action": "watch",
                    "action_label": "观望",
                },
                ensure_ascii=False,
            )
            result = self._make_backtest_result(
                analysis_history_id=history.id,
                analysis_date=date(2024, 1, 1),
                eval_window_days=1,
            )
            result.operation_advice = "持有观察"
            result.position_recommendation = "long"
            session.add(result)
            session.commit()

        data = service.get_recent_evaluations(
            code="600519",
            eval_window_days=1,
            limit=10,
            page=1,
        )

        self.assertEqual(data["total"], 1)
        item = data["items"][0]
        self.assertEqual(item["operation_advice"], "持有观察")
        self.assertEqual(item["action"], "watch")
        self.assertEqual(item["action_label"], "观望")
        self.assertEqual(item["position_recommendation"], "long")

    def test_get_recent_evaluations_supports_tracking_fields_and_analysis_date_filters(self) -> None:
        self._seed_analysis(
            query_id="q2",
            analysis_date=date(2024, 1, 10),
            created_at=datetime(2024, 1, 10, 0, 0, 0),
            operation_advice="买入",
            trend_prediction="看多",
            start_close=100.0,
            forward_bars=[
                StockDaily(code="600519", date=date(2024, 1, 11), high=101.0, low=95.0, close=96.0),
            ],
        )

        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=1, min_age_days=0, limit=20)

        data = service.get_recent_evaluations(
            code="600519",
            eval_window_days=1,
            limit=10,
            page=1,
            analysis_date_from=date(2024, 1, 10),
            analysis_date_to=date(2024, 1, 10),
        )
        self.assertEqual(data["total"], 1)
        item = data["items"][0]
        self.assertEqual(item["stock_name"], "贵州茅台")
        self.assertEqual(item["trend_prediction"], "看多")
        self.assertEqual(item["actual_movement"], "down")
        self.assertAlmostEqual(item["actual_return_pct"], -4.0)
        self.assertFalse(item["direction_correct"])

    def test_get_summary_supports_analysis_date_range(self) -> None:
        self._seed_analysis(
            query_id="q2",
            analysis_date=date(2024, 1, 10),
            created_at=datetime(2024, 1, 10, 0, 0, 0),
            operation_advice="买入",
            trend_prediction="看多",
            start_close=100.0,
            forward_bars=[
                StockDaily(code="600519", date=date(2024, 1, 11), high=101.0, low=95.0, close=96.0),
            ],
        )

        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=1, min_age_days=0, limit=20)

        summary = service.get_summary(
            scope="stock",
            code="600519",
            eval_window_days=1,
            analysis_date_from=date(2024, 1, 10),
            analysis_date_to=date(2024, 1, 10),
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["total_evaluations"], 1)
        self.assertEqual(summary["completed_count"], 1)
        self.assertEqual(summary["win_count"], 0)
        self.assertEqual(summary["loss_count"], 1)
        self.assertAlmostEqual(summary["direction_accuracy_pct"], 0.0)

    def test_get_summary_date_range_filters_to_single_window_and_engine(self) -> None:
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        with self.db.get_session() as session:
            base_result = session.query(BacktestResult).filter(
                BacktestResult.code == "600519",
                BacktestResult.eval_window_days == 3,
                BacktestResult.engine_version == "v1",
            ).one()
            session.add_all([
                BacktestResult(
                    analysis_history_id=base_result.analysis_history_id,
                    code=base_result.code,
                    analysis_date=base_result.analysis_date,
                    eval_window_days=1,
                    engine_version="v1",
                    eval_status="completed",
                    evaluated_at=datetime(2024, 1, 5, 0, 0, 0),
                    operation_advice="买入",
                    position_recommendation="long",
                    start_price=100.0,
                    end_close=96.0,
                    stock_return_pct=-4.0,
                    direction_expected="up",
                    direction_correct=False,
                    outcome="loss",
                    simulated_return_pct=-4.0,
                ),
                BacktestResult(
                    analysis_history_id=base_result.analysis_history_id,
                    code=base_result.code,
                    analysis_date=base_result.analysis_date,
                    eval_window_days=3,
                    engine_version="v2",
                    eval_status="completed",
                    evaluated_at=datetime(2024, 1, 6, 0, 0, 0),
                    operation_advice="买入",
                    position_recommendation="long",
                    start_price=100.0,
                    end_close=96.0,
                    stock_return_pct=-4.0,
                    direction_expected="up",
                    direction_correct=False,
                    outcome="loss",
                    simulated_return_pct=-4.0,
                ),
            ])
            session.commit()

        rows = service.repo.list_results(
            code="600519",
            eval_window_days=3,
            engine_version="v1",
            analysis_date_from=date(2024, 1, 1),
            analysis_date_to=date(2024, 1, 1),
        )
        self.assertEqual(len(rows), 1)

    def _stub_bars(self, count: int) -> list[dict]:
        bars = []
        for idx in range(count):
            bars.append(
                {
                    "date": date(2024, 1, 1).replace(day=1),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1000.0,
                    "pct_chg": 0.0,
                    "pct_chg_calc": 0.0,
                    "ma5": 100.0,
                    "ma10": 100.0,
                    "ma20": 100.0,
                    "ma60": 100.0,
                    "ma120": 100.0,
                    "volume_ma20": 1000.0,
                    "rsi14": 50.0,
                    "low5": 95.0,
                    "close_up": True,
                    "atr14": 2.0,
                    "range_high100_prev": 105.0,
                    "range_low100_prev": 95.0,
                    "range_width100_pct": 10.53,
                    "candle_close_pos": 0.7,
                    "lower_shadow_ratio": 0.4,
                    "upper_shadow_ratio": 0.1,
                    "donchian_high20_prev": 105.0,
                    "donchian_low20_prev": 95.0,
                }
            )
            bars[-1]["date"] = date(2024, 1, 1).fromordinal(date(2024, 1, 1).toordinal() + idx)
        return bars

    def test_donchian_breakout_simulation_enters_and_exits(self) -> None:
        rows = []
        for idx in range(25):
            rows.append(
                {
                    "date": date(2024, 1, 1).fromordinal(date(2024, 1, 1).toordinal() + idx),
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.8,
                    "close": 10.0,
                    "volume": 1000.0,
                }
            )
        rows[20].update({"open": 10.1, "high": 11.3, "low": 10.0, "close": 11.0})
        rows[21].update({"open": 11.0, "high": 11.5, "low": 10.8, "close": 11.2})
        rows[22].update({"open": 11.1, "high": 11.2, "low": 8.7, "close": 8.9})

        bars = BacktestService._normalize_strategy_bars(rows)
        result = BacktestService._simulate_strategy_from_bars(
            bars=bars,
            strategy="donchian_breakout",
            initial_cash=100000,
            stop_loss_pct=8.0,
            take_profit_pct=20.0,
        )

        self.assertEqual(result["trade_count"], 1)
        self.assertEqual(result["trades"][0]["exit_reason"], "signal_exit")

    def test_rsi_composite_entry_and_exit_rules(self) -> None:
        bars = self._stub_bars(130)
        for idx, rsi in enumerate([48.0, 44.0, 32.0, 28.0, 24.0, 26.0, 31.0, 35.0, 37.0, 42.0], start=120):
            bars[idx]["rsi14"] = rsi
        bars[129].update(
            {
                "close": 98.0,
                "pct_chg": 2.1,
                "close_up": True,
                "low5": 95.5,
                "ma120": 120.0,
                "volume": 600.0,
                "volume_ma20": 1000.0,
            }
        )
        bars[128]["rsi14"] = 35.0

        self.assertTrue(BacktestService._strategy_entry_signal(bars, 129, "rsi_composite"))

        bars[129]["rsi14"] = 78.0
        exit_reason = BacktestService._strategy_custom_exit_reason(
            bars=bars,
            index=129,
            strategy="rsi_composite",
            entry_index=120,
            entry_price=90.0,
            peak_close=99.0,
        )
        self.assertEqual(exit_reason, "rsi_overheat")

    def test_wyckoff_enhanced_lps_entry_and_upthrust_exit(self) -> None:
        bars = self._stub_bars(130)
        bars[118].update(
            {
                "close": 108.0,
                "range_high100_prev": 105.0,
                "volume": 1200.0,
                "volume_ma20": 1000.0,
            }
        )
        bars[129].update(
            {
                "close": 104.0,
                "low": 103.0,
                "high": 105.0,
                "range_high100_prev": 105.0,
                "range_low100_prev": 95.0,
                "range_width100_pct": 10.53,
                "volume": 700.0,
                "volume_ma20": 1000.0,
            }
        )
        bars[128]["close"] = 103.5

        self.assertTrue(BacktestService._strategy_entry_signal(bars, 129, "wyckoff_enhanced"))

        bars[129].update({"high": 108.5, "close": 103.0, "upper_shadow_ratio": 0.45})
        exit_reason = BacktestService._strategy_custom_exit_reason(
            bars=bars,
            index=129,
            strategy="wyckoff_enhanced",
            entry_index=120,
            entry_price=101.0,
            peak_close=109.0,
        )
        self.assertEqual(exit_reason, "upthrust")

    def test_get_summary_date_range_rejects_excessive_row_counts(self) -> None:
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        with patch.object(BacktestService, "MAX_DYNAMIC_SUMMARY_ROWS", 0):
            with self.assertRaisesRegex(ValueError, "Date-filtered summary matches too many rows"):
                service.get_summary(
                    scope="stock",
                    code="600519",
                    analysis_date_from=date(2024, 1, 1),
                    analysis_date_to=date(2024, 1, 1),
                )

    def test_get_summary_phase_filter_cap_uses_phase_candidate_message(self) -> None:
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        with patch.object(BacktestService, "MAX_DYNAMIC_SUMMARY_ROWS", 0):
            with self.assertRaisesRegex(ValueError, "Phase-filtered summary candidate set matches too many rows"):
                service.get_summary(
                    scope="stock",
                    code="600519",
                    analysis_phase="intraday",
                )

    def test_phase_filter_results_allows_exact_dynamic_cap(self) -> None:
        service = BacktestService(self.db)
        phase_snapshot = json.dumps({"market_phase_summary": {"phase": "intraday", "market": "cn"}})
        raw_result = json.dumps(
            {"operation_advice": "持有观察", "action": "watch", "action_label": "观望"},
            ensure_ascii=False,
        )
        rows = [
            (
                self._make_backtest_result(analysis_history_id=idx + 1, analysis_date=date(2024, 1, idx + 1)),
                "贵州茅台",
                "看多",
                datetime(2024, 1, idx + 1, 0, 0, 0),
                phase_snapshot,
                raw_result,
                "simple",
            )
            for idx in range(2)
        ]

        class RepoStub:
            def get_results_with_context_batch(self, **kwargs):
                offset = int(kwargs["offset"])
                limit = int(kwargs["limit"])
                return rows[offset: offset + limit]

        service.repo = RepoStub()

        with patch.object(BacktestService, "MAX_DYNAMIC_SUMMARY_ROWS", 2):
            data = service.get_recent_evaluations(
                code="600519",
                eval_window_days=1,
                limit=10,
                page=1,
                analysis_phase="intraday",
            )

        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["items"]), 2)
        self.assertEqual(data["items"][0]["action"], "watch")
        self.assertEqual(data["items"][0]["action_label"], "观望")

    def test_phase_filter_without_window_matches_summary_window(self) -> None:
        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=3, min_age_days=0, limit=10)

        with self.db.get_session() as session:
            base_result = session.query(BacktestResult).filter(
                BacktestResult.code == "600519",
                BacktestResult.eval_window_days == 3,
                BacktestResult.engine_version == "v1",
            ).one()
            session.add(
                BacktestResult(
                    analysis_history_id=base_result.analysis_history_id,
                    code=base_result.code,
                    analysis_date=base_result.analysis_date,
                    eval_window_days=1,
                    engine_version="v1",
                    eval_status="completed",
                    evaluated_at=datetime(2024, 1, 5, 0, 0, 0),
                    operation_advice="买入",
                    position_recommendation="long",
                    start_price=100.0,
                    end_close=96.0,
                    stock_return_pct=-4.0,
                    direction_expected="up",
                    direction_correct=False,
                    outcome="loss",
                    simulated_return_pct=-4.0,
                )
            )
            session.commit()

        evaluations = service.get_recent_evaluations(
            code="600519",
            limit=10,
            page=1,
            analysis_phase="premarket",
        )
        self.assertEqual(evaluations["total"], 1)
        self.assertEqual(evaluations["items"][0]["eval_window_days"], 1)

        summary = service.get_summary(
            scope="stock",
            code="600519",
            analysis_phase="premarket",
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["eval_window_days"], 1)
        self.assertEqual(summary["total_evaluations"], 1)

    def test_phase_filter_overfetches_before_pagination_and_updates_summary_breakdown(self) -> None:
        self._seed_analysis(
            query_id="q2",
            analysis_date=date(2024, 1, 10),
            created_at=datetime(2024, 1, 10, 0, 0, 0),
            operation_advice="买入",
            trend_prediction="看多",
            start_close=100.0,
            forward_bars=[
                StockDaily(code="600519", date=date(2024, 1, 11), high=101.0, low=95.0, close=96.0),
            ],
            phase="intraday",
        )

        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=1, min_age_days=0, limit=20)

        intraday = service.get_recent_evaluations(
            code="600519",
            eval_window_days=1,
            limit=1,
            page=1,
            analysis_phase="intraday",
        )
        self.assertEqual(intraday["total"], 1)
        self.assertEqual(len(intraday["items"]), 1)
        self.assertEqual(intraday["items"][0]["market_phase"], "intraday")
        self.assertEqual(intraday["items"][0]["market_phase_summary"]["phase"], "intraday")

        premarket = service.get_recent_evaluations(
            code="600519",
            eval_window_days=1,
            limit=1,
            page=1,
            analysis_phase="premarket",
        )
        self.assertEqual(premarket["total"], 1)
        self.assertEqual(premarket["items"][0]["market_phase"], "premarket")

        summary = service.get_summary(
            scope="stock",
            code="600519",
            eval_window_days=1,
            analysis_phase="intraday",
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["total_evaluations"], 1)
        self.assertEqual(summary["diagnostics"]["phase_breakdown"]["intraday"], 1)
        self.assertEqual(summary["diagnostics"]["phase_breakdown"]["premarket"], 0)
        self.assertNotIn("premarket", summary["diagnostics"]["raw_phase_counts"])
        self.assertEqual(summary["diagnostics"]["raw_phase_counts"]["intraday"], 1)

    def test_phase_filter_buckets_detailed_internal_phases(self) -> None:
        self._seed_analysis(
            query_id="q2",
            analysis_date=date(2024, 1, 10),
            created_at=datetime(2024, 1, 10, 0, 0, 0),
            operation_advice="买入",
            trend_prediction="看多",
            start_close=100.0,
            forward_bars=[
                StockDaily(code="600519", date=date(2024, 1, 11), high=101.0, low=95.0, close=96.0),
            ],
            phase="lunch_break",
        )
        self._seed_analysis(
            query_id="q3",
            analysis_date=date(2024, 1, 12),
            created_at=datetime(2024, 1, 12, 0, 0, 0),
            operation_advice="买入",
            trend_prediction="看多",
            start_close=100.0,
            forward_bars=[
                StockDaily(code="600519", date=date(2024, 1, 13), high=101.0, low=95.0, close=96.0),
            ],
            phase="closing_auction",
        )
        self._seed_analysis(
            query_id="q4",
            analysis_date=date(2024, 1, 14),
            created_at=datetime(2024, 1, 14, 0, 0, 0),
            operation_advice="买入",
            trend_prediction="看多",
            start_close=100.0,
            forward_bars=[
                StockDaily(code="600519", date=date(2024, 1, 15), high=101.0, low=95.0, close=96.0),
            ],
            phase="non_trading",
        )

        service = BacktestService(self.db)
        service.run_backtest(code="600519", force=False, eval_window_days=1, min_age_days=0, limit=20)

        intraday = service.get_recent_evaluations(
            code="600519",
            eval_window_days=1,
            limit=10,
            page=1,
            analysis_phase="intraday",
        )
        self.assertEqual(intraday["total"], 2)
        self.assertEqual(
            {item["market_phase_summary"]["phase"] for item in intraday["items"]},
            {"lunch_break", "closing_auction"},
        )
        self.assertTrue(all(item["market_phase"] == "intraday" for item in intraday["items"]))

        unknown = service.get_recent_evaluations(
            code="600519",
            eval_window_days=1,
            limit=10,
            page=1,
            analysis_phase="unknown",
        )
        self.assertEqual(unknown["total"], 1)
        self.assertEqual(unknown["items"][0]["market_phase"], "unknown")
        self.assertEqual(unknown["items"][0]["market_phase_summary"]["phase"], "non_trading")

    def test_phase_filter_rejects_values_outside_public_query_contract(self) -> None:
        service = BacktestService(self.db)

        with self.assertRaisesRegex(ValueError, "analysis_phase must be one of"):
            service.get_recent_evaluations(code=None, analysis_phase="lunch_break")

        with self.assertRaisesRegex(ValueError, "analysis_phase must be one of"):
            service.get_summary(code=None, scope="overall", analysis_phase="banana")

    def test_multi_stock_summaries(self) -> None:
        """Verify separate summaries for multiple stocks + correct overall aggregate."""
        old_created_at = datetime(2024, 1, 1, 0, 0, 0)

        with self.db.get_session() as session:
            # Second stock with sell advice -- price drops (win for cash/down)
            session.add(
                AnalysisHistory(
                    query_id="q2",
                    code="000001",
                    name="平安银行",
                    report_type="simple",
                    sentiment_score=30,
                    operation_advice="卖出",
                    trend_prediction="看空",
                    analysis_summary="test2",
                    stop_loss=None,
                    take_profit=None,
                    created_at=old_created_at,
                    context_snapshot='{"enhanced_context": {"date": "2024-01-01"}}',
                )
            )
            session.add(
                StockDaily(code="000001", date=date(2024, 1, 1), open=10.0, high=10.2, low=9.8, close=10.0)
            )
            session.add_all([
                StockDaily(code="000001", date=date(2024, 1, 2), high=10.0, low=9.5, close=9.6),
                StockDaily(code="000001", date=date(2024, 1, 3), high=9.7, low=9.3, close=9.4),
                StockDaily(code="000001", date=date(2024, 1, 4), high=9.5, low=9.0, close=9.1),
            ])
            session.commit()

        service = BacktestService(self.db)
        stats = service.run_backtest(code=None, force=False, eval_window_days=3, min_age_days=0, limit=10)
        self.assertEqual(stats["saved"], 2)
        self.assertEqual(stats["completed"], 2)

        with self.db.get_session() as session:
            # Each stock has its own summary
            s1 = session.query(BacktestSummary).filter(
                BacktestSummary.scope == "stock", BacktestSummary.code == "600519"
            ).first()
            s2 = session.query(BacktestSummary).filter(
                BacktestSummary.scope == "stock", BacktestSummary.code == "000001"
            ).first()
            self.assertIsNotNone(s1)
            self.assertIsNotNone(s2)
            self.assertEqual(s1.win_count, 1)
            self.assertEqual(s2.win_count, 1)

            # Overall aggregates both
            overall = session.query(BacktestSummary).filter(
                BacktestSummary.scope == "overall",
                BacktestSummary.code == OVERALL_SENTINEL_CODE,
            ).first()
            self.assertIsNotNone(overall)
            self.assertEqual(overall.total_evaluations, 2)
            self.assertEqual(overall.completed_count, 2)
            self.assertEqual(overall.win_count, 2)

    def test_run_backtest_excludes_market_review_records(self) -> None:
        with self.db.get_session() as session:
            session.add(
                AnalysisHistory(
                    query_id="q-market-review",
                    code="MARKET",
                    name="大盘复盘",
                    report_type="market_review",
                    sentiment_score=50,
                    operation_advice="查看复盘",
                    trend_prediction="大盘复盘",
                    analysis_summary="market review summary",
                    stop_loss=None,
                    take_profit=None,
                    created_at=datetime(2024, 1, 3, 0, 0, 0),
                    context_snapshot='{"enhanced_context": {"date": "2024-01-03"}}',
                )
            )
            session.commit()

        service = BacktestService(self.db)
        stats = service.run_backtest(code=None, force=False, eval_window_days=3, min_age_days=0, limit=10)

        self.assertEqual(stats["processed"], 1)
        self.assertEqual(stats["saved"], 1)
        self.assertEqual(self._count_results(), 1)
        with self.db.get_session() as session:
            self.assertEqual(
                session.query(BacktestResult).filter(BacktestResult.code == "MARKET").count(),
                0,
            )

    def test_run_backtest_includes_null_report_type_records(self) -> None:
        with self.db.get_session() as session:
            session.add(
                AnalysisHistory(
                    query_id="q-null-report-type",
                    code="000858",
                    name="五粮液",
                    report_type=None,
                    sentiment_score=60,
                    operation_advice="持有",
                    trend_prediction="震荡",
                    analysis_summary="legacy null report_type row",
                    stop_loss=None,
                    take_profit=None,
                    created_at=datetime(2024, 1, 3, 0, 0, 0),
                    context_snapshot='{"enhanced_context": {"date": "2024-01-03"}}',
                )
            )
            session.add_all(
                [
                    StockDaily(code="000858", date=date(2024, 1, 3), open=12.0, high=12.8, low=11.5, close=12.2),
                    StockDaily(code="000858", date=date(2024, 1, 4), open=12.2, high=13.0, low=12.0, close=12.6),
                    StockDaily(code="000858", date=date(2024, 1, 5), open=12.6, high=12.9, low=11.9, close=12.4),
                ]
            )
            session.commit()

        service = BacktestService(self.db)
        stats = service.run_backtest(code=None, force=False, eval_window_days=3, min_age_days=0, limit=10)
        self.assertGreaterEqual(stats["processed"], 2)
        self.assertGreaterEqual(stats["saved"], 2)
        with self.db.get_session() as session:
            self.assertEqual(
                session.query(BacktestResult).filter(BacktestResult.code == "000858").count(),
                1,
            )


if __name__ == "__main__":
    unittest.main()
