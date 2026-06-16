# -*- coding: utf-8 -*-
"""Backtest API schemas."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from api.v1.schemas.market_phase import MarketPhaseSummary
from src.schemas.decision_action import DecisionAction


class BacktestRunRequest(BaseModel):
    code: Optional[str] = Field(None, description="仅回测指定股票")
    force: bool = Field(False, description="强制重新计算")
    eval_window_days: Optional[int] = Field(None, ge=1, le=120, description="评估窗口（交易日数）")
    min_age_days: Optional[int] = Field(None, ge=0, le=365, description="分析记录最小天龄（0=不限）")
    limit: int = Field(200, ge=1, le=2000, description="最多处理的分析记录数")


class StrategySimulationRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=16, description="股票代码")
    strategy: str = Field("momentum_quality", min_length=1, max_length=64, description="历史模拟策略")
    start_date: Optional[date] = Field(None, description="开始日期")
    end_date: Optional[date] = Field(None, description="结束日期")
    initial_cash: float = Field(100000.0, gt=0, description="初始资金")
    stop_loss_pct: float = Field(8.0, ge=0, le=50, description="单笔止损百分比")
    take_profit_pct: float = Field(20.0, ge=0, le=200, description="单笔止盈百分比")


class BacktestRunResponse(BaseModel):
    processed: int = Field(..., description="候选记录数")
    saved: int = Field(..., description="写入回测结果数")
    completed: int = Field(..., description="完成回测数")
    insufficient: int = Field(..., description="数据不足数")
    errors: int = Field(..., description="错误数")


class StrategyTradeItem(BaseModel):
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    exit_reason: str


class StrategySimulationResponse(BaseModel):
    code: str
    strategy: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_start_date: Optional[str] = None
    data_end_date: Optional[str] = None
    bars_count: int = 0
    data_source: str = ""
    trades: List[StrategyTradeItem] = Field(default_factory=list)
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    win_rate_pct: Optional[float] = None
    total_return_pct: Optional[float] = None
    buy_hold_return_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    final_equity: float
    message: str = ""


class BacktestResultItem(BaseModel):
    analysis_history_id: int
    code: str
    stock_name: Optional[str] = None
    analysis_date: Optional[str] = None
    eval_window_days: int
    engine_version: str
    eval_status: str
    evaluated_at: Optional[str] = None
    operation_advice: Optional[str] = None
    action: Optional[DecisionAction] = None
    action_label: Optional[str] = None
    trend_prediction: Optional[str] = None
    market_phase: Optional[str] = None
    market_phase_summary: Optional[MarketPhaseSummary] = None
    position_recommendation: Optional[str] = None
    start_price: Optional[float] = None
    end_close: Optional[float] = None
    max_high: Optional[float] = None
    min_low: Optional[float] = None
    stock_return_pct: Optional[float] = None
    actual_return_pct: Optional[float] = None
    actual_movement: Optional[str] = None
    direction_expected: Optional[str] = None
    direction_correct: Optional[bool] = None
    outcome: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    hit_stop_loss: Optional[bool] = None
    hit_take_profit: Optional[bool] = None
    first_hit: Optional[str] = None
    first_hit_date: Optional[str] = None
    first_hit_trading_days: Optional[int] = None
    simulated_entry_price: Optional[float] = None
    simulated_exit_price: Optional[float] = None
    simulated_exit_reason: Optional[str] = None
    simulated_return_pct: Optional[float] = None


class BacktestResultsResponse(BaseModel):
    total: int
    page: int
    limit: int
    items: List[BacktestResultItem] = Field(default_factory=list)


class PerformanceMetrics(BaseModel):
    scope: str
    code: Optional[str] = None
    eval_window_days: int
    engine_version: str
    computed_at: Optional[str] = None

    total_evaluations: int
    completed_count: int
    insufficient_count: int
    long_count: int
    cash_count: int
    win_count: int
    loss_count: int
    neutral_count: int

    direction_accuracy_pct: Optional[float] = None
    win_rate_pct: Optional[float] = None
    neutral_rate_pct: Optional[float] = None
    avg_stock_return_pct: Optional[float] = None
    avg_simulated_return_pct: Optional[float] = None

    stop_loss_trigger_rate: Optional[float] = None
    take_profit_trigger_rate: Optional[float] = None
    ambiguous_rate: Optional[float] = None
    avg_days_to_first_hit: Optional[float] = None

    advice_breakdown: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
