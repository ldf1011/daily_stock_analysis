#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark AlphaSift-style strategy simulations on a random A-share sample.

The benchmark uses real daily bars through the project's data layer, then runs
the same buy/sell simulation rules for each strategy so the ranking is
comparable and reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider.baostock_fetcher import BaostockFetcher  # noqa: E402
from data_provider.base import DataFetcherManager, normalize_stock_code  # noqa: E402
from src.services.backtest_service import BacktestService  # noqa: E402
from src.storage import DatabaseManager  # noqa: E402


STRATEGIES = [
    ("balanced_alpha", "均衡多因子"),
    ("capital_heat", "资金热度"),
    ("donchian_breakout", "唐奇安通道突破"),
    ("dual_low", "双低选股"),
    ("momentum_quality", "趋势质量"),
    ("oversold_reversal", "超跌反转"),
    ("quality_value", "稳健价值"),
    ("rsi_composite", "RSI综合优选"),
    ("shrink_pullback", "缩量回踩"),
    ("volume_breakout", "放量突破"),
    ("wyckoff_enhanced", "威科夫增强"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机抽样 88 只股票评测策略强弱")
    parser.add_argument("--sample-size", type=int, default=88, help="最终有效样本数量")
    parser.add_argument("--seed", type=int, default=20260613, help="随机种子")
    parser.add_argument("--months", type=int, default=12, help="评测窗口月份数")
    parser.add_argument("--min-bars", type=int, default=80, help="纳入样本的最低日线数量")
    parser.add_argument("--max-attempts", type=int, default=360, help="最多尝试拉取股票数量")
    parser.add_argument("--output", default="reports/strategy_ranking_88.json", help="输出 JSON 报告路径")
    parser.add_argument("--no-fetch", action="store_true", help="只使用本地已有数据，不在线补数据")
    parser.add_argument(
        "--fetcher",
        choices=("baostock", "manager"),
        default="baostock",
        help="在线补数据来源；baostock 更适合批量，manager 会走全量数据源兜底。",
    )
    return parser.parse_args()


def load_a_share_universe() -> List[Dict[str, Any]]:
    index_path = ROOT / "static" / "stocks.index.json"
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    universe: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 8:
            continue
        full_code, code, name, *_rest = row
        market = row[6]
        kind = row[7]
        if market != "CN" or kind != "stock":
            continue
        name_text = str(name or "")
        code_text = str(code or "")
        if not (code_text.isdigit() and len(code_text) == 6):
            continue
        if any(marker in name_text.upper() for marker in ("ST", "退", "PT")):
            continue
        # Keep common A-share boards; avoid indices/funds and unusual 9xxxxx codes.
        if not code_text.startswith(("0", "3", "6")):
            continue
        universe.append({"code": str(full_code), "plain_code": code_text, "name": name_text})
    return universe


def coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        result = float(value)
        if result != result:
            return None
        return result
    except Exception:
        return None


def suffix_code(code: str) -> str:
    plain = normalize_stock_code(code)
    if plain.isdigit() and len(plain) == 6:
        if plain.startswith("6"):
            return f"{plain}.SH"
        if plain.startswith(("0", "3")):
            return f"{plain}.SZ"
    return code


def local_bar_count(service: BacktestService, code: str, start: date, end: date) -> int:
    candidates = service._strategy_code_candidates(suffix_code(code), normalize_stock_code(code))
    best_count = 0
    for candidate in candidates:
        rows = service.stock_repo.get_range(candidate, start, end)
        best_count = max(best_count, len(rows))
    return best_count


def ensure_daily_data(
    *,
    service: BacktestService,
    manager: DataFetcherManager,
    stock: Dict[str, Any],
    start: date,
    end: date,
    min_bars: int,
    no_fetch: bool,
) -> Tuple[bool, str, int]:
    code = stock["code"]
    plain = stock["plain_code"]
    count = local_bar_count(service, code, start, end)
    if count >= min_bars:
        return True, "db_cache", count
    if no_fetch:
        return False, "local_insufficient", count
    try:
        df, source = manager.get_daily_data(
            plain,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            days=max(260, (end - start).days + 80),
        )
        if df is None or df.empty:
            return False, "fetch_empty", count
        for candidate in {plain, suffix_code(plain), suffix_code(code)}:
            service.db.save_daily_data(df, candidate, source)
        count = local_bar_count(service, code, start, end)
        return count >= min_bars, source, count
    except Exception as exc:
        return False, f"fetch_failed: {type(exc).__name__}: {str(exc)[:140]}", count


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * pct))
    return sorted_values[index]


def avg(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def aggregate_strategy(strategy_id: str, label: str, results: List[Dict[str, Any]], sample_size: int) -> Dict[str, Any]:
    returns = [coerce_float(item.get("total_return_pct")) for item in results]
    returns_clean = [value for value in returns if value is not None]
    buy_hold = [coerce_float(item.get("buy_hold_return_pct")) for item in results]
    drawdowns = [coerce_float(item.get("max_drawdown_pct")) for item in results]
    trade_count = sum(int(item.get("trade_count") or 0) for item in results)
    selected_count = sum(1 for item in results if int(item.get("trade_count") or 0) > 0)
    win_count = sum(int(item.get("win_count") or 0) for item in results)
    loss_count = sum(int(item.get("loss_count") or 0) for item in results)
    avg_return = avg(returns_clean)
    avg_buy_hold = avg(buy_hold)
    avg_excess = (avg_return - avg_buy_hold) if avg_return is not None and avg_buy_hold is not None else None
    win_rate = (win_count / trade_count * 100) if trade_count else 0.0
    coverage = selected_count / sample_size * 100 if sample_size else 0.0
    avg_drawdown = avg(drawdowns) or 0.0
    # Balanced score: excess/return matter, but penalize inactivity and drawdown.
    score = (
        (avg_return or 0.0) * 1.4
        + (avg_excess or 0.0) * 1.8
        + win_rate * 0.25
        + coverage * 0.12
        - avg_drawdown * 0.7
    )
    return {
        "strategy": strategy_id,
        "name": label,
        "score": round(score, 2),
        "tested_stocks": sample_size,
        "selected_stocks": selected_count,
        "selection_rate_pct": round(coverage, 2),
        "trade_count": trade_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "trade_win_rate_pct": round(win_rate, 2),
        "avg_return_pct": round(avg_return, 2) if avg_return is not None else None,
        "median_return_pct": round(statistics.median(returns_clean), 2) if returns_clean else None,
        "p25_return_pct": round(percentile(returns_clean, 0.25), 2) if returns_clean else None,
        "p75_return_pct": round(percentile(returns_clean, 0.75), 2) if returns_clean else None,
        "avg_buy_hold_pct": round(avg_buy_hold, 2) if avg_buy_hold is not None else None,
        "avg_excess_pct": round(avg_excess, 2) if avg_excess is not None else None,
        "avg_max_drawdown_pct": round(avg_drawdown, 2),
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    end = date.today()
    start = end - timedelta(days=int(args.months * 30.5))

    universe = load_a_share_universe()
    random.shuffle(universe)

    db = DatabaseManager.get_instance()
    service = BacktestService(db)
    manager = DataFetcherManager(fetchers=[BaostockFetcher()]) if args.fetcher == "baostock" else DataFetcherManager()
    selected: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    print(f"目标: 随机有效样本 {args.sample_size} 只, 窗口 {start} ~ {end}, 最少K线 {args.min_bars}")
    for attempt, stock in enumerate(universe[: args.max_attempts], start=1):
        ok, source, bars = ensure_daily_data(
            service=service,
            manager=manager,
            stock=stock,
            start=start,
            end=end,
            min_bars=args.min_bars,
            no_fetch=args.no_fetch,
        )
        if ok:
            selected.append({**stock, "data_source": source, "bars": bars})
            print(f"[{len(selected):02d}/{args.sample_size}] OK {stock['code']} {stock['name']} bars={bars} source={source}")
        else:
            failures.append({**stock, "reason": source, "bars": bars})
            print(f"[skip {attempt}] {stock['code']} {stock['name']} bars={bars} reason={source}")
        if len(selected) >= args.sample_size:
            break

    if len(selected) < args.sample_size:
        print(f"有效样本不足: {len(selected)}/{args.sample_size}", file=sys.stderr)

    per_strategy_results: Dict[str, List[Dict[str, Any]]] = {strategy_id: [] for strategy_id, _ in STRATEGIES}
    stock_summaries: List[Dict[str, Any]] = []
    for stock_index, stock in enumerate(selected, start=1):
        stock_summary = {
            "code": stock["code"],
            "name": stock["name"],
            "bars": stock["bars"],
            "data_source": stock["data_source"],
            "strategies": {},
        }
        for strategy_id, label in STRATEGIES:
            result = service.simulate_strategy(
                code=stock["code"],
                strategy=strategy_id,
                start_date=start,
                end_date=end,
                initial_cash=100000,
            )
            slim = {
                "code": stock["code"],
                "name": stock["name"],
                "strategy": strategy_id,
                "strategy_name": label,
                "bars_count": result.get("bars_count"),
                "trade_count": result.get("trade_count"),
                "win_count": result.get("win_count"),
                "loss_count": result.get("loss_count"),
                "win_rate_pct": result.get("win_rate_pct"),
                "total_return_pct": result.get("total_return_pct"),
                "buy_hold_return_pct": result.get("buy_hold_return_pct"),
                "max_drawdown_pct": result.get("max_drawdown_pct"),
                "final_equity": result.get("final_equity"),
                "trades": result.get("trades") or [],
            }
            per_strategy_results[strategy_id].append(slim)
            stock_summary["strategies"][strategy_id] = {
                "name": label,
                "trade_count": slim["trade_count"],
                "total_return_pct": slim["total_return_pct"],
                "win_rate_pct": slim["win_rate_pct"],
                "max_drawdown_pct": slim["max_drawdown_pct"],
                "trades": slim["trades"],
            }
        stock_summaries.append(stock_summary)
        print(f"[simulate {stock_index:02d}/{len(selected)}] {stock['code']} {stock['name']} done")

    ranking = [
        aggregate_strategy(strategy_id, label, per_strategy_results[strategy_id], len(selected))
        for strategy_id, label in STRATEGIES
    ]
    ranking.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "requested_sample_size": args.sample_size,
        "effective_sample_size": len(selected),
        "period": {"start_date": start.isoformat(), "end_date": end.isoformat(), "months": args.months},
        "method": {
            "sample": "static/stocks.index.json 随机抽取 CN 普通股票，排除 ST/退市/PT，仅保留 0/3/6 开头 A 股。",
            "selection": "策略在窗口内至少触发 1 次买入视为选股命中；交易明细来自策略入场、止损、止盈、信号退出或区间结束。",
            "score": "score = avg_return*1.4 + avg_excess*1.8 + trade_win_rate*0.25 + selection_rate*0.12 - avg_drawdown*0.7",
        },
        "ranking": ranking,
        "selected_stocks": selected,
        "failures_count": len(failures),
        "failures_sample": failures[:50],
        "per_strategy_results": per_strategy_results,
        "stock_summaries": stock_summaries,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n综合排序：")
    for item in ranking:
        print(
            f"{item['rank']}. {item['name']} ({item['strategy']}) "
            f"score={item['score']} avg={item['avg_return_pct']}% excess={item['avg_excess_pct']}% "
            f"win={item['trade_win_rate_pct']}% select={item['selection_rate_pct']}% trades={item['trade_count']} "
            f"dd={item['avg_max_drawdown_pct']}%"
        )
    print(f"\n报告已写入: {output_path}")
    return 0 if len(selected) >= args.sample_size else 2


if __name__ == "__main__":
    raise SystemExit(main())
