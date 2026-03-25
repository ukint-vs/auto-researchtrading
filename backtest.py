"""
Run backtest with mandatory dual-gate evaluation.
Gate 1: standard engine. Gate 2: realistic engine (next-bar execution, re-entry
penalty, market impact, partial fills, connection gaps, higher fees).

The score printed is gate1. Gate2 and the ratio are also printed.
If gate2/gate1 ratio < 0.50, score is replaced with -888 (veto).

Usage: uv run backtest.py
"""

import os
import sys
import time
import signal as sig

from prepare import load_data, run_backtest, compute_score, TIME_BUDGET, DEFAULT_SYMBOLS, get_symbols

# Timeout guard
def timeout_handler(signum, frame):
    print("TIMEOUT: backtest exceeded time budget")
    exit(1)

sig.signal(sig.SIGALRM, timeout_handler)
sig.alarm(TIME_BUDGET * 2 + 60)  # budget for both gates

t_start = time.time()

from strategy import Strategy

data = load_data("val")

print(f"Loaded {sum(len(df) for df in data.values())} bars across {len(data)} symbols")
print(f"Symbols: {list(data.keys())}")

# ── Gate 1: standard engine ──
strategy1 = Strategy()
result1 = run_backtest(strategy1, data)
score1 = compute_score(result1)

# ── Gate 2: realistic engine ──
os.environ["REALISTIC_BACKTEST"] = "1"
# Re-import prepare to pick up the env var change
import importlib
import prepare
importlib.reload(prepare)
from prepare import run_backtest as run_backtest_real, compute_score as compute_score_real

strategy2 = Strategy()
result2 = run_backtest_real(strategy2, data)
score2 = compute_score_real(result2)

# Reset realistic mode
os.environ.pop("REALISTIC_BACKTEST", None)

t_end = time.time()

# ── Ratio and veto ──
ratio = score2 / score1 if score1 > 0 else 0.0
vetoed = ratio < 0.50 and score1 > 0

effective_score = -888.0 if vetoed else score1

# ── Output ──
print("---")
print(f"score:              {effective_score:.6f}")
print(f"sharpe:             {result1.sharpe:.6f}")
print(f"total_return_pct:   {result1.total_return_pct:.6f}")
print(f"max_drawdown_pct:   {result1.max_drawdown_pct:.6f}")
print(f"num_trades:         {result1.num_trades}")
print(f"win_rate_pct:       {result1.win_rate_pct:.6f}")
print(f"profit_factor:      {result1.profit_factor:.6f}")
print(f"annual_turnover:    {result1.annual_turnover:.2f}")
print(f"backtest_seconds:   {result1.backtest_seconds:.1f}")
print(f"total_seconds:      {t_end - t_start:.1f}")

print(f"\n--- Gate 2: realistic ---")
print(f"gate2_score:        {score2:.6f}")
print(f"gate2_sharpe:       {result2.sharpe:.6f}")
print(f"gate2_return:       {result2.total_return_pct:.6f}")
print(f"gate2_trades:       {result2.num_trades}")
print(f"gate2_max_dd:       {result2.max_drawdown_pct:.6f}")
print(f"ratio:              {ratio:.4f}")
if vetoed:
    print(f"VETOED:             ratio {ratio:.2f} < 0.50 — score forced to -888")

# Dual scoreboard: legacy 7-coin reference score
sig.alarm(0)
training_symbols = get_symbols("training")
if set(training_symbols) != set(DEFAULT_SYMBOLS):
    legacy_data = load_data("val", symbols=DEFAULT_SYMBOLS)
    if legacy_data:
        legacy_strat = Strategy(symbols=list(DEFAULT_SYMBOLS))
        legacy_result = run_backtest(legacy_strat, legacy_data)
        legacy_score = compute_score(legacy_result)
        print(f"\n--- Legacy 7-coin reference ---")
        print(f"legacy_score:       {legacy_score:.6f}")
        print(f"legacy_sharpe:      {legacy_result.sharpe:.6f}")
