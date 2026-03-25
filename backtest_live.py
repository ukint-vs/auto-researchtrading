"""
Live-realistic backtest (Gate 2). Same strategy, harsher execution model.
Uses REALISTIC_MODE in prepare.py: next-bar-open execution, re-entry slippage
penalty, market impact, partial fills, connection gaps, higher fees.

The ratio gate2/gate1 measures how much edge survives live friction.
Target ratio: >= 0.50. Below that, the strategy is a backtest artifact.

Usage: uv run backtest_live.py
"""
import os
import time
import signal as sig

# Activate realistic mode BEFORE importing prepare
os.environ["REALISTIC_BACKTEST"] = "1"

from prepare import load_data, run_backtest, compute_score, TIME_BUDGET, get_symbols, DEFAULT_SYMBOLS

# Timeout guard
def timeout_handler(signum, frame):
    print("TIMEOUT: backtest exceeded time budget")
    exit(1)
sig.signal(sig.SIGALRM, timeout_handler)
sig.alarm(TIME_BUDGET + 30)

t_start = time.time()
from strategy import Strategy
strategy = Strategy()
data = load_data("val")

print(f"Loaded {sum(len(df) for df in data.values())} bars across {len(data)} symbols")
print(f"Symbols: {list(data.keys())}")
print(f"=== GATE 2: LIVE-REALISTIC ===")
print(f"  Next-bar-open execution: ON")
print(f"  Re-entry slippage penalty: ON")
print(f"  Market impact: ON")
print(f"  Partial fills: ON")
print(f"  Connection gaps: ON")

result = run_backtest(strategy, data)
score = compute_score(result)
t_end = time.time()

print("---")
print(f"score:              {score:.6f}")
print(f"sharpe:             {result.sharpe:.6f}")
print(f"total_return_pct:   {result.total_return_pct:.6f}")
print(f"max_drawdown_pct:   {result.max_drawdown_pct:.6f}")
print(f"num_trades:         {result.num_trades}")
print(f"win_rate_pct:       {result.win_rate_pct:.6f}")
print(f"profit_factor:      {result.profit_factor:.6f}")
print(f"annual_turnover:    {result.annual_turnover:.2f}")
print(f"backtest_seconds:   {result.backtest_seconds:.1f}")
print(f"total_seconds:      {t_end - t_start:.1f}")

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
