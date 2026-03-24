"""
Run backtest. Usage: uv run backtest.py
Imports strategy from strategy.py, runs on validation data, prints metrics.
This file is fixed — do not modify.
"""

import time
import signal as sig

from prepare import load_data, run_backtest, compute_score, TIME_BUDGET, DEFAULT_SYMBOLS, get_symbols

# Timeout guard
def timeout_handler(signum, frame):
    print("TIMEOUT: backtest exceeded time budget")
    exit(1)

sig.signal(sig.SIGALRM, timeout_handler)
sig.alarm(TIME_BUDGET + 30)  # 30s grace for startup

t_start = time.time()

from strategy import Strategy

strategy = Strategy()
data = load_data("val")

print(f"Loaded {sum(len(df) for df in data.values())} bars across {len(data)} symbols")
print(f"Symbols: {list(data.keys())}")

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

# Dual scoreboard: legacy 7-coin reference score (cancel alarm to avoid timeout)
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
