"""
Backtest with $300 initial capital — matches mainnet live config.
4 coins: ETH, XRP, DOGE, SOL. Validation period.

Usage: uv run backtest_300usd_live.py [--intra-bar]
"""
import sys
import time
import prepare

# Override initial capital to match live account
prepare.INITIAL_CAPITAL = 300.0

from prepare import load_data, run_backtest, compute_score

INTRA_BAR = "--intra-bar" in sys.argv
COINS = ["ETH", "XRP", "DOGE", "SOL"]

from strategy import Strategy

strategy = Strategy(symbols=COINS)
data = load_data("val", symbols=COINS)

print(f"Capital: $300 | Coins: {COINS}")
print(f"Loaded {sum(len(df) for df in data.values())} bars across {len(data)} symbols")
if INTRA_BAR:
    print("Intra-bar simulation: ENABLED")

t_start = time.time()
result = run_backtest(strategy, data, intra_bar_sim=INTRA_BAR)
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
print(f"final_equity:       ${result.total_return_pct / 100 * 300 + 300:.2f}")
print(f"backtest_seconds:   {result.backtest_seconds:.1f}")
