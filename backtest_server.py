"""
Resident backtest evaluator. Keeps data hot in memory, reloads strategy on each run.
Eliminates ~2-3s of startup/import/parquet-load overhead per autoresearch iteration.

Usage:
    uv run backtest_server.py          # Interactive: type 'run' + Enter to backtest
    echo run | uv run backtest_server.py   # Single-shot mode
"""

import sys
import time
import json
import importlib
import signal as sig

from prepare import load_data, run_backtest, compute_score, TIME_BUDGET

# Load data ONCE on startup
t_load = time.time()
data = load_data("val")
load_time = time.time() - t_load

total_bars = sum(len(df) for df in data.values())
print(f"Loaded {total_bars} bars across {len(data)} symbols in {load_time:.1f}s", file=sys.stderr)
print(f"Symbols: {list(data.keys())}", file=sys.stderr)
print("Ready. Send 'run' to backtest, 'quit' to exit.", file=sys.stderr)
sys.stderr.flush()

for line in sys.stdin:
    cmd = line.strip().lower()
    if cmd == "quit" or cmd == "exit":
        break
    if cmd != "run":
        continue

    # Reload strategy module (picks up autoresearch edits to strategy.py)
    if "strategy" in sys.modules:
        del sys.modules["strategy"]
    try:
        from strategy import Strategy
    except Exception as e:
        print(json.dumps({"error": str(e), "score": -999}))
        sys.stdout.flush()
        continue

    # Set timeout
    def timeout_handler(signum, frame):
        raise TimeoutError("backtest exceeded time budget")

    sig.signal(sig.SIGALRM, timeout_handler)
    sig.alarm(TIME_BUDGET + 10)

    try:
        t_start = time.time()
        strategy = Strategy()
        result = run_backtest(strategy, data)
        score = compute_score(result)
        elapsed = time.time() - t_start

        print(json.dumps({
            "score": round(score, 6),
            "sharpe": round(result.sharpe, 6),
            "total_return_pct": round(result.total_return_pct, 6),
            "max_drawdown_pct": round(result.max_drawdown_pct, 6),
            "num_trades": result.num_trades,
            "win_rate_pct": round(result.win_rate_pct, 6),
            "profit_factor": round(result.profit_factor, 6),
            "annual_turnover": round(result.annual_turnover, 2),
            "backtest_seconds": round(result.backtest_seconds, 1),
            "total_seconds": round(elapsed, 1),
        }))
    except TimeoutError:
        print(json.dumps({"error": "timeout", "score": -999}))
    except Exception as e:
        print(json.dumps({"error": str(e), "score": -999}))
    finally:
        sig.alarm(0)  # Cancel timeout

    sys.stdout.flush()
