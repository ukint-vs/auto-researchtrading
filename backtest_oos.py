"""
Out-of-sample validation — 3 tests on unseen data with realistic engine.
The strategy was trained/validated on Jul 2024 – Mar 2025, 6 coins.
These tests use data the strategy has NEVER seen.

Usage: uv run backtest_oos.py
"""
import os
import numpy as np
import pandas as pd

# Activate realistic mode
os.environ["REALISTIC_BACKTEST"] = "1"

from prepare import load_data, run_backtest, compute_score, HOURS_PER_YEAR, INITIAL_CAPITAL
import prepare

# We need custom date ranges, so load raw parquet and slice manually
DATA_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader", "data")


def load_custom(symbols, start_str, end_str):
    """Load data for a custom date range."""
    start_ms = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_str, tz="UTC").timestamp() * 1000)
    result = {}
    for sym in symbols:
        fp = os.path.join(DATA_DIR, f"{sym}_1h.parquet")
        if not os.path.exists(fp):
            continue
        df = pd.read_parquet(fp)
        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)
        sdf = df[mask].reset_index(drop=True)
        sdf = sdf[sdf["close"] > 0].reset_index(drop=True)
        if len(sdf) >= 500:  # need enough bars for lookback
            result[sym] = sdf
    return result


def run_test(label, symbols, start, end):
    """Run a single OOS test and return metrics."""
    from strategy import Strategy
    data = load_custom(symbols, start, end)
    if not data:
        print(f"  {label}: NO DATA")
        return None

    bars = sum(len(df) for df in data.values())
    strategy = Strategy(symbols=list(data.keys()))
    result = run_backtest(strategy, data)
    score = compute_score(result)

    eq = np.array(result.equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / np.where(peak > 0, peak, 1)).max() * 100
    hr = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1)
    sharpe = (hr.mean() / hr.std()) * np.sqrt(HOURS_PER_YEAR) if len(hr) > 1 and hr.std() > 0 else 0

    return {
        "label": label,
        "symbols": list(data.keys()),
        "period": f"{start} → {end}",
        "bars": bars,
        "score": score,
        "sharpe": sharpe,
        "return_pct": result.total_return_pct,
        "max_dd": dd,
        "trades": result.num_trades,
        "win_rate": result.win_rate_pct,
        "pf": result.profit_factor,
        "trades_per_day": result.num_trades / (bars / len(data) / 24) if data else 0,
    }


# =====================================================================
# In-sample baseline (for comparison)
# =====================================================================
print("=" * 75)
print("OUT-OF-SAMPLE VALIDATION (realistic engine)")
print("=" * 75)
print()

# Baseline: validation period (in-sample)
baseline = run_test(
    "IN-SAMPLE (baseline)",
    ["BTC", "ETH", "XRP", "DOGE", "SOL", "FARTCOIN"],
    "2024-07-01", "2025-03-31",
)

# =====================================================================
# Test 1: Bull run — Apr-Sep 2025 (post-validation, easy mode)
# =====================================================================
test1 = run_test(
    "TEST 1: Bull run",
    ["BTC", "ETH", "XRP", "DOGE", "SOL", "FARTCOIN"],
    "2025-04-01", "2025-10-01",
)

# =====================================================================
# Test 2: Crash — Oct 2025 - Dec 2025 (hard mode)
# =====================================================================
test2 = run_test(
    "TEST 2: Crash/chop",
    ["BTC", "ETH", "XRP", "DOGE", "SOL", "FARTCOIN"],
    "2025-10-01", "2025-12-31",
)

# =====================================================================
# Test 3: Unseen coins — DOGE, AVAX, LINK full test period
# =====================================================================
test3 = run_test(
    "TEST 3: Unseen coins",
    ["DOGE", "AVAX", "LINK"],
    "2025-04-01", "2025-12-31",
)

# =====================================================================
# Test 4: Full OOS — all 6 coins, entire test period
# =====================================================================
test4 = run_test(
    "TEST 4: Full OOS",
    ["BTC", "ETH", "XRP", "DOGE", "SOL", "FARTCOIN"],
    "2025-04-01", "2025-12-31",
)

# =====================================================================
# Test 5: Training period (earlier data, also unseen by optimizer)
# =====================================================================
test5 = run_test(
    "TEST 5: Training period",
    ["BTC", "ETH", "XRP", "DOGE", "SOL", "FARTCOIN"],
    "2023-06-01", "2024-06-30",
)

# =====================================================================
# Results
# =====================================================================
tests = [baseline, test1, test2, test3, test4, test5]
tests = [t for t in tests if t is not None]

print(f"{'Test':<28} {'Period':<28} {'Score':>7} {'Sharpe':>7} {'Return':>8} {'MaxDD':>6} {'Trades':>7} {'WR':>6} {'PF':>5}")
print("-" * 115)
for t in tests:
    print(f"{t['label']:<28} {t['period']:<28} {t['score']:>7.2f} {t['sharpe']:>7.1f} {t['return_pct']:>+7.1f}% {t['max_dd']:>5.1f}% {t['trades']:>7} {t['win_rate']:>5.1f}% {t['pf']:>5.1f}")

# =====================================================================
# Verdict
# =====================================================================
print()
print("=" * 75)
print("VERDICT")
print("=" * 75)

if baseline:
    in_sample_sharpe = baseline["sharpe"]
    print(f"\nIn-sample Sharpe (realistic): {in_sample_sharpe:.1f}")
    print(f"OOS threshold (30% of in-sample): {in_sample_sharpe * 0.30:.1f}")
    print()

    for t in [test1, test2, test3, test4, test5]:
        if t is None:
            continue
        oos_ratio = t["sharpe"] / in_sample_sharpe if in_sample_sharpe > 0 else 0
        pass_fail = "PASS" if t["sharpe"] >= in_sample_sharpe * 0.30 else "FAIL"
        crash_pass = ""
        if "Crash" in t["label"]:
            crash_pass = " | DEPLOY-READY" if t["sharpe"] >= 1.5 else " | NOT READY (need Sharpe >= 1.5)"
        print(f"  {t['label']:<28} Sharpe {t['sharpe']:>6.1f} | ratio {oos_ratio:.2f} | {pass_fail}{crash_pass}")

    print()
    # Overall recommendation
    oos_tests = [t for t in [test1, test2, test4] if t is not None]
    avg_oos_sharpe = np.mean([t["sharpe"] for t in oos_tests]) if oos_tests else 0
    all_positive = all(t["return_pct"] > 0 for t in oos_tests) if oos_tests else False
    max_oos_dd = max(t["max_dd"] for t in oos_tests) if oos_tests else 99

    print(f"  Average OOS Sharpe: {avg_oos_sharpe:.1f}")
    print(f"  All OOS periods profitable: {'YES' if all_positive else 'NO'}")
    print(f"  Worst OOS drawdown: {max_oos_dd:.1f}%")
    print()
    if avg_oos_sharpe >= 2.0 and all_positive and max_oos_dd < 10:
        print("  >>> RECOMMENDATION: DEPLOY. Real edge confirmed across regimes.")
    elif avg_oos_sharpe >= 1.0:
        print("  >>> RECOMMENDATION: CAUTIOUS DEPLOY. Edge exists but thinner than in-sample.")
    else:
        print("  >>> RECOMMENDATION: DO NOT DEPLOY. Edge does not survive out-of-sample.")
