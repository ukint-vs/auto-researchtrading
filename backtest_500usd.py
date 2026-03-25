"""
Live-config backtest: $500 USDC on Hyperliquid.
Tests strategy.py across different time periods and start dates.
Uses the validated training coins from the current universe.
"""
import numpy as np
import pandas as pd
from prepare import load_data, run_backtest, HOURS_PER_YEAR, get_symbols
import prepare
import strategy as strat_mod
from strategy import Strategy

CAPITAL = 500

# --- helpers ---
orig_capital = prepare.INITIAL_CAPITAL
orig_pos_pct = strat_mod.BASE_POSITION_PCT


def restore():
    prepare.INITIAL_CAPITAL = orig_capital
    strat_mod.BASE_POSITION_PCT = orig_pos_pct


def configure(capital=CAPITAL, pos_pct=None):
    prepare.INITIAL_CAPITAL = capital
    if pos_pct is not None:
        strat_mod.BASE_POSITION_PCT = pos_pct


def run_bt(data_dict, start_date=None, end_date=None):
    d = {}
    for sym, df in data_dict.items():
        filtered = df
        if start_date:
            ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
            filtered = filtered[filtered["timestamp"] >= ms]
        if end_date:
            ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)
            filtered = filtered[filtered["timestamp"] < ms]
        filtered = filtered.reset_index(drop=True)
        if len(filtered) > 0:
            d[sym] = filtered
    if not d:
        return None
    return run_backtest(Strategy(), d)


def metrics(r):
    if r is None:
        return {"final": CAPITAL, "ret": 0, "sharpe": 0, "dd": 0,
                "trades": 0, "wr": 0, "pf": 0, "turnover": 0}
    eq = np.array(r.equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / np.where(peak > 0, peak, 1)).max() * 100
    hr = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1)
    sharpe = (hr.mean() / hr.std()) * np.sqrt(HOURS_PER_YEAR) if len(hr) > 1 and hr.std() > 0 else 0
    return {"final": eq[-1], "ret": r.total_return_pct, "sharpe": sharpe,
            "dd": dd, "trades": r.num_trades, "wr": r.win_rate_pct,
            "pf": r.profit_factor, "turnover": r.annual_turnover}


def fmt_header():
    return f"{'Period':<30} {'Final':>8} {'Return':>9} {'Sharpe':>7} {'MaxDD':>6} {'Trades':>7} {'WR':>6} {'PF':>6}"


def fmt_row(label, m):
    return (f"{label:<30} ${m['final']:>7,.0f} {m['ret']:>+8.1f}% "
            f"{m['sharpe']:>7.1f} {m['dd']:>5.1f}% {m['trades']:>7} "
            f"{m['wr']:>5.1f}% {m['pf']:>5.1f}")


# --- Load data ---
coins = get_symbols("training")
print(f"Strategy coins: {coins}")
print(f"Capital: ${CAPITAL}")
print(f"Position size: {strat_mod.BASE_POSITION_PCT:.1%} of equity per coin")
print()

# Load all available splits
print("Loading data...")
val_data = load_data("val", symbols=coins)
train_data = load_data("train", symbols=coins)
print(f"  val:   {sum(len(df) for df in val_data.values())} bars, {len(val_data)} symbols")
print(f"  train: {sum(len(df) for df in train_data.values())} bars, {len(train_data)} symbols")

# Try test split too
test_data = load_data("test", symbols=coins)
if test_data:
    print(f"  test:  {sum(len(df) for df in test_data.values())} bars, {len(test_data)} symbols")
print()

# =====================================================================
# TEST 1: Full validation period at $500
# =====================================================================
configure(CAPITAL)
print("=" * 85)
print(f"TEST 1: Full periods at ${CAPITAL}")
print("=" * 85)
print()
print(fmt_header())
print("-" * 85)

r = run_bt(val_data)
print(fmt_row("Validation (Jul24-Mar25)", metrics(r)))

r = run_bt(train_data)
print(fmt_row("Training (Jun23-Jun24)", metrics(r)))

if test_data:
    r = run_bt(test_data)
    print(fmt_row("Test (Apr25+)", metrics(r)))

# =====================================================================
# TEST 2: Rolling windows within validation
# =====================================================================
print(f"\n{'=' * 85}")
print(f"TEST 2: Rolling 3-month windows (val period)")
print("=" * 85)
print()

windows = [
    ("2024-07-01", "2024-10-01", "Jul-Sep 2024"),
    ("2024-08-01", "2024-11-01", "Aug-Oct 2024"),
    ("2024-09-01", "2024-12-01", "Sep-Nov 2024"),
    ("2024-10-01", "2025-01-01", "Oct-Dec 2024"),
    ("2024-11-01", "2025-02-01", "Nov24-Jan25"),
    ("2024-12-01", "2025-03-01", "Dec24-Feb25"),
    ("2025-01-01", "2025-03-31", "Jan-Mar 2025"),
]

print(fmt_header())
print("-" * 85)
wins = 0
for start, end, label in windows:
    r = run_bt(val_data, start_date=start, end_date=end)
    m = metrics(r)
    print(fmt_row(label, m))
    if m["ret"] > 0:
        wins += 1

print(f"\nProfitable windows: {wins}/{len(windows)} ({wins/len(windows)*100:.0f}%)")

# =====================================================================
# TEST 3: Start date sensitivity (simulating "I deposit $500 today")
# =====================================================================
print(f"\n{'=' * 85}")
print(f"TEST 3: Start date sensitivity (if you deposited ${CAPITAL} on date X)")
print("=" * 85)
print()

starts = [
    ("2024-07-01", "Full val (9mo)"),
    ("2024-08-01", "Aug 2024 (8mo)"),
    ("2024-09-01", "Sep 2024 (7mo)"),
    ("2024-10-01", "Oct 2024 (6mo)"),
    ("2024-11-01", "Nov 2024 (5mo)"),
    ("2024-12-01", "Dec 2024 (4mo)"),
    ("2025-01-01", "Jan 2025 (3mo)"),
    ("2025-02-01", "Feb 2025 (2mo)"),
    ("2025-03-01", "Mar 2025 (1mo)"),
]

print(fmt_header())
print("-" * 85)
for date, label in starts:
    r = run_bt(val_data, start_date=date)
    m = metrics(r)
    marker = ""
    if m["ret"] < -10:
        marker = "  ⚠ LOSS"
    elif m["ret"] > 50:
        marker = "  ★"
    print(fmt_row(f"{date} {label}", m) + marker)

# =====================================================================
# TEST 4: Position sizing sensitivity at $500
# =====================================================================
print(f"\n{'=' * 85}")
print(f"TEST 4: Position sizing sweep (${CAPITAL}, full val)")
print("=" * 85)
print()

pos_pcts = [0.03, 0.04, 0.05, 0.065, 0.08, 0.10, 0.15, 0.20, 0.30]
print(f"{'POS%':>6} {'$/coin':>7} {'Final':>8} {'Return':>9} {'Sharpe':>7} {'MaxDD':>6} {'Trades':>7} {'Note'}")
print("-" * 85)
for pct in pos_pcts:
    configure(CAPITAL, pos_pct=pct)
    r = run_bt(val_data)
    m = metrics(r)
    per_coin = CAPITAL * pct
    note = ""
    if per_coin < 10:
        note = "⚠ <$10/coin (HL min)"
    elif m["dd"] > 30:
        note = "⚠ high DD"
    elif pct == orig_pos_pct:
        note = "← current"
    print(f"{pct:>6.3f} ${per_coin:>6.1f} ${m['final']:>7,.0f} {m['ret']:>+8.1f}% "
          f"{m['sharpe']:>7.1f} {m['dd']:>5.1f}% {m['trades']:>7} {note}")

# =====================================================================
# TEST 5: Per-coin breakdown (which coins help, which hurt)
# =====================================================================
configure(CAPITAL)  # reset to default sizing
print(f"\n{'=' * 85}")
print(f"TEST 5: Per-coin contribution (${CAPITAL}, default sizing, full val)")
print("=" * 85)
print()

# Run all coins together first
r_all = run_bt(val_data)
m_all = metrics(r_all)
print(f"All coins together: {fmt_row('ALL', m_all)}")
print()

# Leave-one-out analysis
print(f"{'Removed':<12} {'Final':>8} {'Return':>9} {'Sharpe':>7} {'Impact'}")
print("-" * 55)
for coin in coins:
    subset = {k: v for k, v in val_data.items() if k != coin}
    if not subset:
        continue
    r = run_bt(subset)
    m = metrics(r)
    impact = m_all["ret"] - m["ret"]
    direction = "helps" if impact > 0 else "hurts"
    print(f"  -{coin:<10} ${m['final']:>7,.0f} {m['ret']:>+8.1f}% {m['sharpe']:>7.1f} "
          f"{impact:>+6.1f}pp ({direction})")

# Single-coin runs
print(f"\n{'Coin':<12} {'Final':>8} {'Return':>9} {'Sharpe':>7} {'MaxDD':>6} {'Trades':>7}")
print("-" * 60)
for coin in coins:
    subset = {k: v for k, v in val_data.items() if k == coin}
    if not subset:
        continue
    r = run_bt(subset)
    m = metrics(r)
    print(f"  {coin:<10} ${m['final']:>7,.0f} {m['ret']:>+8.1f}% {m['sharpe']:>7.1f} {m['dd']:>5.1f}% {m['trades']:>7}")

# =====================================================================
# TEST 6: Worst-case drawdown timeline
# =====================================================================
print(f"\n{'=' * 85}")
print(f"TEST 6: Equity curve stats (${CAPITAL}, full val)")
print("=" * 85)

r = run_bt(val_data)
if r and r.equity_curve:
    eq = np.array(r.equity_curve)
    peak = np.maximum.accumulate(eq)
    dd_pct = (peak - eq) / np.where(peak > 0, peak, 1) * 100

    print(f"\n  Starting equity:  ${eq[0]:,.0f}")
    print(f"  Final equity:     ${eq[-1]:,.0f}")
    print(f"  Peak equity:      ${eq.max():,.0f}")
    print(f"  Trough equity:    ${eq.min():,.0f}")
    print(f"  Max drawdown:     {dd_pct.max():.1f}%")
    print(f"  Max $ drawdown:   ${(peak - eq).max():,.0f}")
    print(f"  Time underwater:  {(dd_pct > 0).sum()}/{len(dd_pct)} bars ({(dd_pct > 0).mean()*100:.0f}%)")
    print(f"  Bars in >5% DD:   {(dd_pct > 5).sum()}")
    print(f"  Bars in >10% DD:  {(dd_pct > 10).sum()}")

    # Monthly returns approximation
    print(f"\n  Approximate monthly returns:")
    chunk = max(1, len(eq) // 9)  # ~9 months
    for i in range(0, len(eq) - 1, chunk):
        end_i = min(i + chunk, len(eq) - 1)
        start_val = eq[i]
        end_val = eq[end_i]
        pct = (end_val - start_val) / start_val * 100
        month_num = i // chunk + 1
        print(f"    Month {month_num}: ${start_val:>7,.0f} → ${end_val:>7,.0f} ({pct:>+6.1f}%)")

restore()
print(f"\n{'=' * 85}")
print("Done.")
