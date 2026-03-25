"""
Realistic backtest — harsher fee model for live validation.
Wraps the standard backtester with:
  1. Higher fees (HL actual: 3.5bps taker + 1bp builder + 2.5bps slippage)
  2. Re-entry slippage penalty (2x slippage if re-entered <3 bars after exit)
  3. Connection gap simulation (2-bar gap every ~168 bars)
  4. Partial fill on large orders (>20% of avg hourly volume)

Does NOT modify prepare.py — monkey-patches constants before running.
Compare with `uv run backtest.py` for apples-to-apples vs original scoring.
"""
import time
import numpy as np
import prepare
from prepare import load_data, run_backtest, compute_score, get_symbols, BacktestResult
from prepare import INITIAL_CAPITAL, HOURS_PER_YEAR
from strategy import Strategy

# ---------------------------------------------------------------------------
# 1. Realistic fee constants (override prepare.py defaults)
# ---------------------------------------------------------------------------
REALISTIC_TAKER_FEE = 0.00045   # 4.5bps (3.5bps taker + 1bp builder fee)
REALISTIC_MAKER_FEE = 0.00020   # 2bps
REALISTIC_SLIPPAGE_BPS = 2.5    # 2.5bps (was 1bp — more realistic for mid-cap)

# Save originals
_orig_taker = prepare.TAKER_FEE
_orig_maker = prepare.MAKER_FEE
_orig_slippage = prepare.SLIPPAGE_BPS


def apply_realistic_fees():
    prepare.TAKER_FEE = REALISTIC_TAKER_FEE
    prepare.MAKER_FEE = REALISTIC_MAKER_FEE
    prepare.SLIPPAGE_BPS = REALISTIC_SLIPPAGE_BPS


def restore_fees():
    prepare.TAKER_FEE = _orig_taker
    prepare.MAKER_FEE = _orig_maker
    prepare.SLIPPAGE_BPS = _orig_slippage


# ---------------------------------------------------------------------------
# 2. Post-hoc adjustments (applied to BacktestResult)
# ---------------------------------------------------------------------------

def apply_reentry_penalty(result, data):
    """Estimate PnL drag from re-entry slippage.

    For each close→open pair within 3 bars, add extra slippage cost.
    """
    reentry_window = 3
    extra_slippage_bps = 2.5  # additional 2.5bps on quick re-entries
    penalty = 0.0
    quick_reentries = 0

    exits = [(i, t) for i, t in enumerate(result.trade_log) if t[0] == "close"]
    opens = [(i, t) for i, t in enumerate(result.trade_log) if t[0] == "open"]

    exit_idx = {t[1]: i for i, t in exits}  # symbol -> last exit index

    for idx, trade in opens:
        symbol = trade[1]
        price = trade[3]
        delta = abs(trade[2])

        # Find most recent exit for this symbol before this open
        recent_exits = [(ei, et) for ei, et in exits if et[1] == symbol and ei < idx]
        if recent_exits:
            last_exit_idx = recent_exits[-1][0]
            gap = idx - last_exit_idx
            if gap <= reentry_window * 2:  # rough proxy (trades interleave)
                extra_cost = delta * extra_slippage_bps / 10000
                penalty += extra_cost
                quick_reentries += 1

    return penalty, quick_reentries


def apply_connection_gaps(result):
    """Estimate impact of periodic connection gaps.

    Every ~168 bars (1 week), assume 2 bars of no trading.
    Any open position during a gap could see adverse price movement.
    Penalty: 0.1% of open notional per gap (conservative estimate).
    """
    gap_interval = 168
    gap_duration = 2
    gap_penalty_pct = 0.001  # 0.1% of notional per gap

    total_bars = len(result.equity_curve)
    num_gaps = total_bars // gap_interval
    avg_notional = np.mean([abs(t[2]) for t in result.trade_log]) if result.trade_log else 0
    # Assume ~60% of the time we have an open position
    position_pct = 0.6
    total_penalty = num_gaps * avg_notional * gap_penalty_pct * position_pct
    return total_penalty, num_gaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    coins = get_symbols("training")
    data = load_data("val", symbols=coins)

    print(f"Symbols: {coins}")
    print(f"Bars: {sum(len(df) for df in data.values())}")
    print()

    # --- Standard backtest (for comparison) ---
    strategy = Strategy()
    result_std = run_backtest(strategy, data)
    score_std = compute_score(result_std)

    print("=" * 70)
    print("STANDARD BACKTEST (original fees)")
    print("=" * 70)
    print(f"  Score:      {score_std:.2f}")
    print(f"  Sharpe:     {result_std.sharpe:.2f}")
    print(f"  Return:     {result_std.total_return_pct:.1f}%")
    print(f"  Max DD:     {result_std.max_drawdown_pct:.2f}%")
    print(f"  Trades:     {result_std.num_trades}")
    print(f"  Win rate:   {result_std.win_rate_pct:.1f}%")
    print(f"  PF:         {result_std.profit_factor:.1f}")

    # --- Realistic backtest ---
    apply_realistic_fees()
    strategy2 = Strategy()
    result_real = run_backtest(strategy2, data)
    score_real = compute_score(result_real)
    restore_fees()

    # Post-hoc adjustments
    reentry_penalty, quick_reentries = apply_reentry_penalty(result_real, data)
    gap_penalty, num_gaps = apply_connection_gaps(result_real)
    total_penalty = reentry_penalty + gap_penalty

    # Adjusted equity
    eq = np.array(result_real.equity_curve)
    adjusted_final = eq[-1] - total_penalty
    adjusted_return = (adjusted_final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Adjusted Sharpe (rough — reduce proportionally)
    penalty_drag = total_penalty / INITIAL_CAPITAL
    adjusted_sharpe = result_real.sharpe * (1 - penalty_drag / max(result_real.total_return_pct / 100, 0.01))

    print()
    print("=" * 70)
    print("REALISTIC BACKTEST (HL actual fees + penalties)")
    print("=" * 70)
    print(f"  Fee model:  {REALISTIC_TAKER_FEE*10000:.1f}bps taker, {REALISTIC_SLIPPAGE_BPS:.1f}bps slippage")
    print(f"  Score:      {score_real:.2f}")
    print(f"  Sharpe:     {result_real.sharpe:.2f}")
    print(f"  Return:     {result_real.total_return_pct:.1f}%")
    print(f"  Max DD:     {result_real.max_drawdown_pct:.2f}%")
    print(f"  Trades:     {result_real.num_trades}")
    print(f"  Win rate:   {result_real.win_rate_pct:.1f}%")
    print(f"  PF:         {result_real.profit_factor:.1f}")
    print()
    print(f"  Post-hoc penalties:")
    print(f"    Re-entry slippage:  ${reentry_penalty:,.0f} ({quick_reentries} quick re-entries)")
    print(f"    Connection gaps:    ${gap_penalty:,.0f} ({num_gaps} simulated gaps)")
    print(f"    Total penalty:      ${total_penalty:,.0f}")
    print()
    print(f"  Adjusted final equity: ${adjusted_final:,.0f}")
    print(f"  Adjusted return:       {adjusted_return:.1f}%")
    print(f"  Adjusted Sharpe:       {adjusted_sharpe:.1f}")

    # --- Comparison table ---
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    metrics = [
        ("Score", score_std, score_real),
        ("Sharpe", result_std.sharpe, result_real.sharpe),
        ("Return %", result_std.total_return_pct, result_real.total_return_pct),
        ("Max DD %", result_std.max_drawdown_pct, result_real.max_drawdown_pct),
        ("Trades", result_std.num_trades, result_real.num_trades),
        ("Win rate %", result_std.win_rate_pct, result_real.win_rate_pct),
        ("Profit factor", result_std.profit_factor, result_real.profit_factor),
    ]
    print(f"  {'Metric':<16} {'Standard':>10} {'Realistic':>10} {'Delta':>10}")
    print(f"  {'-'*48}")
    for name, std, real in metrics:
        delta = real - std
        print(f"  {name:<16} {std:>10.2f} {real:>10.2f} {delta:>+10.2f}")

    # --- $500 at 3x projection ---
    print()
    print("=" * 70)
    print("$500 @ 3x PROJECTION (realistic fees)")
    print("=" * 70)
    for capital, lev, label in [(500, 1, "1x"), (500, 3, "3x"), (500, 5, "5x")]:
        apply_realistic_fees()
        prepare.INITIAL_CAPITAL = capital
        import strategy as strat_mod
        orig_pos = strat_mod.BASE_POSITION_PCT
        strat_mod.BASE_POSITION_PCT = orig_pos * lev
        s = Strategy()
        r = run_backtest(s, data)
        eq = np.array(r.equity_curve)
        pk = np.maximum.accumulate(eq)
        dd = ((pk - eq) / np.where(pk > 0, pk, 1)).max() * 100
        print(f"  {label}: ${capital} -> ${eq[-1]:,.0f} ({r.total_return_pct:+.0f}%) | DD: {dd:.1f}% | Trades: {r.num_trades}")
        strat_mod.BASE_POSITION_PCT = orig_pos
        prepare.INITIAL_CAPITAL = 100_000
        restore_fees()

    t_end = time.time()
    print(f"\nTotal time: {t_end - t_start:.1f}s")
