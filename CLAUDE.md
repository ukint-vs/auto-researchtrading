# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AUTORESEARCH HARD CONSTRAINTS

**Read program.md ABSOLUTE RULES section before every experiment.**
- backtest.py runs both gates automatically — DO NOT skip or work around this
- ONE atomic change per experiment — NO batch sweeps
- DO NOT create or restore strategy backup/checkpoint files
- DO NOT lower BASE_THRESHOLD, MIN_ENTRY_MOVE, COOLDOWN_BARS, or vote ratio
- DO NOT add signals that fire > 65% of bars

## OpenWolf

@.wolf/OPENWOLF.md

This project uses OpenWolf for context management. Read and follow .wolf/OPENWOLF.md every session. Check .wolf/cerebrum.md before generating code. Check .wolf/anatomy.md before reading files.

## Commands

```bash
# 1h mode (default — CryptoCompare data)
uv run prepare.py              # Download 1h data for pipeline coins
uv run backtest.py             # Run strategy on 1h validation data

# 5m mode (Binance data, 1h indicators aggregated from 5m)
BAR_INTERVAL=5m uv run prepare.py              # Download 5m data from Binance
BAR_INTERVAL=5m uv run backtest.py             # Run strategy on 5m validation data

# Common
uv run prepare.py --symbols BTC ETH  # Download specific symbols only
uv run prepare.py --discover   # Print discovered universe from Hyperliquid and exit
uv run prepare.py --refresh    # Force re-discovery (ignore cache) then download
uv run prepare.py --info       # Show data completeness per symbol per split
uv run run_benchmarks.py       # Run all 5 benchmark strategies, print leaderboard
uv run export_equity.py        # Export equity curve to equity_curve.csv
uv run export_milestones.py    # Export equity curves at key autoresearch milestones
uv run generate_charts.py      # Generate visualization PNGs of experiment progression
```

Dependencies: numpy, pandas, scipy, requests, pyarrow (managed by uv via pyproject.toml).

## Multi-Timeframe Architecture

The strategy supports both 1h and 5m bar intervals via `BAR_INTERVAL` env var.

**At 1h (default):** Indicators computed directly on 1h bars from CryptoCompare. History buffer: 500 bars.

**At 5m:** The backtest loop runs at 5m resolution (78k bars), but **all indicators are computed on 1h bars aggregated from 5m data**. This gives 5m execution precision (better entry/exit prices) with 1h signal quality (proven ensemble). The aggregation is cached and only recomputed at hour boundaries — so 5m runs at ~9s per gate, comparable to 1h.

Key implementation details:
- `_aggregate_to_1h()` groups 5m bars by hour → 1h OHLCV
- `_get_1h_arrays()` caches 1h data, only rebuilds at new hour boundaries
- `_compute_and_cache_signals()` caches signal decisions between hours
- Between hours, `on_bar()` uses cached signals + 5m `mid` price for stops
- `COOLDOWN_BARS` is scaled by `BAR_MULTIPLIER` (12 at 5m = 1 hour cooldown)
- `history=None` between hours (engine skips DataFrame construction)

## Architecture

**Scaffold** (do not modify during autoresearch):
- `prepare.py` — Data download, coin discovery, backtest engine, scoring formula
- `backtest.py` — Entry point: loads data via prepare.py, instantiates Strategy, runs backtest (dual scoreboard)
- `benchmarks/` — 5 reference strategies for comparison

**Mutable** (the only file to edit for strategy work):
- `strategy.py` — The trading strategy that the autonomous loop evolves

**Data flow:**
```
1h mode:
  CryptoCompare + Hyperliquid APIs → {SYMBOL}_1h.parquet
  → 500-bar history buffer → Strategy.on_bar() → Scoring

5m mode:
  Binance Klines API + Hyperliquid funding → {SYMBOL}_5m.parquet
  → 700-bar history buffer → aggregate to 1h (cached) → Strategy.on_bar() → Scoring
  → Execution at 5m price precision, indicators at 1h resolution
```

## Key Abstractions

- **BarData** — Dataclass: symbol, timestamp, OHLC, volume, funding_rate, history (DataFrame or None between hours at 5m)
- **PortfolioState** — Dataclass: cash, positions (symbol → signed USD notional), entry_prices, equity, timestamp
- **Signal** — Trade instruction: symbol, target_position (signed USD: +long, -short, 0=close), order_type="market"
- **Strategy** — Class with `__init__()` and `on_bar(bar_data: dict, portfolio: PortfolioState) -> list[Signal]`

## Scoring Formula

```
score = sharpe × √(min(trades/50, 1.0)) - drawdown_penalty - turnover_penalty

sharpe = mean(bar_returns) / std(bar_returns) × √(BARS_PER_YEAR)
drawdown_penalty = max(0, max_drawdown% - 15%) × 0.05
turnover_penalty = max(0, annual_turnover/capital - 500) × 0.001
```

**Hard cutoffs (score = -999):** fewer than 10 trades, >50% max drawdown, or lost >50% of capital.

## Constraints

- Only edit `strategy.py` during autonomous experiments (everything else is immutable)
- No new dependencies beyond numpy, pandas, scipy, requests, pyarrow, stdlib
- Max leverage: 20x
- Data: auto-discovered from Hyperliquid; validation period Jul 2024 – Mar 2025; $100k initial capital
- Funding rates applied every 8 hours, scaled per bar (fr / BARS_PER_FUNDING)
- 5 coins: BTC, ETH, DOGE, XRP, SOL

## Autonomous Experiment Loop

```bash
# 5m autoresearch (recommended — more realistic execution)
BAR_INTERVAL=5m claude        # Start Claude Code with 5m mode
/autoresearch                 # Launch the autonomous strategy evolution loop

# 1h autoresearch (legacy)
claude                        # Start Claude Code from repo root
/autoresearch                 # Launch the autonomous strategy evolution loop
```

The loop: mutate strategy.py → backtest → keep if score improves → revert if worse. Only strategy.py changes between experiments.

## GGOSC Oscillator (8th signal + TP exit)

GGOSC (Gradient-Gated Oscillator) is a momentum oscillator that normalizes EMA crossover by ATR. It was added in s6 and improved gate1 from 24.85 → 27.33 on 1h.

**How it works:**
```
midpoint = (high + low) / 2
osc = (EMA_fast(midpoint) - EMA_slow(midpoint)) / ATR(midpoint)
```
- Positive osc → bullish momentum (fast EMA above slow, normalized by volatility)
- Negative osc → bearish momentum
- The ATR normalization makes it comparable across coins with different volatility

**Current parameters:**
- `GGOSC_FAST = 3` — fast EMA span (3 bars = responsive)
- `GGOSC_SLOW = 10` — slow EMA span (10 bars = trend)
- `GGOSC_SIGNAL = 3` — signal smoothing (used internally for buffer sizing)
- `GGOSC_TP1_MULT = 0.8` — take profit at 0.8× ATR from entry price

**Two roles in the strategy:**
1. **Entry signal** (8th vote): `ggosc_val > 0` = bull, `ggosc_val < 0` = bear
2. **Exit (TP1)**: Close position when price reaches entry ± 0.8× ATR_at_entry

**Parameters to sweep:**
- GGOSC_FAST: 2, 3, 4, 5 (faster = more responsive, noisier)
- GGOSC_SLOW: 7, 10, 14, 20 (slower = smoother trend, more lag)
- GGOSC_TP1_MULT: 0.5, 0.8, 1.0, 1.2, 1.5 (lower = tighter TP, higher = let winners run)
- Threshold: currently 0 (any positive/negative). Try 0.1, 0.2 for stronger signals

**Experiment ideas:**
1. Tune GGOSC_TP1_MULT (0.5 vs 1.0 vs 1.5) — major impact on trade duration
2. Add a threshold: only vote when |ggosc_val| > 0.1 (reduces noise)
3. Use GGOSC slope (current - previous) as momentum acceleration filter
4. Disable GGOSC TP1 exit (keep as signal only) — the TP may cut winners short
5. Use GGOSC as exit-only (remove from vote, keep TP) — test signal vs exit contribution

## DSP Building Blocks Available in strategy.py

The strategy has pre-built Ehlers DSP methods ready for experimentation.

**Available methods on Strategy class:**

- `_ehlers_eot(closes, lpperiod, k1, k2)` → `(Q1, Q2)` arrays — Ehlers Even-Better Trigonometric Oscillator.
- `_boom_hunter(closes, eot1_lp, eot2_lp, eot3_lp)` → `(trigger, q2, q3, q4, q5, q6)` — Three EOT instances.
- `_calc_sdo(highs, lows, closes)` → `(sdo, signal)` — Stochastic-Donchian Oscillator (bounded 0-100).
- `_calc_ggosc(highs, lows, closes)` → `float` — Gradient-Gated Oscillator (unbounded, 0-centered).
- `_calc_rsi_divergence(closes)` → `(rsi, bull_div, bear_div)` — RSI with pivot-based divergence detection.

**What didn't work (don't repeat these):**
- Replacing EMA crossover with EOT1 crossover directly (exp103, scored 20.19)
- Replacing dual-momentum with SDO zone-exit crossover (exp104, scored 19.07)
- Replacing RSI > 50 with RSI divergence only (exp105, scored 18.13)
- Adding EOT2 trend confirm as 7th signal with MIN_VOTES 5/7 (exp106, scored 18.25)

**Key insight:** Direct signal replacement scored worse. Try DSP as FILTERS, EXIT modifiers, or ADDITIVE signals instead.

## Current Best Score (Dual-Gate)

### 1h Baseline (CryptoCompare data)
**Gate 1: 26.85** | **Gate 2: 19.88** | **Ratio: 0.74**
(5 coins: BTC ETH DOGE XRP SOL, validation set)

### 5m Baseline (Binance data, 1h indicators from aggregated 5m)
**Gate 1: 16.15** | **Gate 2: 10.89** | **Ratio: 0.67**
(5 coins: BTC ETH DOGE XRP SOL, validation set, ~9s per gate)

Gate 1 = `uv run backtest.py` (standard scoring, keep/revert decision)
Gate 2 = `uv run backtest_live.py` (realistic: next-bar execution, re-entry penalty, market impact)
Ratio = gate2/gate1 — must be >= 0.50 or revert.

Strategy: 8-signal ensemble, 5/8 vote (62.5%), COOLDOWN=1, MIN_ENTRY_MOVE=15bps.
Signals: momentum, vshort, EMA, RSI, MACD, RSI div, Donchian, GGOSC.
Exits: GGOSC TP1 (0.8× ATR), SDO trailing stop (1.85× ATR), RSI mean-reversion, signal flip.
At 5m: indicators on aggregated 1h bars, execution at 5m price precision.

## Live-Realism Constraints (NEVER violate)

1. **COOLDOWN_BARS >= 1** — zero cooldown is banned
2. **Vote ratio >= 60%** — MIN_VOTES / total_signals >= 0.60
3. **No padding signals** — fire rate must be 20-65% (bull side)
4. **MIN_ENTRY_MOVE = 0.0015** — skip entries below 15bps momentum
5. **No more than 10 signals** — adding signals to dilute votes = cheating
6. **BASE_POSITION_PCT <= 0.06**
7. **Dual-gate evaluation** — gate2/gate1 ratio >= 0.50 or revert
