# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## OpenWolf

@.wolf/OPENWOLF.md

This project uses OpenWolf for context management. Read and follow .wolf/OPENWOLF.md every session. Check .wolf/cerebrum.md before generating code. Check .wolf/anatomy.md before reading files.

## Commands

```bash
uv run prepare.py              # Download data (one-time, cached to ~/.cache/autotrader/data/)
uv run prepare.py --symbols BTC ETH  # Download specific symbols only
uv run backtest.py             # Run strategy.py against validation data, prints score
uv run run_benchmarks.py       # Run all 5 benchmark strategies, print leaderboard
uv run export_equity.py        # Export equity curve to equity_curve.csv
uv run export_milestones.py    # Export equity curves at key autoresearch milestones
uv run generate_charts.py      # Generate visualization PNGs of experiment progression
```

Dependencies: numpy, pandas, scipy, requests, pyarrow (managed by uv via pyproject.toml).

## Architecture

**Immutable scaffold** (do not modify):
- `prepare.py` — Data download (CryptoCompare OHLCV + Hyperliquid funding rates), backtest engine, scoring formula
- `backtest.py` — Entry point: loads data via prepare.py, instantiates Strategy, runs backtest
- `benchmarks/` — 5 reference strategies for comparison

**Mutable** (the only file to edit for strategy work):
- `strategy.py` — The trading strategy that the autonomous loop evolves

**Data flow:**
```
CryptoCompare + Hyperliquid APIs → Parquet (~/.cache/autotrader/data/{SYMBOL}_1h.parquet)
  → 500-bar history buffer per symbol per hour
  → Strategy.on_bar(bar_data, portfolio) → list[Signal(symbol, target_position_usd)]
  → Backtester executes with slippage (1bp), maker fee (2bp), taker fee (5bp)
  → Scoring
```

## Key Abstractions

- **BarData** — Dataclass: symbol, timestamp, OHLC, volume, funding_rate, history (last 500 bars as DataFrame)
- **PortfolioState** — Dataclass: cash, positions (symbol → signed USD notional), entry_prices, equity, timestamp
- **Signal** — Trade instruction: symbol, target_position (signed USD: +long, -short, 0=close), order_type="market"
- **Strategy** — Class with `__init__()` and `on_bar(bar_data: dict, portfolio: PortfolioState) -> list[Signal]`

## Scoring Formula

```
score = sharpe × √(min(trades/50, 1.0)) - drawdown_penalty - turnover_penalty

sharpe = mean(daily_returns) / std(daily_returns) × √365
drawdown_penalty = max(0, max_drawdown% - 15%) × 0.05
turnover_penalty = max(0, annual_turnover/capital - 500) × 0.001
```

**Hard cutoffs (score = -999):** fewer than 10 trades, >50% max drawdown, or lost >50% of capital.

## Constraints

- Only edit `strategy.py` during autonomous experiments (everything else is immutable)
- No new dependencies beyond numpy, pandas, scipy, requests, pyarrow, stdlib
- 120-second backtest time budget per run
- Max leverage: 20x
- Data: BTC, ETH, SOL, DOGE, AVAX, LINK, XRP hourly bars; validation period Jul 2024 – Mar 2025; $100k initial capital
- Funding rates applied every 8 hours, scaled to hourly (fr / 8.0); longs pay positive funding

## Autonomous Experiment Loop

```bash
claude                    # Start Claude Code from repo root
/autoresearch            # Launch the autonomous strategy evolution loop
```

The loop: mutate strategy.py → backtest → keep if score improves → revert if worse. Only strategy.py changes between experiments.

## DSP Building Blocks Available in strategy.py

The strategy has pre-built Ehlers DSP methods ready for experimentation. The autoresearch loop should actively try combining these with the existing signals.

**Available methods on Strategy class:**

- `_ehlers_eot(closes, lpperiod, k1, k2)` → `(Q1, Q2)` arrays — Ehlers Even-Better Trigonometric Oscillator. Highpass filter (100-bar cutoff) → Supersmoother → peak-normalized → quotient lines.
- `_boom_hunter(closes, eot1_lp, eot2_lp, eot3_lp)` → `(trigger, q2, q3, q4, q5, q6)` — Three EOT instances. **Pass `closes[-BH_INPUT_LEN:]` for performance.** Trigger=SMA(2) of EOT1 Q1. EOT2 q3/q4 = trend lines. EOT3 q5/q6 = extreme detector.
- `_calc_sdo(highs, lows, closes, stoch_len, donch_len, smooth_len)` → `(sdo, signal)` — Stochastic-Donchian Oscillator (bounded 0-100).
- `_calc_rsi_divergence(closes, period, lookback)` → `(rsi, bull_div, bear_div)` — RSI with pivot-based divergence detection.

**Parameters to sweep:**
- BH_EOT1_LP: 4, 5, 6, 7, 8 (fast oscillator)
- BH_EOT2_LP: 20, 27, 35 (trend oscillator)
- BH_EOT3_LP: 8, 11, 14 (extreme detector)
- BH_BULL/BEAR_THRESHOLD: 0.0, 0.1, 0.2 (trigger sensitivity)
- BH_EXTREME_OB/OS: 0.6, 0.8, 0.9 (extreme zone thresholds)
- BH_TREND_WIDTH: 0.2, 0.3, 0.5 (EOT2 convergence threshold)
- SDO stoch_len: 10, 14, 20 (currently 14)
- SDO donch_len: 14, 20, 26 (currently 20)
- SDO oversold/overbought thresholds: 15-25 / 75-85
- RSI divergence lookback: 8, 10, 14, 20

**Boom Hunter usage in on_bar:**
```python
bh_closes = closes[-BH_INPUT_LEN:]  # truncate for performance
trigger, q2, q3, q4, q5, q6 = self._boom_hunter(bh_closes, BH_EOT1_LP, BH_EOT2_LP, BH_EOT3_LP)
# Entry signals:
bh_bull = trigger[-1] > BH_BULL_THRESHOLD   # EOT1 trigger bullish
bh_bear = trigger[-1] < -BH_BEAR_THRESHOLD  # EOT1 trigger bearish
# Trend filter (EOT2 convergence = trending market):
bh_trending = abs(q3[-1] - q4[-1]) < BH_TREND_WIDTH
# Extreme detector (EOT3 for exit tightening):
bh_overbought = q5[-1] > BH_EXTREME_OB
bh_oversold = q5[-1] < BH_EXTREME_OS
```

**Experiment ideas (try these atomically, one at a time):**
1. Add BH trigger as an additive entry signal (bh_bull/bh_bear in the vote count)
2. Use EOT2 trending filter: only enter when bh_trending is True (market is directional, not choppy)
3. Use EOT3 extreme for EXIT: tighten ATR stop when bh_overbought/bh_oversold (like SDO stop)
4. Combine EOT2 trending + EOT3 extreme: enter only when trending AND not at extremes
5. Use SDO as a trend-quality filter (only enter when SDO is between 30-70, meaning not at extremes)
6. Add RSI divergence as an additive signal with MIN_VOTES still at 4
7. Use EOT2 convergence width as a volatility filter (like BB compression but DSP-based)
8. Combine BB compression AND EOT2 convergence for a dual-compression signal
9. Use SDO slope (current - previous) as momentum confirmation instead of raw returns

**What didn't work (don't repeat these):**
- Replacing EMA crossover with EOT1 crossover directly (exp103, scored 20.19)
- Replacing dual-momentum with SDO zone-exit crossover (exp104, scored 19.07)
- Replacing RSI > 50 with RSI divergence only (exp105, scored 18.13)
- Adding EOT2 trend confirm as 7th signal with MIN_VOTES 5/7 (exp106, scored 18.25)

**Key insight:** Direct signal replacement scored worse. Try DSP as FILTERS, EXIT modifiers, or ADDITIVE signals with lower MIN_VOTES thresholds instead.

## Multi-Timeframe (4h) Building Block

`aggregate_to_4h(bd.history)` converts the 1h history DataFrame into 4h OHLCV arrays (dict with numpy arrays: open, high, low, close, volume, timestamp). With 500 1h bars → ~125 4h bars.

**Usage in on_bar:**
```python
bars_4h = aggregate_to_4h(bd.history)
closes_4h = bars_4h["close"]
ema_4h = ema(closes_4h[-20:], 7)  # 4h EMA
rsi_4h = calc_rsi(closes_4h, 8)    # 4h RSI
```

**Experiment ideas:**
1. Use 4h EMA trend as a higher-TF filter (only enter 1h longs when 4h EMA is bullish)
2. Use 4h RSI as an overbought/oversold gate (skip entries when 4h RSI > 70 or < 30)
3. Use 4h ATR for position sizing (bigger TF volatility → smaller size)
4. Use 4h momentum (4h close vs 6-bar-ago close) as a macro trend confirm
5. Use 4h BB width compression as a breakout detector (lower noise than 1h)

## Current Best Score

**32.06** (7 coins, equal weight, validation set — full run, no timeout). Beat this to keep your change.
