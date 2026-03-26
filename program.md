# autotrader

Autonomous trading strategy research on Hyperliquid perpetual futures.

## ABSOLUTE RULES — read before every experiment

```
FORBIDDEN (immediate revert if violated):
- Restoring any checkpoint/backup file (strategy_backup.py, strategy_checkpoint.py, etc.)
- Creating checkpoint/backup copies of strategy.py
- Batch sweeps that bypass gate2 (the server and backtest.py both enforce dual-gate)
- vshort threshold multiplier below 0.3 (below that it's padding)
- Any parameter change that increases a signal's fire rate above 65%
- Lowering BASE_THRESHOLD below 0.013 (more tiny trades = more fee drag live)
- Lowering MIN_ENTRY_MOVE below 0.0015 (15bps fee buffer is non-negotiable)
- Setting COOLDOWN_BARS below 1 (at 1h) or below BAR_MULTIPLIER (at 5m)
- Setting MIN_VOTES/total_signals below 0.60
- Adding more than 10 signals total
- Modifying backtest.py, backtest_live.py, or prepare.py

backtest.py runs BOTH gates automatically. The score it prints is gate1,
but if gate2/gate1 ratio < 0.50, it prints -888 instead. You cannot
optimize on gate1 alone — the veto is built into the scoring.

ONE experiment = ONE atomic change → run backtest.py → keep or revert.
If you make two changes at once, you cannot attribute which one helped.
```

## Context

This project adapts Karpathy's autoresearch pattern for trading strategy discovery.
The owner (Nunchi) has existing production strategies that were designed for **tick-level market making** (20-second intervals). Those strategies underperform when ported to **hourly directional trading** on this backtest harness.

Your job: **discover novel strategies** that are viable for live trading with real money.

## 5m Mode

Set `BAR_INTERVAL=5m` before running backtest. The strategy computes all indicators on
1h bars aggregated from 5m data (same proven signals), but executes at 5m price precision.
This gives more realistic entry/exit fills. Runtime is ~9s per gate.

```bash
BAR_INTERVAL=5m uv run backtest.py    # 5m dual-gate evaluation
```

## Dual-Gate Evaluation

Every experiment runs TWO backtests:

```bash
BAR_INTERVAL=5m uv run backtest.py           # Gate 1: standard scoring (keep/revert decision)
BAR_INTERVAL=5m uv run backtest_live.py      # Gate 2: realistic scoring (veto power)
```

**Gate 1** — apples-to-apples comparison with all previous experiments.
**Gate 2** — next-bar-open execution, re-entry penalty, market impact, partial fills,
connection gaps, higher fees. Acts as a veto.

### Decision logic

```
gate1_score > current_best?
  YES → check ratio = gate2_score / gate1_score
    ratio >= 0.50 → KEEP
    ratio < 0.50  → REVERT (improvement is a backtest artifact)
  NO  → REVERT
```

### Commit message format

```
exp{N}: {description}

gate1: {score} (Sharpe {sharpe}, {trades} trades, {dd}% DD)
gate2: {score} (Sharpe {sharpe}, {trades} trades, {dd}% DD)
ratio: {ratio:.2f}
signal_fire_rates: mom={X}% vshort={X}% ema={X}% rsi={X}% macd={X}% rsi_div={X}% donch={X}% ggosc={X}%
```

## Mandatory Constraints (NEVER violate)

1. **COOLDOWN_BARS >= 1** (1h) or **>= BAR_MULTIPLIER** (5m) — zero cooldown is banned.
2. **Vote ratio >= 60%** — MIN_VOTES / total_signals must be >= 0.60.
3. **No padding signals** — every signal must fire 20-65% of bars (bull side).
4. **MIN_ENTRY_MOVE = 0.0015** — skip entries below 15bps momentum.
5. **No more than 10 signals** — adding signals to dilute votes is cheating.
6. **BASE_POSITION_PCT <= 0.06** — larger sizes don't translate to live.
7. **Gate2/Gate1 ratio >= 0.50** — or revert regardless of gate1 improvement.

## Current Signal Audit

| Signal | Fire Rate | Status |
|--------|-----------|--------|
| Momentum 12h | 35% | ✅ Keep |
| Very-short 6h | 33% | ✅ Keep |
| EMA 3/23 | 51% | ✅ Trend-following, expected |
| RSI 4 | 51% | ✅ Oscillator, expected |
| MACD 7/34 | 50% | ✅ Trend-following, expected |
| RSI divergence | 21% | ✅ Leading signal |
| Donchian 8-bar | 41% | ✅ Keep |
| GGOSC 3/10 | ~50% | ✅ Momentum oscillator (s6) |
| ~~BB compress~~ | ~~92%~~ | ❌ REMOVED — padding |
| ~~Micro 3-bar~~ | ~~51%~~ | ❌ REMOVED — coin flip |

Current: 8 signals, MIN_VOTES = 5 (5/8 = 62.5%) ✅

## Experiment Protocol

### Each experiment
```
1. PLAN: describe the change and hypothesis (1-2 sentences)
2. MEASURE: record signal fire rates if adding/modifying a signal
3. IMPLEMENT: make ONE atomic change to strategy.py
4. TEST: BAR_INTERVAL=5m uv run backtest.py
5. DECIDE: gate1 improved AND ratio >= 0.50 → KEEP, else → REVERT
6. LOG: commit with both scores, ratio, and signal fire rates
```

### Out-of-sample validation (every 10 keeps)
```
Run on test split. If OOS Sharpe < 30% of in-sample → revert last 10.
```

### Simplification pass (every 15 keeps)
```
Remove each signal one at a time. Keep removal if gate1 drops < 1.0 point.
```

## What "Good" Looks Like (5m targets)

| Metric | Gate 1 Target | Gate 2 Target | OOS Target |
|--------|---------------|---------------|------------|
| Score | 15-25 | 10-18 | Sharpe > 2.0 |
| Trades | 15000-30000 | 12000-25000 | — |
| Max DD | < 2% | < 3% | < 10% |
| Win rate | 55-75% | 50-70% | > 50% |

Scores above 30 gate1 are likely overfit. A strategy scoring 18/12/3.0 OOS
is better than one scoring 30/10/1.0 OOS. The first makes money live.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar26`).
2. **Create the branch**: `git checkout -b autotrader/<tag>` from current master.
3. **Read the in-scope files**: `prepare.py`, `strategy.py`, `backtest.py`, `backtest_live.py`, this file.
4. **Verify data exists**: `ls ~/.cache/autotrader/data/*_5m.parquet`
5. **Establish baselines**: `BAR_INTERVAL=5m uv run backtest.py`, record scores.
6. **Confirm and go**.

## Rules

**What you CAN do:**
- Modify `strategy.py` — this is the only file you edit.

**What you CANNOT do:**
- Modify `prepare.py`, `backtest.py`, `backtest_live.py`, or anything in `benchmarks/`.
- Install new packages. Only numpy, pandas, scipy, and standard library.
- Look at test set data (except during OOS validation every 10 keeps).
- Set COOLDOWN_BARS = 0.
- Add signals with fire rate > 65%.
- Lower MIN_VOTES/total below 0.60.

## Scoring Formula (from prepare.py)

```
score = sharpe * sqrt(trade_count_factor) - drawdown_penalty - turnover_penalty
trade_count_factor = min(num_trades / 50, 1.0)
drawdown_penalty = max(0, max_drawdown_pct - 15) * 0.05
turnover_penalty = max(0, annual_turnover/capital - 500) * 0.001
Hard cutoffs: <10 trades → -999, >50% drawdown → -999, lost >50% → -999
```

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, think harder. The loop runs until interrupted.
