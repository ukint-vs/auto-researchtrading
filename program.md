# autotrader

Autonomous trading strategy research on Hyperliquid perpetual futures.

## ABSOLUTE RULES — read before every experiment

```
FORBIDDEN (immediate revert if violated):
- Restoring any checkpoint/backup file (strategy_backup.py, strategy_checkpoint.py, etc.)
- Creating checkpoint/backup copies of strategy.py
- Batch parameter sweeps — ONE change per experiment, test via backtest.py (runs both gates)
- vshort threshold multiplier below 0.3 (below that it's padding)
- Any parameter change that increases a signal's fire rate above 65%
- Lowering BASE_THRESHOLD below 0.013 (more tiny trades = more fee drag live)
- Lowering MIN_ENTRY_MOVE below 0.0015 (15bps fee buffer is non-negotiable)
- Setting COOLDOWN_BARS below 1
- Setting MIN_VOTES/total_signals below 0.60
- Adding more than 8 signals total
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

Your job: **discover novel hourly-timeframe strategies** that are viable for live trading with real money.

## Dual-Gate Evaluation

Every experiment runs TWO backtests:

```bash
uv run backtest.py           # Gate 1: standard scoring (keep/revert decision)
uv run backtest_live.py      # Gate 2: realistic scoring (veto power)
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
signal_fire_rates: mom={X}% vshort={X}% ema={X}% rsi={X}% macd={X}% rsi_div={X}% donch={X}%
```

## Mandatory Constraints (NEVER violate)

1. **COOLDOWN_BARS >= 1** — zero cooldown is banned. Exploits instant-fill model.
2. **Vote ratio >= 60%** — MIN_VOTES / total_signals must be >= 0.60.
3. **No padding signals** — every signal must fire 20-65% of bars (bull side).
4. **MIN_ENTRY_MOVE = 0.0015** — skip entries below 15bps momentum.
5. **No more than 8 signals** — adding signals to dilute votes is cheating.
6. **BASE_POSITION_PCT <= 0.06** — larger sizes don't translate to live.
7. **Gate2/Gate1 ratio >= 0.50** — or revert regardless of gate1 improvement.

## Signal Quality Checklist

Before adding ANY signal, answer:
1. Fire rate 20-65%? (if >65%, it's padding; if <15%, it never contributes)
2. Standalone predictive value? (test as sole signal: Sharpe > 0?)
3. Redundant with existing? (add it, keep MIN_VOTES same — if no improvement, redundant)
4. Survives realistic engine? (check gate2 ratio before and after)

## Current Signal Audit

| Signal | Fire Rate | Status |
|--------|-----------|--------|
| Momentum 12h | 35% | ✅ Keep |
| Very-short 6h | 33% | ✅ Keep |
| EMA 5/23 | 51% | ✅ Trend-following, expected |
| RSI 4 | 51% | ✅ Oscillator, expected |
| MACD 7/30 | 50% | ✅ Trend-following, expected |
| RSI divergence | 21% | ✅ Leading signal |
| Donchian 5-bar | 41% | ✅ Keep |
| ~~BB compress~~ | ~~92%~~ | ❌ REMOVED — padding |
| ~~Micro 3-bar~~ | ~~51%~~ | ❌ REMOVED — coin flip |

Current: 7 signals, MIN_VOTES = 5 (5/7 = 71%) ✅

## Experiment Protocol

### Each experiment
```
1. PLAN: describe the change and hypothesis (1-2 sentences)
2. MEASURE: record signal fire rates if adding/modifying a signal
3. IMPLEMENT: make ONE atomic change to strategy.py
4. TEST gate1: uv run backtest.py
5. TEST gate2: uv run backtest_live.py
6. DECIDE: gate1 improved AND ratio >= 0.50 → KEEP, else → REVERT
7. LOG: commit with both scores, ratio, and signal fire rates
```

### Out-of-sample validation (every 10 keeps)
```
Run on test split. If OOS Sharpe < 30% of in-sample → revert last 10.
```

### Simplification pass (every 15 keeps)
```
Remove each signal one at a time. Keep removal if gate1 drops < 1.0 point.
```

## What "Good" Looks Like

| Metric | Gate 1 Target | Gate 2 Target | OOS Target |
|--------|---------------|---------------|------------|
| Score | 12-20 | 8-15 | Sharpe > 2.0 |
| Trades | 2000-5000 | 1500-4000 | — |
| Max DD | < 2% | < 3% | < 10% |
| Win rate | 65-80% | 60-75% | > 50% |

Scores above 25 gate1 are likely overfit. A strategy scoring 15/10/3.0 OOS
is better than one scoring 25/8/1.0 OOS. The first makes money live.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`).
2. **Create the branch**: `git checkout -b autotrader/<tag>` from current master.
3. **Read the in-scope files**: `prepare.py`, `strategy.py`, `backtest.py`, `backtest_live.py`, this file.
4. **Verify data exists**: `ls ~/.cache/autotrader/data/`
5. **Establish baselines**: run both gates, record scores.
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
