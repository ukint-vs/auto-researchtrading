Read program.md and CLAUDE.md for full context. Read the ABSOLUTE RULES in program.md first.

## CRITICAL: Dual-gate enforcement

backtest.py runs BOTH gates (standard + realistic) automatically. The printed
score is -888 if the realistic gate vetoes the change (ratio < 0.50).
You MUST use `uv run backtest.py` for every experiment — no shortcuts.

DO NOT use the backtest server (backtest_server.py / backtest_client.py) —
it bypasses Gate 2 and produces invalid scores. Use `uv run backtest.py` only.

DO NOT create or restore strategy checkpoint/backup files.
DO NOT batch sweep parameters — ONE atomic change per experiment.

## Get current best score

Run `uv run backtest.py` once to get the baseline before starting experiments.

## Autonomous experiment loop

1. Read strategy.py to understand the current state
2. Read the "DSP Building Blocks" section in CLAUDE.md for available methods
3. Pick ONE atomic change to try (one constant, one signal tweak, one logic change)
4. Edit strategy.py
5. Run `uv run backtest.py` and grep the `score:` line
6. If score IMPROVED over the current best AND is not -888: keep, update docstring
7. If score is equal, worse, or -888 (gate2 veto): revert strategy.py exactly
8. Pick the next idea and repeat

NEVER STOP. Once the loop begins, do NOT pause to ask the human. You are autonomous.

## Key rules

- Only edit strategy.py — nothing else
- One change per experiment (atomic) — no batch sweeps
- Always revert if score doesn't improve or is -888
- Log what you tried and the result in the docstring
- Do not lower BASE_THRESHOLD below 0.013
- Do not lower MIN_ENTRY_MOVE below 0.0015
- Do not set COOLDOWN_BARS below 1
- Do not add signals with fire rate > 65%
- Do not lower MIN_VOTES/total_signals below 0.60
- If score goes above 30, you are likely overfitting — check if changes are structural
  improvements or parameter tuning on seen data

## Shutdown

When interrupted, exit cleanly.
