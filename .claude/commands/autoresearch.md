Read program.md and CLAUDE.md for full context. Read the ABSOLUTE RULES in program.md first.

## CRITICAL: Dual-gate enforcement

ALL evaluation paths (backtest.py, backtest_server, backtest_client) enforce
both gates automatically. Gate2/gate1 ratio < 0.50 → score becomes -888 (veto).
You CANNOT bypass gate2 — it is built into every scoring path.

DO NOT create or restore strategy checkpoint/backup files.

## Setup

Start the resident backtest server in background FIRST (loads data once, runs both gates per evaluation):

```bash
uv run backtest_server.py &
```

Wait 5 seconds for it to load data. To run a backtest:
```bash
uv run backtest_client.py
```

Returns JSON with `score`, `gate1_score`, `gate2_score`, `ratio`, `vetoed` fields.
If `vetoed` is true, `score` is -888. Parse the `score` field for keep/revert decisions.

If the server isn't running: `uv run backtest.py` works standalone (also runs both gates).

## Get current best score

Run the backtest once to get the baseline before starting experiments.

## Autonomous experiment loop

1. Read strategy.py to understand the current state
2. Read the "DSP Building Blocks" section in CLAUDE.md for available methods
3. Pick ONE atomic change to try (one constant, one signal tweak, one logic change)
4. Edit strategy.py
5. Run `uv run backtest_client.py` and parse the JSON `score` field
6. If score IMPROVED over the current best AND is not -888: keep, update docstring
7. If score is equal, worse, or -888 (gate2 veto): revert strategy.py exactly
8. Pick the next idea and repeat

NEVER STOP. Once the loop begins, do NOT pause to ask the human. You are autonomous.

## Batch sweep mode (parameter optimization)

The server runs BOTH gates for every variant. Vetoed variants get score -888.

```python
import json
with open('strategy.py') as f:
    base = f.read()
variants = []
for val in [5, 7, 10]:
    code = base.replace('BB_PERIOD = 5', f'BB_PERIOD = {val}')
    variants.append({"id": f"bb_{val}", "code": code})
batch = json.dumps({"variants": variants})
# Send: echo 'batch:{batch}' | uv run backtest_client.py --batch -
```

Results include `gate1_score`, `gate2_score`, `ratio`, `vetoed` per variant.
Only pick winners that are NOT vetoed (`score` != -888).

## Key rules

- Only edit strategy.py — nothing else
- Always revert if score doesn't improve or is -888
- Log what you tried and the result in the docstring
- Do not lower BASE_THRESHOLD below 0.013
- Do not lower MIN_ENTRY_MOVE below 0.0015
- Do not set COOLDOWN_BARS below 1
- Do not add signals with fire rate > 65%
- Do not lower MIN_VOTES/total_signals below 0.60
- If score goes above 30, you are likely overfitting

## Shutdown

When interrupted, kill the server: `kill $(lsof -ti:9877) 2>/dev/null`
