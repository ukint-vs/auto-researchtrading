Read program.md and CLAUDE.md for full context.

## Setup

Start the resident backtest server in background FIRST (loads data once, saves ~2-3s per iteration):

```bash
uv run backtest_server.py &
```

Wait 5 seconds for it to load data and start listening.

To run a backtest, use:
```bash
uv run backtest_client.py
```

This returns a single JSON line. Parse the `score` field. If the server isn't running, it returns `{"error": "server not running", "score": -999}`.

If the server is not running, restart it: `uv run backtest_server.py &`

Fallback (no server): `uv run backtest.py` works standalone but is ~2-3s slower per run.

## Get current best score

Run the backtest once to get the baseline score before starting experiments.

## Autonomous experiment loop

1. Read strategy.py to understand the current state
2. Read the "DSP Building Blocks" section in CLAUDE.md for available methods and experiment ideas
3. Pick ONE atomic change to try
4. Edit strategy.py
5. Run `uv run backtest_client.py` and parse the JSON `score` field
6. If score IMPROVED over the current best: keep the change, update the docstring with the new score
7. If score is equal or worse: revert strategy.py to the previous version exactly
8. Pick the next experiment idea and repeat

NEVER STOP. Once the loop begins, do NOT pause to ask the human. You are autonomous. If you run out of ideas, think harder — try parameter sweeps, combinations, filters, exit modifications. The loop runs until interrupted.

## Key rules

- Only edit strategy.py
- One change per experiment (atomic)
- Always revert if score doesn't improve
- Log what you tried and the result in the docstring
- If the server dies or `uv run backtest_client.py` returns an error, restart server: `uv run backtest_server.py &`
- If JSON has `"error"` field, treat as score -999 (revert the change)

## Batch sweep mode (parameter optimization)

When you want to test multiple constant values at once (e.g., sweep BB_PERIOD across 5, 7, 10):

1. Read strategy.py to get the current source
2. Generate N variants as code strings (each with one constant changed)
3. Build batch JSON and send to server:

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

Or generate the JSON inline:
```bash
python3 -c "..." | uv run backtest_client.py --batch -
```

The server runs all variants in parallel and returns results sorted by score (best first).
Each result has: variant_id, score, sharpe, num_trades, total_return_pct, backtest_seconds.
Errors have: variant_id, score=-999, error_type (syntax/runtime), error message.

Use batch mode for:
- Parameter sweeps (BB_PERIOD, RSI thresholds, EMA periods, etc.)
- Testing multiple signal combinations simultaneously
- Quick grid search before detailed single-experiment refinement

## Shutdown

When interrupted, kill the server: `kill $(lsof -ti:9877) 2>/dev/null`
