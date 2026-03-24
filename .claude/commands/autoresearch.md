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

## Shutdown

When interrupted, kill the server: `kill $(lsof -ti:9877) 2>/dev/null`
