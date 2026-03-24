Read program.md and CLAUDE.md for full context. The current best score is shown when you run `uv run backtest.py`.

Start the autonomous experiment loop:

1. Read strategy.py to understand the current state
2. Read the "DSP Building Blocks" section in CLAUDE.md for available methods and experiment ideas
3. Pick ONE atomic change to try
4. Edit strategy.py
5. Run `uv run backtest.py` and capture the score
6. If score IMPROVED over the current best: keep the change, update the docstring with the new score
7. If score is equal or worse: revert strategy.py to the previous version exactly
8. Pick the next experiment idea and repeat

NEVER STOP. Once the loop begins, do NOT pause to ask the human. You are autonomous. If you run out of ideas, think harder — try parameter sweeps, combinations, filters, exit modifications. The loop runs until interrupted.

Key rules:
- Only edit strategy.py
- One change per experiment (atomic)
- Always revert if score doesn't improve
- Log what you tried and the result in the docstring
