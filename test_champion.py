"""
Out-of-sample validation: run champion strategy on the test split (Apr-Dec 2025).
This data was NEVER seen during the 103 autoresearch experiments.

Usage: uv run test_champion.py
"""

from prepare import load_data, run_backtest, compute_score

from strategy import Strategy

strategy = Strategy()
data = load_data("test")

print(f"Loaded {sum(len(df) for df in data.values())} bars across {len(data)} symbols")
print(f"Symbols: {list(data.keys())}")
print(f"Split: test (Apr 2025 - Dec 2025)")
print()

result = run_backtest(strategy, data)
score = compute_score(result)

print("--- OUT-OF-SAMPLE RESULTS ---")
print(f"score:              {score:.6f}")
print(f"sharpe:             {result.sharpe:.6f}")
print(f"total_return_pct:   {result.total_return_pct:.6f}")
print(f"max_drawdown_pct:   {result.max_drawdown_pct:.6f}")
print(f"num_trades:         {result.num_trades}")
print(f"win_rate_pct:       {result.win_rate_pct:.6f}")
print(f"profit_factor:      {result.profit_factor:.6f}")
print(f"backtest_seconds:   {result.backtest_seconds:.1f}")
print()

# Compare to validation baseline
VAL_SCORE = 21.40
if score > 0:
    retention = score / VAL_SCORE * 100
    print(f"Validation score:   {VAL_SCORE:.2f}")
    print(f"Test score:         {score:.2f}")
    print(f"Retention:          {retention:.1f}% of validation performance")
    if retention > 50:
        print("VERDICT: Strategy has real edge. Proceed with confidence.")
    elif retention > 0:
        print("VERDICT: Partial edge, some overfitting. Proceed cautiously.")
    else:
        print("VERDICT: No edge on test set. Strategy is overfit.")
else:
    print(f"VERDICT: Hard cutoff triggered (score={score}). Strategy fails on unseen data.")
