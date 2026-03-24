# TODOS

## Goal: Universal Trading Strategy → Top 50 Hyperliquid Deployment

Build a strategy that generalizes across all crypto pairs, then optionally tune per-asset.

### Phase 1: Universal Strategy (current)
- [x] Core signal ensemble (momentum, EMA, RSI, MACD, BB, RSI divergence)
- [x] Adaptive exits (ATR trailing stop, SDO-tightened, RSI mean-reversion, signal flip)
- [x] Autoresearch loop for parameter optimization
- [x] Score: 25.09 on 7-coin validation set (Jul 2024 – Mar 2025)
- [ ] Implement parallel backtest processing (in progress)
- [ ] Continue autoresearch: structural changes, new signal types, exit logic improvements

### Phase 2: Expand Training Universe
- [ ] Expand data pipeline (`prepare.py`) to pull top 20-30 coins from Hyperliquid/CryptoCompare
- [ ] Include mid-cap coins in training set (not just BTC/ETH/SOL) to prevent large-cap overfitting
- [ ] Autoresearch on 10-coin diverse training set (mix of large, mid, small cap)
- [ ] Validate that strategy scores well across all caps, not just large

### Phase 3: Robustness for Many Pairs
- [ ] Liquidity-aware position sizing (cap max position by daily volume)
- [ ] Dynamic slippage model (1bp is wrong for illiquid coins — need volume-based estimate)
- [ ] Per-coin vol regime detection (current TARGET_VOL assumes large-cap crypto vol)
- [ ] Equal-weight 1/N allocation across N coins (no hand-tuned weights)
- [ ] Stress test: does the strategy stay profitable on coins it's never seen?

### Phase 4: Live Deployment
- [ ] Top 50 Hyperliquid coin discovery (dynamic, market-cap ranked)
- [ ] Live paper trading on expanded universe
- [ ] Per-asset tuning pass (optional — only if clear per-regime edge exists)
- [ ] Production monitoring and position management

### Backlog
- [ ] Benchmark autoresearch loop throughput (experiments/hour)
- [ ] Explore regime-switching (trending vs ranging) as a meta-strategy layer
- [ ] Cross-asset correlation management (reduce exposure when everything moves together)
