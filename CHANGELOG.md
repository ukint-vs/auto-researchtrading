# Changelog

## [Unreleased] — Phase 2: Expand Training Universe

### Added
- Auto-discovery of top 20 coins from Hyperliquid `metaAndAssetCtxs` API
- `discover_coins()`, `get_symbols()`, `_validate_symbols()` in prepare.py
- Data quality filters: zero-price bar removal, >1000x price ratio rejection, min 1000 bars per split
- CLI flags: `--discover` (print universe), `--refresh` (force re-discovery), `--info` (data completeness)
- Versioned universe cache at `~/.cache/autotrader/coin_universe.json`
- Dual scoreboard in backtest.py (expanded + legacy 7-coin reference)
- `Strategy(symbols=)` param for explicit symbol override
- `test_discovery.py` with 21 unit tests
- Named constants `MIN_BARS_PER_SPLIT`, `MAX_PRICE_RATIO`

### Changed
- `load_data()` accepts optional `symbols` param (defaults to validated training set)
- `download_data()` defaults to pipeline set (20 coins) with progress counter
- Strategy uses stable 1/N weights from `get_symbols("training")` instead of hardcoded constants
- SIGALRM cancelled before legacy scoreboard run to prevent timeout

### Removed
- Hardcoded `ACTIVE_SYMBOLS` and `SYMBOL_WEIGHTS` from strategy.py
- SOL high-correlation weight hack

### Scores
- Expanded universe (6 coins): **28.90**
- Legacy reference (7 coins): **32.06**
- Out-of-sample test split: **24.51** (114% retention)

---

## [1.0.0] — Phase 1: Universal Strategy

### Added
- Core signal ensemble: momentum, EMA, RSI, MACD, BB width, RSI divergence
- Adaptive exits: ATR trailing stop, SDO-tightened, RSI mean-reversion, signal flip
- Ehlers DSP building blocks: EOT, Boom Hunter, SDO
- Multi-timeframe aggregation (4h, 12h, 24h)
- Resident backtest server with TCP interface and parallel batch mode
- Autoresearch autonomous loop (`/autoresearch` command)
- Out-of-sample test script (`test_champion.py`)
- DSP unit tests (`test_dsp.py`)
- Performance optimizations: numpy vectorization, pointer-based bar iteration, iloc pre-slicing

### Scores
- 7-coin validation: **32.06** (after 201 autonomous experiments)
- Evolution: 2.7 → 8.4 → 13.5 → 19.7 → 20.6 → 25.09 → 32.06
