"""
Autotrader backtesting engine. Fixed evaluation harness — DO NOT MODIFY.
Downloads Hyperliquid historical data, runs backtests, computes scores.

Usage:
    python prepare.py                  # download data
    python prepare.py --symbols BTC    # download specific symbols
"""

import os
import sys
import time
import math
import signal
import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 120              # backtest time budget in seconds (2 minutes)
INITIAL_CAPITAL = 100_000.0    # $100K starting capital
MAKER_FEE = 0.0002             # 2 bps
TAKER_FEE = 0.0005             # 5 bps
SLIPPAGE_BPS = 1.0             # 1 bps simulated slippage
MAX_LEVERAGE = 20              # max leverage allowed
LOOKBACK_BARS = 500            # history buffer provided to strategy
BAR_INTERVAL = "1h"
MAINTENANCE_MARGIN_PCT = 5.0   # % of initial capital — intra-bar liquidation threshold

DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "XRP"]
SYMBOLS = DEFAULT_SYMBOLS  # backward compat alias

# Date splits (UTC timestamps)
TRAIN_START = "2023-06-01"
TRAIN_END = "2024-06-30"
VAL_START = "2024-07-01"
VAL_END = "2025-03-31"
TEST_START = "2025-04-01"
TEST_END = "2025-12-31"

HOURS_PER_YEAR = 8760

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autotrader")
DATA_DIR = os.path.join(CACHE_DIR, "data")
UNIVERSE_CACHE = os.path.join(CACHE_DIR, "coin_universe.json")

# Universe discovery constants
MIN_DAILY_VOLUME_USD = 1_000_000
PIPELINE_SIZE = 20
TRAINING_SIZE = 10
STABLECOINS = {"USDC", "USDT", "DAI", "TUSD", "BUSD", "FRAX", "LUSD", "USDP", "PYUSD", "FDUSD", "USD"}
CAP_TIERS = {
    "BTC": 1, "ETH": 1,
    "SOL": 2, "XRP": 2, "DOGE": 2, "ADA": 2, "AVAX": 2, "LINK": 2,
    "BNB": 2, "TRX": 2, "DOT": 2, "MATIC": 2, "POL": 2, "SHIB": 2,
    "LTC": 2, "UNI": 2, "BCH": 2, "NEAR": 2, "APT": 2, "FIL": 2,
    "ICP": 2, "ATOM": 2, "ARB": 2, "OP": 2, "SUI": 2, "SEI": 2,
}
TRAINING_TIER_MIN = {1: 2, 2: 3}  # min coins per tier in training set
MIN_BARS_PER_SPLIT = 1000         # exclude symbols with fewer bars in a split
MAX_PRICE_RATIO = 1000            # exclude symbols with extreme price swings (bad data)

# ---------------------------------------------------------------------------
# Universe discovery
# ---------------------------------------------------------------------------

import json
from datetime import datetime


def discover_coins(top_n=PIPELINE_SIZE):
    """Query Hyperliquid for top perp markets ranked by volume + OI.

    Returns dict with pipeline (all top_n), training (diverse subset),
    tiers mapping, and a version string for reproducibility.
    """
    hl_url = "https://api.hyperliquid.xyz/info"
    try:
        resp = requests.post(hl_url, json={"type": "metaAndAssetCtxs"}, timeout=30)
        resp.raise_for_status()
        meta, asset_ctxs = resp.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
        print(f"  Warning: Hyperliquid API failed ({e}), using default symbols")
        return _default_universe()

    # Parse market data
    coins = []
    for i, ctx in enumerate(asset_ctxs):
        try:
            name = meta["universe"][i]["name"]
            volume = float(ctx.get("dayNtlVlm", 0))
            oi = float(ctx.get("openInterest", 0))
            coins.append({"name": name, "volume": volume, "oi": oi})
        except (KeyError, IndexError, ValueError):
            continue

    # Filter stablecoins and low-volume
    coins = [c for c in coins if c["name"] not in STABLECOINS and c["volume"] >= MIN_DAILY_VOLUME_USD]

    if not coins:
        print("  Warning: all coins filtered out, using default symbols")
        return _default_universe()

    # Rank by composite score: 60% volume + 40% OI
    coins.sort(key=lambda c: c["volume"], reverse=True)
    for rank, c in enumerate(coins):
        c["vol_rank"] = rank
    coins.sort(key=lambda c: c["oi"], reverse=True)
    for rank, c in enumerate(coins):
        c["oi_rank"] = rank
    coins.sort(key=lambda c: 0.6 * c["vol_rank"] + 0.4 * c["oi_rank"])

    pipeline = [c["name"] for c in coins[:top_n]]

    # Select diverse training subset
    training = _select_training_subset(pipeline)

    version = f"v1-{datetime.utcnow().strftime('%Y%m%d')}"
    tiers = {sym: CAP_TIERS.get(sym, 3) for sym in pipeline}

    return {"pipeline": pipeline, "training": training, "tiers": tiers, "version": version}


def _select_training_subset(pipeline):
    """Select TRAINING_SIZE coins from pipeline ensuring tier diversity."""
    selected = []
    remaining = list(pipeline)

    # Ensure BTC and ETH are always included (tier 1 anchors)
    for sym in ["BTC", "ETH"]:
        if sym in remaining:
            selected.append(sym)
            remaining.remove(sym)

    # Fill tier minimums
    for tier, min_count in sorted(TRAINING_TIER_MIN.items()):
        current = sum(1 for s in selected if CAP_TIERS.get(s, 3) == tier)
        for sym in list(remaining):
            if current >= min_count:
                break
            if CAP_TIERS.get(sym, 3) == tier:
                selected.append(sym)
                remaining.remove(sym)
                current += 1

    # Fill rest by rank order
    for sym in remaining:
        if len(selected) >= TRAINING_SIZE:
            break
        selected.append(sym)

    return selected[:TRAINING_SIZE]


def _default_universe():
    """Return a universe dict based on DEFAULT_SYMBOLS (fallback)."""
    tiers = {sym: CAP_TIERS.get(sym, 3) for sym in DEFAULT_SYMBOLS}
    return {
        "pipeline": list(DEFAULT_SYMBOLS),
        "training": list(DEFAULT_SYMBOLS),
        "tiers": tiers,
        "version": "v0-default",
    }


def get_symbols(subset="training"):
    """Load symbols from cached universe. Falls back to DEFAULT_SYMBOLS.

    When subset="training", returns only symbols that have valid data
    for the validation split (>= 1000 bars, no extreme price ratios).
    Validated results are cached in the universe JSON to avoid re-reading
    parquet files on every call.
    """
    if os.path.exists(UNIVERSE_CACHE):
        try:
            with open(UNIVERSE_CACHE) as f:
                universe = json.load(f)
            symbols = universe.get(subset, DEFAULT_SYMBOLS)
            if symbols:
                if subset == "training":
                    # Use cached validated list if available
                    cached = universe.get("validated_training")
                    if cached:
                        return cached
                    validated = _validate_symbols(symbols, "val")
                    # Cache the result
                    universe["validated_training"] = validated
                    try:
                        with open(UNIVERSE_CACHE, "w") as f:
                            json.dump(universe, f, indent=2)
                    except OSError:
                        pass  # non-critical, just skip caching
                    return validated
                return symbols
        except (json.JSONDecodeError, KeyError):
            pass
    return list(DEFAULT_SYMBOLS)


def _validate_symbols(symbols, split="val"):
    """Filter symbols to only those with valid data for the given split."""
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (VAL_START, VAL_END),
        "test": (TEST_START, TEST_END),
    }
    if split not in splits:
        return symbols
    start_str, end_str = splits[split]
    start_ms = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_str, tz="UTC").timestamp() * 1000)

    valid = []
    for symbol in symbols:
        filepath = os.path.join(DATA_DIR, f"{symbol}_1h.parquet")
        if not os.path.exists(filepath):
            continue
        df = pd.read_parquet(filepath)
        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)
        split_df = df[mask].reset_index(drop=True)
        split_df = split_df[split_df["close"] > 0].reset_index(drop=True)
        if len(split_df) < MIN_BARS_PER_SPLIT:
            continue
        price_ratio = split_df["close"].max() / split_df["close"].min()
        if price_ratio > MAX_PRICE_RATIO:
            continue
        valid.append(symbol)
    return valid if valid else list(DEFAULT_SYMBOLS)


def save_universe(universe):
    """Save discovered universe to cache file. Clears validated cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    universe["timestamp"] = datetime.utcnow().isoformat()
    universe.pop("validated_training", None)  # force re-validation after fresh discovery
    with open(UNIVERSE_CACHE, "w") as f:
        json.dump(universe, f, indent=2)
    print(f"  Universe saved to {UNIVERSE_CACHE} (version: {universe['version']})")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BarData:
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    funding_rate: float
    history: pd.DataFrame  # last LOOKBACK_BARS bars

@dataclass
class Signal:
    symbol: str
    target_position: float   # target USD notional (signed: +long, -short)
    order_type: str = "market"

@dataclass
class PortfolioState:
    cash: float
    positions: dict          # symbol -> signed USD notional
    entry_prices: dict       # symbol -> avg entry price
    equity: float = 0.0
    timestamp: int = 0

@dataclass
class BacktestResult:
    sharpe: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    num_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    annual_turnover: float = 0.0
    backtest_seconds: float = 0.0
    equity_curve: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
CRYPTOCOMPARE_URL = "https://min-api.cryptocompare.com/data/v2/histohour"

def _download_cryptocompare_candles(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download hourly OHLCV from CryptoCompare (no geo-restrictions)."""
    all_rows = []
    # CryptoCompare uses 'toTs' (end timestamp in seconds) and returns up to 2000 bars
    current_end = end_ms // 1000
    start_s = start_ms // 1000

    while current_end > start_s:
        params = {
            "fsym": symbol,
            "tsym": "USD",
            "limit": 2000,
            "toTs": current_end,
        }
        resp = requests.get(CRYPTOCOMPARE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        bars = data.get("Data", {}).get("Data", [])
        if not bars:
            break
        for bar in bars:
            ts_s = bar["time"]
            if ts_s < start_s:
                continue
            all_rows.append({
                "timestamp": ts_s * 1000,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar.get("volumefrom", 0)),
            })
        # Move window back
        earliest = bars[0]["time"]
        if earliest >= current_end:
            break
        current_end = earliest - 1
        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def _download_hl_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download funding rate history from Hyperliquid."""
    all_rows = []
    current = start_ms
    while current < end_ms:
        body = {
            "type": "fundingHistory",
            "coin": symbol,
            "startTime": current,
            "endTime": min(current + 30 * 24 * 3600 * 1000, end_ms),
        }
        try:
            resp = requests.post(HL_INFO_URL, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for row in data:
                all_rows.append({
                    "timestamp": int(row["time"]),
                    "funding_rate": float(row["fundingRate"]),
                })
            current = int(data[-1]["time"]) + 1
        except Exception:
            break
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    return pd.DataFrame(all_rows)


def _download_hl_candles(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Download OHLCV candles from Hyperliquid."""
    all_rows = []
    current = start_ms
    chunk_ms = 30 * 24 * 3600 * 1000  # 30 days
    while current < end_ms:
        body = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": min(current + chunk_ms, end_ms),
            }
        }
        try:
            resp = requests.post(HL_INFO_URL, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                current += chunk_ms
                continue
            for row in data:
                all_rows.append({
                    "timestamp": int(row["t"]),
                    "open": float(row["o"]),
                    "high": float(row["h"]),
                    "low": float(row["l"]),
                    "close": float(row["c"]),
                    "volume": float(row["v"]),
                })
            current = int(data[-1]["t"]) + 3600 * 1000
        except Exception:
            current += chunk_ms
        time.sleep(0.2)
    return pd.DataFrame(all_rows)


def download_data(symbols=None):
    """Download historical OHLCV + funding data for all symbols."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if symbols is None:
        symbols = get_symbols("pipeline")

    start_ms = int(pd.Timestamp(TRAIN_START, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(TEST_END, tz="UTC").timestamp() * 1000)

    total = len(symbols)
    for idx, symbol in enumerate(symbols, 1):
        filepath = os.path.join(DATA_DIR, f"{symbol}_1h.parquet")
        if os.path.exists(filepath):
            existing = pd.read_parquet(filepath)
            print(f"  [{idx}/{total}] {symbol}: already have {len(existing)} bars")
            continue

        print(f"  [{idx}/{total}] {symbol}: downloading candles from CryptoCompare...")

        # Use CryptoCompare for reliable historical OHLCV (no geo-restrictions)
        df = _download_cryptocompare_candles(symbol, start_ms, end_ms)
        if len(df) < 100:
            print(f"  {symbol}: CryptoCompare insufficient ({len(df)} bars), trying HL...")
            df = _download_hl_candles(symbol, "1h", start_ms, end_ms)

        if df.empty:
            print(f"  {symbol}: NO DATA AVAILABLE, skipping")
            continue

        # Download funding rates
        print(f"  {symbol}: downloading funding rates...")
        funding = _download_hl_funding(symbol, start_ms, end_ms)

        # Merge
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if not funding.empty:
            funding = funding.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            # Merge nearest — funding is every 8h, candles every 1h
            df = pd.merge_asof(df, funding, on="timestamp", direction="backward")
        if "funding_rate" not in df.columns:
            df["funding_rate"] = 0.0
        df["funding_rate"] = df["funding_rate"].fillna(0.0)

        df.to_parquet(filepath, index=False)
        print(f"  {symbol}: saved {len(df)} bars to {filepath}")


def load_data(split: str = "val", symbols=None) -> dict:
    """Load OHLCV+funding data for the given split. Returns {symbol: DataFrame}.

    Args:
        split: "train", "val", or "test"
        symbols: list of symbols to load. Defaults to get_symbols("training").
    """
    splits = {
        "train": (TRAIN_START, TRAIN_END),
        "val": (VAL_START, VAL_END),
        "test": (TEST_START, TEST_END),
    }
    assert split in splits, f"split must be one of {list(splits.keys())}"
    start_str, end_str = splits[split]
    start_ms = int(pd.Timestamp(start_str, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_str, tz="UTC").timestamp() * 1000)

    if symbols is None:
        symbols = get_symbols("training")

    result = {}
    for symbol in symbols:
        filepath = os.path.join(DATA_DIR, f"{symbol}_1h.parquet")
        if not os.path.exists(filepath):
            continue
        df = pd.read_parquet(filepath)
        mask = (df["timestamp"] >= start_ms) & (df["timestamp"] < end_ms)
        split_df = df[mask].reset_index(drop=True)
        # Filter out zero-price bars (CryptoCompare returns 0s before coin existed)
        split_df = split_df[split_df["close"] > 0].reset_index(drop=True)
        # Reject coins with extreme price ratio (bad data or pre-launch artifacts)
        if len(split_df) > 0:
            price_ratio = split_df["close"].max() / split_df["close"].min()
            if price_ratio > MAX_PRICE_RATIO:
                print(f"  Warning: {symbol} has extreme price ratio ({price_ratio:.0f}x) in {split} split, excluding")
                continue
        if len(split_df) < MIN_BARS_PER_SPLIT:
            if len(split_df) > 0:
                print(f"  Warning: {symbol} has only {len(split_df)} bars in {split} split, excluding")
            continue
        result[symbol] = split_df
    return result

# ---------------------------------------------------------------------------
# Backtesting engine (DO NOT CHANGE)
# ---------------------------------------------------------------------------

def run_backtest(strategy, data: dict, intra_bar_sim: bool = False) -> BacktestResult:
    """
    Run strategy over data. Returns BacktestResult with full metrics.
    Enforces TIME_BUDGET.
    When intra_bar_sim=True, uses OHLC high/low for liquidation checks,
    stop-loss accuracy, and volatility-adjusted slippage.
    """
    t_start = time.time()

    # Build unified timeline
    all_timestamps = set()
    for symbol, df in data.items():
        all_timestamps.update(df["timestamp"].tolist())
    timestamps = sorted(all_timestamps)

    if not timestamps:
        return BacktestResult()

    # Index data by symbol: sort by timestamp, use positional pointers
    symbol_dfs = {}
    symbol_ts = {}  # timestamp arrays for O(1) compare
    symbol_ptrs = {}
    for symbol, df in data.items():
        sdf = df.sort_values("timestamp").reset_index(drop=True)
        symbol_dfs[symbol] = sdf
        symbol_ts[symbol] = sdf["timestamp"].values
        symbol_ptrs[symbol] = 0

    def _calc_unrealized_pnl(positions, entry_prices, bar_data):
        pnl = 0.0
        for sym, pos in positions.items():
            if sym in bar_data:
                entry = entry_prices.get(sym, bar_data[sym].close)
                if entry > 0:
                    pnl += pos * (bar_data[sym].close - entry) / entry
        return pnl

    def _calc_unrealized_pnl_at_prices(positions, entry_prices, prices):
        """Unrealized PnL at arbitrary prices (dict: symbol -> price)."""
        pnl = 0.0
        for sym, pos in positions.items():
            if sym in prices:
                entry = entry_prices.get(sym, prices[sym])
                if entry > 0:
                    pnl += pos * (prices[sym] - entry) / entry
        return pnl

    def _vol_adjusted_slippage(bar_range_pct, base_bps=SLIPPAGE_BPS):
        """Scale slippage with bar range — wider bars = more slippage."""
        range_mult = max(1.0, bar_range_pct / 0.005)
        return base_bps * range_mult

    # Portfolio state
    portfolio = PortfolioState(
        cash=INITIAL_CAPITAL,
        positions={},
        entry_prices={},
        equity=INITIAL_CAPITAL,
        timestamp=0,
    )

    equity_curve = [INITIAL_CAPITAL]
    hourly_returns = []
    trade_log = []
    total_volume = 0.0
    prev_equity = INITIAL_CAPITAL

    for ts in timestamps:
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET:
            break

        portfolio.timestamp = ts

        # Build bar data using pointer iteration + iloc pre-slicing
        bar_data = {}
        for symbol in data:
            ptr = symbol_ptrs[symbol]
            ts_arr = symbol_ts[symbol]
            if ptr >= len(ts_arr):
                continue
            if ts_arr[ptr] != ts:
                continue

            sdf = symbol_dfs[symbol]
            row = sdf.iloc[ptr]

            # iloc window for history (replaces list-of-dicts + DataFrame construction)
            start = max(0, ptr - LOOKBACK_BARS + 1)
            hist_df = sdf.iloc[start:ptr + 1]

            bar_data[symbol] = BarData(
                symbol=symbol,
                timestamp=ts,
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                funding_rate=row.get("funding_rate", 0.0) if "funding_rate" in sdf.columns else 0.0,
                history=hist_df,
            )
            symbol_ptrs[symbol] = ptr + 1

        if not bar_data:
            continue

        # Update portfolio equity (mark-to-market)
        unrealized_pnl = _calc_unrealized_pnl(portfolio.positions, portfolio.entry_prices, bar_data)
        portfolio.equity = portfolio.cash + sum(abs(v) for v in portfolio.positions.values()) + unrealized_pnl

        # Apply funding rates (on open positions)
        for sym, pos_notional in list(portfolio.positions.items()):
            if sym in bar_data:
                fr = bar_data[sym].funding_rate
                # Funding: longs pay when positive, shorts receive
                # Applied every 8h, but we have hourly bars so scale by 1/8
                funding_payment = pos_notional * fr / 8.0
                portfolio.cash -= funding_payment

        # Get signals from strategy
        try:
            signals = strategy.on_bar(bar_data, portfolio)
        except Exception:
            signals = []

        # Execute signals
        for sig in (signals or []):
            if sig.symbol not in bar_data:
                continue

            current_price = bar_data[sig.symbol].close
            current_pos = portfolio.positions.get(sig.symbol, 0.0)
            delta = sig.target_position - current_pos

            if abs(delta) < 1.0:  # < $1 change, skip
                continue

            # Check leverage constraint
            new_positions = dict(portfolio.positions)
            new_positions[sig.symbol] = sig.target_position
            total_exposure = sum(abs(v) for v in new_positions.values())
            if total_exposure > portfolio.equity * MAX_LEVERAGE:
                continue

            # Apply slippage and fees
            if intra_bar_sim and sig.symbol in bar_data:
                bd_sig = bar_data[sig.symbol]
                br_pct = (bd_sig.high - bd_sig.low) / current_price if current_price > 0 else 0
                eff_bps = _vol_adjusted_slippage(br_pct)
            else:
                eff_bps = SLIPPAGE_BPS
            slippage = current_price * eff_bps / 10000
            fee_rate = TAKER_FEE
            if delta > 0:  # buying
                exec_price = current_price + slippage
            else:  # selling
                exec_price = current_price - slippage

            fee = abs(delta) * fee_rate
            portfolio.cash -= fee
            total_volume += abs(delta)

            # Update position
            if sig.target_position == 0:
                # Closing position — realize PnL
                if sig.symbol in portfolio.entry_prices:
                    entry = portfolio.entry_prices[sig.symbol]
                    if entry > 0:
                        pnl = current_pos * (exec_price - entry) / entry
                        portfolio.cash += abs(current_pos) + pnl
                    del portfolio.entry_prices[sig.symbol]
                if sig.symbol in portfolio.positions:
                    del portfolio.positions[sig.symbol]
                trade_log.append(("close", sig.symbol, delta, exec_price, pnl if 'pnl' in dir() else 0))
            else:
                if current_pos == 0:
                    # Opening new position
                    portfolio.cash -= abs(sig.target_position)
                    portfolio.positions[sig.symbol] = sig.target_position
                    portfolio.entry_prices[sig.symbol] = exec_price
                    trade_log.append(("open", sig.symbol, delta, exec_price, 0))
                else:
                    # Modifying position
                    old_notional = abs(current_pos)
                    old_entry = portfolio.entry_prices.get(sig.symbol, exec_price)
                    # Realize PnL on reduced portion
                    if abs(sig.target_position) < abs(current_pos):
                        reduced = abs(current_pos) - abs(sig.target_position)
                        if old_entry > 0:
                            pnl = (current_pos / abs(current_pos)) * reduced * (exec_price - old_entry) / old_entry
                        else:
                            pnl = 0
                        portfolio.cash += reduced + pnl
                    elif abs(sig.target_position) > abs(current_pos):
                        added = abs(sig.target_position) - abs(current_pos)
                        portfolio.cash -= added
                        # Weighted average entry
                        if old_notional + added > 0:
                            new_entry = (old_entry * old_notional + exec_price * added) / (old_notional + added)
                            portfolio.entry_prices[sig.symbol] = new_entry
                    portfolio.positions[sig.symbol] = sig.target_position
                    trade_log.append(("modify", sig.symbol, delta, exec_price, 0))

        # ---------------------------------------------------------------
        # Intra-bar worst-case simulation (OHLC high/low checks)
        # ---------------------------------------------------------------
        if intra_bar_sim and portfolio.positions:
            # Step A: build adverse prices per position
            adverse_prices = {}
            for sym, pos in portfolio.positions.items():
                if sym in bar_data:
                    if pos > 0:
                        adverse_prices[sym] = bar_data[sym].low
                    else:
                        adverse_prices[sym] = bar_data[sym].high

            # Step B: portfolio-level liquidation at adverse extreme
            if adverse_prices:
                adverse_pnl = _calc_unrealized_pnl_at_prices(
                    portfolio.positions, portfolio.entry_prices, adverse_prices)
                adverse_equity = (portfolio.cash
                    + sum(abs(v) for v in portfolio.positions.values())
                    + adverse_pnl)

                if adverse_equity < INITIAL_CAPITAL * MAINTENANCE_MARGIN_PCT / 100:
                    # Force-close all positions at adverse prices
                    for sym in list(portfolio.positions.keys()):
                        if sym in adverse_prices:
                            liq_price = adverse_prices[sym]
                            pos = portfolio.positions[sym]
                            entry = portfolio.entry_prices.get(sym, liq_price)
                            pnl = pos * (liq_price - entry) / entry if entry > 0 else 0
                            portfolio.cash += abs(pos) + pnl
                            total_volume += abs(pos)
                            trade_log.append(("liquidation", sym, -pos, liq_price, pnl))
                        portfolio.positions.pop(sym, None)
                        portfolio.entry_prices.pop(sym, None)
                    # Notify strategy
                    if hasattr(strategy, 'on_liquidation'):
                        try:
                            strategy.on_liquidation(list(adverse_prices.keys()))
                        except Exception:
                            pass

            # Step C: individual stop-loss checks at adverse extreme
            if portfolio.positions and hasattr(strategy, 'get_stop_prices'):
                try:
                    stop_prices = strategy.get_stop_prices()
                except Exception:
                    stop_prices = {}
                for sym in list(portfolio.positions.keys()):
                    if sym not in stop_prices or sym not in bar_data:
                        continue
                    pos = portfolio.positions[sym]
                    stop = stop_prices[sym]
                    bd_s = bar_data[sym]
                    hit = (pos > 0 and bd_s.low <= stop) or (pos < 0 and bd_s.high >= stop)
                    if not hit:
                        continue
                    # Execute at stop price + vol-adjusted slippage
                    br_pct = (bd_s.high - bd_s.low) / bd_s.close if bd_s.close > 0 else 0
                    slip = stop * _vol_adjusted_slippage(br_pct) / 10000
                    exec_price = stop - slip if pos > 0 else stop + slip
                    entry = portfolio.entry_prices.get(sym, exec_price)
                    pnl = pos * (exec_price - entry) / entry if entry > 0 else 0
                    portfolio.cash += abs(pos) + pnl
                    fee = abs(pos) * TAKER_FEE
                    portfolio.cash -= fee
                    total_volume += abs(pos)
                    trade_log.append(("stop_intrabar", sym, -pos, exec_price, pnl))
                    portfolio.positions.pop(sym, None)
                    portfolio.entry_prices.pop(sym, None)
                    # Notify strategy to clear internal state
                    if hasattr(strategy, 'on_stop_hit'):
                        try:
                            strategy.on_stop_hit(sym)
                        except Exception:
                            pass

        # Recalculate equity after trades (and intra-bar sim)
        unrealized_pnl = _calc_unrealized_pnl(portfolio.positions, portfolio.entry_prices, bar_data)
        current_equity = portfolio.cash + sum(abs(v) for v in portfolio.positions.values()) + unrealized_pnl
        equity_curve.append(current_equity)

        # Hourly return
        if prev_equity > 0:
            hourly_returns.append((current_equity - prev_equity) / prev_equity)
        prev_equity = current_equity

        # Liquidation check
        margin_pct = MAINTENANCE_MARGIN_PCT if intra_bar_sim else 1.0
        if current_equity < INITIAL_CAPITAL * margin_pct / 100:
            break

    t_end = time.time()

    # Compute metrics
    returns = np.array(hourly_returns) if hourly_returns else np.array([0.0])
    eq = np.array(equity_curve)

    # Sharpe ratio (annualized from hourly)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(HOURS_PER_YEAR)
    else:
        sharpe = 0.0

    # Total return
    final_equity = eq[-1] if len(eq) > 0 else INITIAL_CAPITAL
    total_return_pct = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / np.where(peak > 0, peak, 1)
    max_drawdown_pct = drawdown.max() * 100

    # Win rate and profit factor
    trade_pnls = [t[4] for t in trade_log if t[0] == "close"]
    num_trades = len(trade_log)
    if trade_pnls:
        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]
        win_rate_pct = len(wins) / len(trade_pnls) * 100 if trade_pnls else 0
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss
    else:
        win_rate_pct = 0.0
        profit_factor = 0.0

    # Annual turnover
    data_hours = len(timestamps)
    if data_hours > 0:
        annual_turnover = total_volume * (HOURS_PER_YEAR / data_hours)
    else:
        annual_turnover = 0.0

    return BacktestResult(
        sharpe=sharpe,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        num_trades=num_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        annual_turnover=annual_turnover,
        backtest_seconds=t_end - t_start,
        equity_curve=equity_curve,
        trade_log=trade_log,
    )

# ---------------------------------------------------------------------------
# Evaluation metric (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def compute_score(result: BacktestResult) -> float:
    """
    Composite risk-adjusted score (HIGHER is better).

    score = sharpe * sqrt(trade_count_factor) - drawdown_penalty - turnover_penalty

    Hard cutoffs for degenerate strategies.
    """
    # Hard cutoffs
    if result.num_trades < 10:
        return -999.0
    if result.max_drawdown_pct > 50.0:
        return -999.0
    final_equity = result.equity_curve[-1] if result.equity_curve else INITIAL_CAPITAL
    if final_equity < INITIAL_CAPITAL * 0.5:
        return -999.0

    # Trade count factor: full credit at 50+ trades
    trade_count_factor = min(result.num_trades / 50.0, 1.0)

    # Drawdown penalty: no penalty below 15%, then 5x per additional percent
    drawdown_penalty = max(0, result.max_drawdown_pct - 15.0) * 0.05

    # Turnover penalty: penalize excessive churning (>500x annual)
    turnover_ratio = result.annual_turnover / INITIAL_CAPITAL if INITIAL_CAPITAL > 0 else 0
    turnover_penalty = max(0, turnover_ratio - 500) * 0.001

    score = result.sharpe * math.sqrt(trade_count_factor) - drawdown_penalty - turnover_penalty
    return score

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autotrader")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to download (default: pipeline)")
    parser.add_argument("--discover", action="store_true", help="Print discovered universe and exit")
    parser.add_argument("--refresh", action="store_true", help="Force re-discovery (ignore cache)")
    parser.add_argument("--info", action="store_true", help="Show data completeness per symbol")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    if args.discover or args.refresh:
        print("Discovering coins from Hyperliquid...")
        universe = discover_coins()
        save_universe(universe)
        print(f"\n  Version:  {universe['version']}")
        print(f"  Pipeline ({len(universe['pipeline'])} coins): {', '.join(universe['pipeline'])}")
        print(f"  Training ({len(universe['training'])} coins): {', '.join(universe['training'])}")
        print(f"\n  Tier breakdown:")
        for tier in [1, 2, 3]:
            tier_coins = [s for s in universe["pipeline"] if universe["tiers"].get(s, 3) == tier]
            if tier_coins:
                print(f"    Tier {tier}: {', '.join(tier_coins)}")
        if args.discover:
            sys.exit(0)
        print()

    if args.info:
        symbols = args.symbols or get_symbols("pipeline")
        for split_name in ["train", "val", "test"]:
            data = load_data(split_name, symbols=symbols)
            total_bars = sum(len(df) for df in data.values())
            print(f"  {split_name}: {len(data)} symbols, {total_bars} total bars")
            for sym in symbols:
                if sym in data:
                    print(f"    {sym}: {len(data[sym])} bars")
                else:
                    filepath = os.path.join(DATA_DIR, f"{sym}_1h.parquet")
                    if not os.path.exists(filepath):
                        print(f"    {sym}: no data file")
                    else:
                        print(f"    {sym}: excluded (< 1000 bars)")
        sys.exit(0)

    print("Downloading data...")
    download_data(args.symbols)
    print()
    print("Done! Ready to backtest.")
