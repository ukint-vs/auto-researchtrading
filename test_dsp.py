"""
Unit tests for Ehlers DSP building blocks in strategy.py.

Tests edge cases: flat price, short input, division-by-zero guards,
and validates against known sinusoidal behavior.

Usage: uv run python3 -m pytest test_dsp.py -v
   or: uv run python3 test_dsp.py
"""

import numpy as np
import sys

from strategy import Strategy


def test_ehlers_eot_sinusoidal():
    """Known behavior: sinusoidal input should produce bounded oscillator output."""
    s = Strategy()
    n = 500
    t = np.arange(n, dtype=float)
    closes = 100 + 5 * np.sin(2 * np.pi * t / 50)  # 50-bar cycle
    q1, q2 = s._ehlers_eot(closes, lpperiod=6, k1=0.00, k2=0.30)
    # Output should be bounded approximately [-1, 1]
    assert np.all(np.abs(q1[120:]) <= 1.5), f"Q1 out of bounds: max={np.max(np.abs(q1[120:]))}"
    assert np.all(np.abs(q2[120:]) <= 1.5), f"Q2 out of bounds: max={np.max(np.abs(q2[120:]))}"
    # After warmup, should have non-trivial oscillation
    assert np.std(q1[120:]) > 0.01, "Q1 is flat on sinusoidal input"
    assert np.std(q2[120:]) > 0.01, "Q2 is flat on sinusoidal input"


def test_ehlers_eot_flat_price():
    """Flat price should not crash. Peak detector floor prevents division by zero."""
    s = Strategy()
    closes = np.full(500, 100.0)
    q1, q2 = s._ehlers_eot(closes, lpperiod=6, k1=0.00, k2=0.30)
    assert not np.any(np.isnan(q1)), "NaN in Q1 on flat price"
    assert not np.any(np.isnan(q2)), "NaN in Q2 on flat price"
    assert not np.any(np.isinf(q1)), "Inf in Q1 on flat price"
    assert not np.any(np.isinf(q2)), "Inf in Q2 on flat price"


def test_ehlers_eot_short_input():
    """Input shorter than warmup should not crash."""
    s = Strategy()
    closes = np.array([100.0, 101.0, 99.0])
    q1, q2 = s._ehlers_eot(closes, lpperiod=6, k1=0.00, k2=0.30)
    assert len(q1) == 3
    assert len(q2) == 3
    assert not np.any(np.isnan(q1))


def test_ehlers_eot_quotient_singularity():
    """Test with k values near singularity (k=0.99, k=-0.99)."""
    s = Strategy()
    n = 500
    t = np.arange(n, dtype=float)
    closes = 100 + 10 * np.sin(2 * np.pi * t / 30)
    q1, q2 = s._ehlers_eot(closes, lpperiod=11, k1=0.99, k2=-0.99)
    assert not np.any(np.isnan(q1)), "NaN with k=0.99"
    assert not np.any(np.isnan(q2)), "NaN with k=-0.99"
    assert not np.any(np.isinf(q1)), "Inf with k=0.99"
    assert not np.any(np.isinf(q2)), "Inf with k=-0.99"


def test_boom_hunter_returns_six_arrays():
    """_boom_hunter should return 6 arrays of correct length."""
    s = Strategy()
    closes = np.random.normal(100, 5, 500)
    closes = np.cumsum(np.random.normal(0, 1, 500)) + 100  # random walk
    result = s._boom_hunter(closes)
    assert len(result) == 6, f"Expected 6 arrays, got {len(result)}"
    for i, arr in enumerate(result):
        assert len(arr) == 500, f"Array {i} has length {len(arr)}, expected 500"
        assert not np.any(np.isnan(arr)), f"NaN in array {i}"


def test_calc_sdo_normal():
    """SDO should return values in roughly 0-100 range."""
    s = Strategy()
    n = 500
    np.random.seed(42)
    closes = np.cumsum(np.random.normal(0, 1, n)) + 100
    highs = closes + np.abs(np.random.normal(0, 0.5, n))
    lows = closes - np.abs(np.random.normal(0, 0.5, n))
    sdo, signal = s._calc_sdo(highs, lows, closes)
    assert len(sdo) == n
    assert len(signal) == n
    # After warmup, should be bounded
    # Donchian component can push SDO outside 0-100; check reasonable bounds
    assert np.min(sdo[25:]) >= -50, f"SDO too low: {np.min(sdo[25:])}"
    assert np.max(sdo[25:]) <= 150, f"SDO too high: {np.max(sdo[25:])}"


def test_calc_sdo_flat_price():
    """Flat price (zero range) should not crash."""
    s = Strategy()
    n = 500
    closes = np.full(n, 100.0)
    highs = np.full(n, 100.0)
    lows = np.full(n, 100.0)
    sdo, signal = s._calc_sdo(highs, lows, closes)
    assert not np.any(np.isnan(sdo)), "NaN in SDO on flat price"
    assert not np.any(np.isinf(sdo)), "Inf in SDO on flat price"


def test_calc_sdo_short_input():
    """Short input (below SDO lookback) should not crash."""
    s = Strategy()
    closes = np.array([100.0, 101.0, 99.0, 100.5, 98.0])
    highs = closes + 0.5
    lows = closes - 0.5
    sdo, signal = s._calc_sdo(highs, lows, closes)
    assert len(sdo) == 5
    assert not np.any(np.isnan(sdo))


def test_calc_rsi_divergence_no_crash():
    """Should not crash on random data."""
    s = Strategy()
    np.random.seed(42)
    closes = np.cumsum(np.random.normal(0, 1, 500)) + 100
    rsi, bull, bear = s._calc_rsi_divergence(closes)
    assert 0 <= rsi <= 100, f"RSI out of range: {rsi}"
    assert isinstance(bull, bool)
    assert isinstance(bear, bool)


def test_calc_rsi_divergence_short_input():
    """Short input should return defaults without crash."""
    s = Strategy()
    closes = np.array([100.0, 101.0, 99.0])
    rsi, bull, bear = s._calc_rsi_divergence(closes)
    assert rsi == 50.0
    assert bull is False
    assert bear is False


def test_calc_rsi_divergence_flat_price():
    """Flat price should not produce false divergences."""
    s = Strategy()
    closes = np.full(500, 100.0)
    rsi, bull, bear = s._calc_rsi_divergence(closes)
    assert bull is False, "False bull divergence on flat price"
    assert bear is False, "False bear divergence on flat price"


# ── aggregate_to_4h tests ──

def test_aggregate_to_4h_correct_ohlcv():
    """4h bars should have correct OHLCV from constituent 1h bars."""
    import pandas as pd
    from strategy import aggregate_to_4h

    # 8 hourly bars starting at midnight UTC → should produce 2 complete 4h bars
    base_ts = 1704067200000  # 2024-01-01 00:00 UTC (divisible by 4h)
    df = pd.DataFrame({
        "timestamp": [base_ts + i * 3600000 for i in range(8)],
        "open":   [100, 101, 102, 103, 104, 105, 106, 107],
        "high":   [110, 111, 112, 113, 114, 115, 116, 117],
        "low":    [ 90,  91,  92,  93,  94,  95,  96,  97],
        "close":  [101, 102, 103, 104, 105, 106, 107, 108],
        "volume": [ 10,  20,  30,  40,  50,  60,  70,  80],
    })
    bars = aggregate_to_4h(df)
    assert len(bars["close"]) == 2, f"Expected 2 4h bars, got {len(bars['close'])}"
    # First 4h bar: hours 0-3
    assert bars["open"][0] == 100, "First bar open should be first hour's open"
    assert bars["high"][0] == 113, "First bar high should be max of 4 highs"
    assert bars["low"][0] == 90, "First bar low should be min of 4 lows"
    assert bars["close"][0] == 104, "First bar close should be last hour's close"
    assert bars["volume"][0] == 100, "First bar volume should be sum of 4 volumes"
    # Second 4h bar: hours 4-7
    assert bars["open"][1] == 104
    assert bars["high"][1] == 117
    assert bars["low"][1] == 94
    assert bars["close"][1] == 108
    assert bars["volume"][1] == 260


def test_aggregate_to_4h_short_input():
    """Input shorter than 4 bars should return as-is without crash."""
    import pandas as pd
    from strategy import aggregate_to_4h

    df = pd.DataFrame({
        "timestamp": [1704067200000, 1704070800000],
        "open": [100, 101], "high": [110, 111],
        "low": [90, 91], "close": [101, 102], "volume": [10, 20],
    })
    bars = aggregate_to_4h(df)
    assert len(bars["close"]) == 2
    assert bars["open"][0] == 100


def test_aggregate_to_4h_partial_window():
    """5 hourly bars should produce 1 complete + 1 partial 4h bar."""
    import pandas as pd
    from strategy import aggregate_to_4h

    base_ts = 1704067200000
    df = pd.DataFrame({
        "timestamp": [base_ts + i * 3600000 for i in range(5)],
        "open":   [100, 101, 102, 103, 104],
        "high":   [110, 111, 112, 113, 114],
        "low":    [ 90,  91,  92,  93,  94],
        "close":  [101, 102, 103, 104, 105],
        "volume": [ 10,  20,  30,  40,  50],
    })
    bars = aggregate_to_4h(df)
    assert len(bars["close"]) == 2, "5h → 1 complete + 1 partial 4h bar"
    # Partial bar (hour 4 only)
    assert bars["open"][1] == 104
    assert bars["close"][1] == 105
    assert bars["volume"][1] == 50


def test_aggregate_to_4h_ratio():
    """500 hourly bars should produce ~125 4h bars."""
    import pandas as pd
    from strategy import aggregate_to_4h

    base_ts = 1704067200000  # aligned to 4h boundary
    df = pd.DataFrame({
        "timestamp": [base_ts + i * 3600000 for i in range(500)],
        "open": np.random.normal(100, 1, 500),
        "high": np.random.normal(101, 1, 500),
        "low": np.random.normal(99, 1, 500),
        "close": np.random.normal(100, 1, 500),
        "volume": np.random.uniform(10, 100, 500),
    })
    bars = aggregate_to_4h(df)
    assert len(bars["close"]) == 125, f"500h / 4 = 125 bars, got {len(bars['close'])}"


def test_aggregate_to_4h_no_nan_inf():
    """Output should never contain NaN or Inf."""
    import pandas as pd
    from strategy import aggregate_to_4h

    base_ts = 1704067200000
    np.random.seed(42)
    closes = np.cumsum(np.random.normal(0, 1, 200)) + 100
    df = pd.DataFrame({
        "timestamp": [base_ts + i * 3600000 for i in range(200)],
        "open": closes + np.random.normal(0, 0.1, 200),
        "high": closes + np.abs(np.random.normal(0, 0.5, 200)),
        "low": closes - np.abs(np.random.normal(0, 0.5, 200)),
        "close": closes,
        "volume": np.random.uniform(1, 100, 200),
    })
    bars = aggregate_to_4h(df)
    for key in ["open", "high", "low", "close", "volume"]:
        assert not np.any(np.isnan(bars[key])), f"NaN in {key}"
        assert not np.any(np.isinf(bars[key])), f"Inf in {key}"


# ── Run tests ──

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed else 0)
