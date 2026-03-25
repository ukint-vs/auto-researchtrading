"""
Champion strategy (expanded universe, score 31.36).

N-coin equal-weight ensemble with 5/9 majority vote (9 signals):
  Dynamic symbols from get_symbols("training") with 1/N weighting.
Signals: Momentum (12h), very-short momentum (6h), EMA(5/23) crossover,
  RSI(4), MACD(7/30/2), BB width compression (period=6, <93rd pctile),
  RSI divergence (lookback=14), Donchian 5-bar 70% breakout, 3-bar micro momentum.
Exits: ATR trailing stop (5.5x, tightened to 1.85x by SDO 10/14 at 85/15 extremes),
  RSI(4) mean-reversion (74/26), signal flip.

Evolution: exp251 (3 coins, 21.40) → exp110 (7 coins, 24.69) → exp112 (SDO stop, 25.05)
  → autoresearch s1: +RSI div, BB 7→5, pos 0.08→0.075 = 24.82
  → autoresearch s2: MACD→15/26/8, VOL_LOOKBACK→42, BASE_THRESHOLD→0.014 = 25.09
  → autoresearch s3: RSI(4), MACD(14/27/2), EMA(2/28), BB(3/<93pctile),
    Donchian 8-bar 68%, 3-bar micro mom, SDO_TIGHT 2.0, log returns = 32.06
  → autoresearch s4 (expanded universe, 6 coins, baseline 28.90):
    SDO_TIGHT 2.0→1.85 (28.96), EMA_SLOW 28→26 (28.99), MACD_FAST 14→12 (29.11),
    BB_PERIOD 3→5 (29.11), Donchian 8→7bar (29.38), VOL_LOOKBACK 38→42 (29.40),
    RSI entry 48/52→50/50 (29.45), SDO_OB/OS 80/20→85/15 (29.47),
    COOLDOWN 2→0 (29.91 — breakthrough, +0.44), pos 0.065→0.058 (29.98),
    EMA_SLOW 26→23 (30.20), MACD_FAST 12→7 (30.38), MACD_SLOW 27→30 (30.45),
    Donchian 7→5bar 70% (31.22), BB_PERIOD 5→6 (31.25), ATR_LB 16→12 (31.26),
    EMA_FAST 2→5 (31.36)
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from prepare import Signal, PortfolioState, BarData, get_symbols

SHORT_WINDOW = 6
MED_WINDOW = 12
MED2_WINDOW = 24
LONG_WINDOW = 36
EMA_FAST = 5
EMA_SLOW = 23
RSI_PERIOD = 8
RSI_BULL = 50
RSI_BEAR = 50
RSI_OVERBOUGHT = 74
RSI_OVERSOLD = 26

MACD_FAST = 7
MACD_SLOW = 30
MACD_SIGNAL = 2

BB_PERIOD = 6

FUNDING_LOOKBACK = 24
FUNDING_BOOST = 0.0
BASE_POSITION_PCT = 0.058
VOL_LOOKBACK = 42
TARGET_VOL = 0.015
ATR_LOOKBACK = 12
ATR_STOP_MULT = 5.5
TAKE_PROFIT_PCT = 99.0
BASE_THRESHOLD = 0.013
BTC_OPPOSE_THRESHOLD = -99.0

PYRAMID_THRESHOLD = 0.015
PYRAMID_SIZE = 0.0
CORR_LOOKBACK = 72
HIGH_CORR_THRESHOLD = 99.0

DD_REDUCE_THRESHOLD = 99.0
DD_REDUCE_SCALE = 0.5

SDO_STOCH_LEN = 10
SDO_DONCH_LEN = 14
SDO_SMOOTH_LEN = 3
SDO_OVERBOUGHT = 85
SDO_OVERSOLD = 15
SDO_TIGHT_ATR_MULT = 1.85

# Higher timeframe indicator params (used with aggregate_tf)
HTF_PERIOD = 4            # hours per bar (4, 12, or 24)
HTF_EMA_FAST = 5          # 4h EMA fast period (~20h lookback)
HTF_EMA_SLOW = 13         # 4h EMA slow period (~52h lookback)
HTF_RSI_PERIOD = 8        # 4h RSI period
HTF_RSI_OVERBOUGHT = 70   # skip entries above this
HTF_RSI_OVERSOLD = 30     # skip entries below this
HTF_TREND_LOOKBACK = 6    # bars for momentum (6 × 4h = 24h)

# Boom Hunter (Ehlers EOT trio) params
BH_INPUT_LEN = 120         # bars of closes to feed (100-bar HP cutoff + settling margin)
BH_EOT1_LP = 6             # fast oscillator period
BH_EOT2_LP = 27            # trend oscillator period
BH_EOT3_LP = 11            # extreme detector period
BH_BULL_THRESHOLD = 0.0    # trigger > threshold = bullish momentum
BH_BEAR_THRESHOLD = 0.0    # trigger < -threshold = bearish momentum
BH_EXTREME_OB = 0.8        # EOT3 q5 > this = overbought extreme
BH_EXTREME_OS = -0.8       # EOT3 q5 < this = oversold extreme
BH_TREND_WIDTH = 0.3       # |q3-q4| < this = trending (EOT2 converged)

COOLDOWN_BARS = 0
MIN_VOTES = 5  # out of 9

def ema(values, span):
    alpha = 2.0 / (span + 1)
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result

def calc_rsi(closes, period):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    rs = avg_gain / max(avg_loss, 1e-10)
    return 100 - 100 / (1 + rs)


def aggregate_tf(history, hours=4):
    """Aggregate 1h bars into higher timeframe bars. Returns dict with OHLCV numpy arrays.

    Args:
        history: DataFrame with timestamp, open, high, low, close, volume columns.
        hours: Timeframe in hours (4, 12, 24, etc.)

    Groups by UTC-aligned windows. Incomplete trailing window is included.
    With 500 1h bars: 4h→~125, 12h→~42, 24h→~21 bars.
    """
    ts = history["timestamp"].values
    o = history["open"].values
    h = history["high"].values
    l = history["low"].values
    c = history["close"].values
    v = history["volume"].values
    n = len(ts)
    if n < hours:
        return {"open": o, "high": h, "low": l, "close": c, "volume": v, "timestamp": ts}

    ms_period = hours * 3600 * 1000
    boundaries = ts // ms_period

    out_o, out_h, out_l, out_c, out_v, out_ts = [], [], [], [], [], []
    i = 0
    while i < n:
        b = boundaries[i]
        j = i
        while j < n and boundaries[j] == b:
            j += 1
        out_o.append(o[i])
        out_h.append(np.max(h[i:j]))
        out_l.append(np.min(l[i:j]))
        out_c.append(c[j - 1])
        out_v.append(np.sum(v[i:j]))
        out_ts.append(ts[i])
        i = j

    return {
        "open": np.array(out_o),
        "high": np.array(out_h),
        "low": np.array(out_l),
        "close": np.array(out_c),
        "volume": np.array(out_v),
        "timestamp": np.array(out_ts),
    }


def aggregate_to_4h(history):
    """Aggregate 1h → 4h. Convenience wrapper for backwards compat."""
    return aggregate_tf(history, hours=4)


class Strategy:
    def __init__(self, symbols=None):
        self._symbols = symbols if symbols is not None else get_symbols("training")
        self._weight = 1.0 / len(self._symbols) if self._symbols else 1.0
        self.entry_prices = {}
        self.peak_prices = {}
        self.atr_at_entry = {}
        self.btc_momentum = 0.0
        self.pyramided = {}
        self.peak_equity = 100000.0
        self.exit_bar = {}
        self.bar_count = 0
        self._current_stops = {}  # symbol -> stop price (for intra-bar sim)

    def _calc_atr(self, history, lookback):
        if len(history) < lookback + 1:
            return None
        highs = history["high"].values[-lookback:]
        lows = history["low"].values[-lookback:]
        closes = history["close"].values[-(lookback+1):-1]
        tr = np.maximum(highs - lows,
                        np.maximum(np.abs(highs - closes), np.abs(lows - closes)))
        return np.mean(tr)

    def _calc_vol(self, closes, lookback):
        if len(closes) < lookback:
            return TARGET_VOL
        log_rets = np.diff(np.log(closes[-lookback:]))
        return max(np.std(log_rets), 1e-6)

    def _calc_correlation(self, bar_data):
        if "BTC" not in bar_data or "ETH" not in bar_data:
            return 0.5
        btc_h = bar_data["BTC"].history
        eth_h = bar_data["ETH"].history
        if len(btc_h) < CORR_LOOKBACK or len(eth_h) < CORR_LOOKBACK:
            return 0.5
        btc_rets = np.diff(np.log(btc_h["close"].values[-CORR_LOOKBACK:]))
        eth_rets = np.diff(np.log(eth_h["close"].values[-CORR_LOOKBACK:]))
        if len(btc_rets) < 10:
            return 0.5
        corr = np.corrcoef(btc_rets, eth_rets)[0, 1]
        return corr if not np.isnan(corr) else 0.5

    def _calc_macd(self, closes):
        if len(closes) < MACD_SLOW + MACD_SIGNAL + 5:
            return 0.0
        fast_ema = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_FAST)
        slow_ema = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_SLOW)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, MACD_SIGNAL)
        return macd_line[-1] - signal_line[-1]

    def _calc_bb_width_pctile(self, closes, period):
        """Calculate current BB width percentile over lookback."""

        if len(closes) < period * 3:
            return 50.0
        # Rolling mean/std via sliding window (pure numpy, no pandas)
        windows = sliding_window_view(closes, period)
        sma = windows.mean(axis=1)
        std = np.sqrt(((windows - sma[:, None]) ** 2).mean(axis=1))  # ddof=0
        all_widths = np.where(sma > 0, 2 * std / sma, 0.0)
        # Match original loop: starts at i=period*2, ends at i=len-1
        # SWV index j uses closes[j:j+period], original at i uses closes[i-period:i]
        # So j = i - period. i=period*2 → j=period, i=len-1 → j=len-1-period
        widths = all_widths[period:len(closes) - period]
        if len(widths) < 2:
            return 50.0
        current_width = widths[-1]
        return 100.0 * np.sum(widths <= current_width) / len(widths)

    def _ehlers_eot(self, closes, lpperiod, k1, k2):
        """Ehlers Even-Better Trigonometric Oscillator.
        Returns (Q1, Q2) numpy arrays — quotient-normalized oscillator lines."""
        n = len(closes)
        hp = np.zeros(n)
        filt = np.zeros(n)
        peak = np.zeros(n)
        q1 = np.zeros(n)
        q2 = np.zeros(n)

        # Highpass filter coefficients (100-bar cycle cutoff)
        alpha1 = (np.cos(0.707 * 2 * np.pi / 100) +
                  np.sin(0.707 * 2 * np.pi / 100) - 1) / np.cos(0.707 * 2 * np.pi / 100)

        # Supersmoother coefficients
        a1 = np.exp(-1.414 * np.pi / lpperiod)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / lpperiod)
        c2 = b1
        c3 = -(a1 * a1)
        c1 = 1 - c2 - c3

        for i in range(2, n):
            # Highpass filter
            hp[i] = ((1 - alpha1 / 2) ** 2 * (closes[i] - 2 * closes[i - 1] + closes[i - 2])
                     + 2 * (1 - alpha1) * hp[i - 1]
                     - (1 - alpha1) ** 2 * hp[i - 2])
            # Supersmoother
            filt[i] = c1 * (hp[i] + hp[i - 1]) / 2 + c2 * filt[i - 1] + c3 * filt[i - 2]
            # Peak detector (fast attack, slow decay)
            peak[i] = max(0.991 * peak[i - 1], abs(filt[i]), 1e-10)
            # Normalized roofing filter
            x = filt[i] / peak[i]
            # Quotient normalization
            denom1 = k1 * x + 1
            denom2 = k2 * x + 1
            q1[i] = (x + k1) / denom1 if abs(denom1) > 1e-10 else 0.0
            q2[i] = (x + k2) / denom2 if abs(denom2) > 1e-10 else 0.0

        return q1, q2

    def _boom_hunter(self, closes, eot1_lp=BH_EOT1_LP, eot2_lp=BH_EOT2_LP, eot3_lp=BH_EOT3_LP):
        """Boom Hunter Pro — three EOT instances.
        Returns (trigger, q2, q3, q4, q5, q6).
        Pass closes[-BH_INPUT_LEN:] for performance (~120 bars instead of 500)."""
        eot1_q1, eot1_q2 = self._ehlers_eot(closes, eot1_lp, 0.00, 0.30)
        eot2_q3, eot2_q4 = self._ehlers_eot(closes, eot2_lp, 0.80, 0.30)
        eot3_q5, eot3_q6 = self._ehlers_eot(closes, eot3_lp, 0.99, -0.99)
        # Trigger = SMA(2) of EOT1 Q1
        trigger = np.convolve(eot1_q1, [0.5, 0.5], mode='same')
        return trigger, eot1_q2, eot2_q3, eot2_q4, eot3_q5, eot3_q6

    def _calc_sdo(self, highs, lows, closes, stoch_len=14, donch_len=20, smooth_len=3):
        """Stochastic-Donchian Oscillator. Returns (sdo_smooth, signal_line)."""

        n = len(closes)
        start = max(stoch_len, donch_len)
        if n <= start:
            return np.full(n, 50.0), np.full(n, 50.0)

        # Stochastic: rolling max/min over stoch_len (pure numpy)
        h_win = sliding_window_view(highs, stoch_len)
        l_win = sliding_window_view(lows, stoch_len)
        hh_s = h_win.max(axis=1)
        ll_s = l_win.min(axis=1)
        rng_s = hh_s - ll_s
        c_s = closes[stoch_len - 1:]
        stoch = np.where(rng_s > 1e-10, (c_s - ll_s) / rng_s * 100, 50.0)

        # Donchian: rolling max/min over donch_len (pure numpy)
        h_win_d = sliding_window_view(highs, donch_len)
        l_win_d = sliding_window_view(lows, donch_len)
        hh_d = h_win_d.max(axis=1)
        ll_d = l_win_d.min(axis=1)
        rng_d = hh_d - ll_d
        mid_d = (hh_d + ll_d) / 2
        c_d = closes[donch_len - 1:]
        donch = np.where(rng_d > 1e-10, (c_d - mid_d) / rng_d * 100, 0.0)

        # Combine: align to start index
        sdo_raw = np.full(n, 50.0)
        s_offset = start - (stoch_len - 1)
        d_offset = start - (donch_len - 1)
        sdo_raw[start:] = 0.5 * stoch[s_offset:] + 0.5 * donch[d_offset:]

        # Simple moving average smooth
        kernel = np.ones(smooth_len) / smooth_len
        sdo_smooth = np.convolve(sdo_raw, kernel, mode='same')
        full_signal = np.full(n, 50.0)
        if n > start:
            signal_line = ema(sdo_smooth[start:], 9)
            full_signal[start:start + len(signal_line)] = signal_line
        return sdo_smooth, full_signal

    def _calc_rsi_divergence(self, closes, period=8, lookback=14):
        """RSI with bull/bear divergence detection.
        Returns (rsi_value, has_bull_div, has_bear_div)."""
        rsi_val = calc_rsi(closes, period)
        has_bull_div = False
        has_bear_div = False

        if len(closes) < lookback + period + 1:
            return rsi_val, has_bull_div, has_bear_div

        # Compute RSI series over lookback window
        rsi_series = []
        for i in range(lookback):
            idx = len(closes) - lookback + i
            rsi_series.append(calc_rsi(closes[:idx + 1], period))
        rsi_arr = np.array(rsi_series)
        price_arr = closes[-lookback:]

        # Find local minima/maxima (simple: lower than both neighbors)
        for i in range(1, lookback - 1):
            # Bull divergence: price lower low, RSI higher low
            if (price_arr[i] < price_arr[i - 1] and price_arr[i] < price_arr[i + 1]):
                # This is a price trough — compare to current
                if price_arr[-1] < price_arr[i] and rsi_arr[-1] > rsi_arr[i]:
                    has_bull_div = True
            # Bear divergence: price higher high, RSI lower high
            if (price_arr[i] > price_arr[i - 1] and price_arr[i] > price_arr[i + 1]):
                if price_arr[-1] > price_arr[i] and rsi_arr[-1] < rsi_arr[i]:
                    has_bear_div = True

        return rsi_val, has_bull_div, has_bear_div

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1

        self.peak_equity = max(self.peak_equity, equity)
        current_dd = (self.peak_equity - equity) / self.peak_equity
        dd_scale = 1.0
        if current_dd > DD_REDUCE_THRESHOLD:
            dd_scale = max(DD_REDUCE_SCALE, 1.0 - (current_dd - DD_REDUCE_THRESHOLD) * 5)

        if "BTC" in bar_data and len(bar_data["BTC"].history) >= LONG_WINDOW + 1:
            btc_closes = bar_data["BTC"].history["close"].values
            self.btc_momentum = (btc_closes[-1] - btc_closes[-MED2_WINDOW]) / btc_closes[-MED2_WINDOW]

        btc_eth_corr = self._calc_correlation(bar_data)
        high_corr = btc_eth_corr > HIGH_CORR_THRESHOLD

        for symbol in self._symbols:
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            if len(bd.history) < max(LONG_WINDOW, EMA_SLOW, MACD_SLOW + MACD_SIGNAL + 5, BB_PERIOD * 3) + 1:
                continue

            closes = bd.history["close"].values
            mid = bd.close

            realized_vol = self._calc_vol(closes, VOL_LOOKBACK)
            vol_ratio = realized_vol / TARGET_VOL
            dyn_threshold = BASE_THRESHOLD * (0.3 + vol_ratio * 0.7)
            dyn_threshold = max(0.005, min(0.025, dyn_threshold))

            ret_vshort = np.log(closes[-1] / closes[-SHORT_WINDOW])
            ret_short = np.log(closes[-1] / closes[-MED_WINDOW])
            ret_med = (closes[-1] - closes[-MED2_WINDOW]) / closes[-MED2_WINDOW]
            ret_long = (closes[-1] - closes[-LONG_WINDOW]) / closes[-LONG_WINDOW]

            mom_bull = ret_short > dyn_threshold
            mom_bear = ret_short < -dyn_threshold
            vshort_bull = ret_vshort > dyn_threshold * 0.7
            vshort_bear = ret_vshort < -dyn_threshold * 0.7

            ema_fast_arr = ema(closes[-(EMA_SLOW+10):], EMA_FAST)
            ema_slow_arr = ema(closes[-(EMA_SLOW+10):], EMA_SLOW)
            ema_bull = ema_fast_arr[-1] > ema_slow_arr[-1]
            ema_bear = ema_fast_arr[-1] < ema_slow_arr[-1]

            rsi = calc_rsi(closes, 4)
            rsi_bull = rsi > RSI_BULL
            rsi_bear = rsi < RSI_BEAR

            macd_hist = self._calc_macd(closes)
            macd_bull = macd_hist > 0
            macd_bear = macd_hist < 0

            # BB width: low percentile = compression = pending breakout
            bb_pctile = self._calc_bb_width_pctile(closes, BB_PERIOD)
            bb_compressed = bb_pctile < 93  # Below 90th percentile = compressed

            # RSI divergence as 7th additive signal
            _, has_bull_div, has_bear_div = self._calc_rsi_divergence(closes, RSI_PERIOD, 14)

            donch_high = np.max(closes[-5:-1])
            donch_low = np.min(closes[-5:-1])
            donch_range = donch_high - donch_low
            donch_bull = closes[-1] >= donch_low + donch_range * 0.70
            donch_bear = closes[-1] <= donch_low + donch_range * 0.30
            # 3-bar micro momentum
            micro_ret = np.log(closes[-1] / closes[-3])
            micro_bull = micro_ret > 0
            micro_bear = micro_ret < 0
            bull_votes = sum([mom_bull, vshort_bull, ema_bull, rsi_bull, macd_bull, bb_compressed, has_bull_div, donch_bull, micro_bull])
            bear_votes = sum([mom_bear, vshort_bear, ema_bear, rsi_bear, macd_bear, bb_compressed, has_bear_div, donch_bear, micro_bear])

            btc_confirm = True
            if symbol != "BTC":
                if bull_votes >= MIN_VOTES and self.btc_momentum < BTC_OPPOSE_THRESHOLD:
                    btc_confirm = False
                if bear_votes >= MIN_VOTES and self.btc_momentum > -BTC_OPPOSE_THRESHOLD:
                    btc_confirm = False

            bullish = bull_votes >= MIN_VOTES and btc_confirm
            bearish = bear_votes >= MIN_VOTES and btc_confirm

            in_cooldown = (self.bar_count - self.exit_bar.get(symbol, -999)) < COOLDOWN_BARS

            vol_scale = 1.0
            weight = self._weight
            mom_strength = abs(ret_short) / dyn_threshold
            strength_scale = 1.0
            size = equity * BASE_POSITION_PCT * weight * vol_scale * strength_scale * dd_scale

            funding_rates = bd.history["funding_rate"].values[-FUNDING_LOOKBACK:]
            avg_funding = np.mean(funding_rates) if len(funding_rates) >= FUNDING_LOOKBACK else 0.0

            current_pos = portfolio.positions.get(symbol, 0.0)
            target = current_pos

            if current_pos == 0:
                if not in_cooldown:
                    funding_mult = 1.0
                    if bullish:
                        if avg_funding < 0:
                            funding_mult = 1.0 + FUNDING_BOOST
                        target = size * funding_mult
                        self.pyramided[symbol] = False
                    elif bearish:
                        if avg_funding > 0:
                            funding_mult = 1.0 + FUNDING_BOOST
                        target = -size * funding_mult
                        self.pyramided[symbol] = False
            else:
                if symbol in self.entry_prices and not self.pyramided.get(symbol, True):
                    entry = self.entry_prices[symbol]
                    pnl = (mid - entry) / entry
                    if current_pos < 0:
                        pnl = -pnl
                    if pnl > PYRAMID_THRESHOLD:
                        if current_pos > 0 and bullish:
                            target = current_pos + size * PYRAMID_SIZE
                            self.pyramided[symbol] = True
                        elif current_pos < 0 and bearish:
                            target = current_pos - size * PYRAMID_SIZE
                            self.pyramided[symbol] = True

                atr = self._calc_atr(bd.history, ATR_LOOKBACK)
                if atr is None:
                    atr = self.atr_at_entry.get(symbol, mid * 0.02)

                if symbol not in self.peak_prices:
                    self.peak_prices[symbol] = mid

                # SDO-adaptive ATR stop: tighten when SDO at extremes
                highs_h = bd.history["high"].values
                lows_h = bd.history["low"].values
                sdo_val, _ = self._calc_sdo(highs_h, lows_h, closes,
                                            SDO_STOCH_LEN, SDO_DONCH_LEN, SDO_SMOOTH_LEN)
                atr_mult = ATR_STOP_MULT
                if current_pos > 0 and sdo_val[-1] > SDO_OVERBOUGHT:
                    atr_mult = SDO_TIGHT_ATR_MULT
                elif current_pos < 0 and sdo_val[-1] < SDO_OVERSOLD:
                    atr_mult = SDO_TIGHT_ATR_MULT



                if current_pos > 0:
                    self.peak_prices[symbol] = max(self.peak_prices[symbol], mid)
                    stop = self.peak_prices[symbol] - atr_mult * atr
                    if mid < stop:
                        target = 0.0
                else:
                    self.peak_prices[symbol] = min(self.peak_prices[symbol], mid)
                    stop = self.peak_prices[symbol] + atr_mult * atr
                    if mid > stop:
                        target = 0.0
                self._current_stops[symbol] = stop

                if symbol in self.entry_prices:
                    entry = self.entry_prices[symbol]
                    pnl = (mid - entry) / entry
                    if current_pos < 0:
                        pnl = -pnl
                    if pnl > TAKE_PROFIT_PCT:
                        target = 0.0

                if current_pos > 0 and rsi > RSI_OVERBOUGHT:
                    target = 0.0
                elif current_pos < 0 and rsi < RSI_OVERSOLD:
                    target = 0.0

                if current_pos > 0 and bearish and not in_cooldown:
                    target = -size
                elif current_pos < 0 and bullish and not in_cooldown:
                    target = size

            if abs(target - current_pos) > 1.0:
                signals.append(Signal(symbol=symbol, target_position=target))
                if target != 0 and current_pos == 0:
                    self.entry_prices[symbol] = mid
                    self.peak_prices[symbol] = mid
                    self.atr_at_entry[symbol] = self._calc_atr(bd.history, ATR_LOOKBACK) or mid * 0.02
                elif target == 0:
                    self.entry_prices.pop(symbol, None)
                    self.peak_prices.pop(symbol, None)
                    self.atr_at_entry.pop(symbol, None)
                    self.pyramided.pop(symbol, None)
                    self._current_stops.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                elif (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol] = mid
                    self.peak_prices[symbol] = mid
                    self.atr_at_entry[symbol] = self._calc_atr(bd.history, ATR_LOOKBACK) or mid * 0.02
                    self.pyramided[symbol] = False

        return signals

    def get_stop_prices(self):
        """Current trailing stop prices for open positions (for intra-bar sim)."""
        return dict(self._current_stops)

    def on_liquidation(self, symbols):
        """Called by backtester when intra-bar liquidation is triggered."""
        for sym in symbols:
            self.entry_prices.pop(sym, None)
            self.peak_prices.pop(sym, None)
            self.atr_at_entry.pop(sym, None)
            self.pyramided.pop(sym, None)
            self._current_stops.pop(sym, None)
            self.exit_bar[sym] = self.bar_count

    def on_stop_hit(self, symbol):
        """Called by backtester when intra-bar stop-loss triggers."""
        self.entry_prices.pop(symbol, None)
        self.peak_prices.pop(symbol, None)
        self.atr_at_entry.pop(symbol, None)
        self.pyramided.pop(symbol, None)
        self._current_stops.pop(symbol, None)
        self.exit_bar[symbol] = self.bar_count
