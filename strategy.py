"""
Champion strategy — live-realistic (expanded universe).

N-coin equal-weight ensemble with 5/8 majority vote (8 signals):
  Dynamic symbols from get_symbols("training") with 1/N weighting.
Signals: Momentum (12h), very-short momentum (6h), EMA(3/23) crossover,
  RSI(4) entry (45/55), MACD(7/34/2), RSI divergence (lookback=14),
  Donchian 8-bar 60% breakout, GGOSC momentum (3/10).
Exits: GGOSC TP1 (0.8x ATR from entry), SDO-tightened trailing stop
  (1.85x ATR when SDO 8/14 at 85/15 extremes), RSI(4) mean-reversion
  (78/22), signal flip.
Live constraints: COOLDOWN=1, MIN_ENTRY_MOVE=15bps fee buffer.

Score: gate1 27.33 / gate2 20.67 / ratio 0.76
OOS: test 29.35 (ratio 0.78), train 23.69 (ratio 0.78)
Evolution: s1-s3 (32.06 pre-realism) → s4 live-realism cleanup (22.38)
  → s5 parameter tuning (24.85) → s6 GGOSC integration (27.33)
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from prepare import Signal, PortfolioState, BarData, get_symbols

# Momentum windows
SHORT_WINDOW = 6
MED_WINDOW = 12

# EMA crossover
EMA_FAST = 3
EMA_SLOW = 23

# RSI
RSI_PERIOD = 8          # for divergence
RSI_BULL = 45           # entry threshold
RSI_BEAR = 55
RSI_OVERBOUGHT = 78     # exit threshold
RSI_OVERSOLD = 22

# MACD
MACD_FAST = 7
MACD_SLOW = 34
MACD_SIGNAL = 2

# Position sizing
BASE_POSITION_PCT = 0.058
VOL_LOOKBACK = 50
TARGET_VOL = 0.015
BASE_THRESHOLD = 0.013

# SDO stop tightening
ATR_LOOKBACK = 12
SDO_STOCH_LEN = 8
SDO_DONCH_LEN = 14
SDO_SMOOTH_LEN = 3
SDO_OVERBOUGHT = 85
SDO_OVERSOLD = 15
SDO_TIGHT_ATR_MULT = 1.85

# GGOSC oscillator (entry confirmation + TP exits)
GGOSC_FAST = 3
GGOSC_SLOW = 10
GGOSC_SIGNAL = 3
GGOSC_TP1_MULT = 0.8       # ATR multiple for take-profit

# Trade management
COOLDOWN_BARS = 1
MIN_VOTES = 5
MIN_ENTRY_MOVE = 0.0015


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


class Strategy:
    def __init__(self, symbols=None):
        self._symbols = symbols if symbols is not None else get_symbols("training")
        self._weight = 1.0 / len(self._symbols) if self._symbols else 1.0
        self.entry_prices = {}
        self.peak_prices = {}
        self.atr_at_entry = {}
        self.exit_bar = {}
        self.bar_count = 0
        self._current_stops = {}

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

    def _calc_macd(self, closes):
        if len(closes) < MACD_SLOW + MACD_SIGNAL + 5:
            return 0.0
        fast_ema = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_FAST)
        slow_ema = ema(closes[-(MACD_SLOW + MACD_SIGNAL + 5):], MACD_SLOW)
        macd_line = fast_ema - slow_ema
        signal_line = ema(macd_line, MACD_SIGNAL)
        return macd_line[-1] - signal_line[-1]

    def _calc_sdo(self, highs, lows, closes):
        """Stochastic-Donchian Oscillator. Returns (sdo_smooth, signal_line)."""
        n = len(closes)
        start = max(SDO_STOCH_LEN, SDO_DONCH_LEN)
        if n <= start:
            return np.full(n, 50.0), np.full(n, 50.0)

        h_win = sliding_window_view(highs, SDO_STOCH_LEN)
        l_win = sliding_window_view(lows, SDO_STOCH_LEN)
        hh_s = h_win.max(axis=1)
        ll_s = l_win.min(axis=1)
        rng_s = hh_s - ll_s
        c_s = closes[SDO_STOCH_LEN - 1:]
        stoch = np.where(rng_s > 1e-10, (c_s - ll_s) / rng_s * 100, 50.0)

        h_win_d = sliding_window_view(highs, SDO_DONCH_LEN)
        l_win_d = sliding_window_view(lows, SDO_DONCH_LEN)
        hh_d = h_win_d.max(axis=1)
        ll_d = l_win_d.min(axis=1)
        rng_d = hh_d - ll_d
        mid_d = (hh_d + ll_d) / 2
        c_d = closes[SDO_DONCH_LEN - 1:]
        donch = np.where(rng_d > 1e-10, (c_d - mid_d) / rng_d * 100, 0.0)

        sdo_raw = np.full(n, 50.0)
        s_offset = start - (SDO_STOCH_LEN - 1)
        d_offset = start - (SDO_DONCH_LEN - 1)
        sdo_raw[start:] = 0.5 * stoch[s_offset:] + 0.5 * donch[d_offset:]

        kernel = np.ones(SDO_SMOOTH_LEN) / SDO_SMOOTH_LEN
        sdo_smooth = np.convolve(sdo_raw, kernel, mode='same')
        full_signal = np.full(n, 50.0)
        if n > start:
            signal_line = ema(sdo_smooth[start:], 9)
            full_signal[start:start + len(signal_line)] = signal_line
        return sdo_smooth, full_signal

    def _calc_ggosc(self, highs, lows, closes):
        """GGOSC momentum oscillator: (EMA_fast - EMA_slow) / ATR of midpoint.
        Returns oscillator value (positive = bullish, negative = bearish)."""
        n = len(closes)
        if n < GGOSC_SLOW + GGOSC_SIGNAL + 2:
            return 0.0
        mid = (highs + lows) / 2.0
        buf = mid[-(GGOSC_SLOW + GGOSC_SIGNAL + 5):]
        ema_f = ema(buf, GGOSC_FAST)
        ema_s = ema(buf, GGOSC_SLOW)
        tr = np.maximum(highs[-len(buf):] - lows[-len(buf):],
                        np.maximum(np.abs(highs[-len(buf):] - np.roll(closes[-len(buf):], 1)),
                                   np.abs(lows[-len(buf):] - np.roll(closes[-len(buf):], 1))))
        tr[0] = highs[-len(buf)] - lows[-len(buf)]
        atr_arr = ema(tr, ATR_LOOKBACK)
        atr_arr = np.where(atr_arr > 1e-10, atr_arr, 1e-10)
        osc_line = (ema_f - ema_s) / atr_arr
        return float(osc_line[-1])

    def _calc_rsi_divergence(self, closes):
        """RSI with bull/bear divergence detection."""
        rsi_val = calc_rsi(closes, RSI_PERIOD)
        has_bull_div = False
        has_bear_div = False
        lookback = 14

        if len(closes) < lookback + RSI_PERIOD + 1:
            return rsi_val, has_bull_div, has_bear_div

        rsi_series = []
        for i in range(lookback):
            idx = len(closes) - lookback + i
            rsi_series.append(calc_rsi(closes[:idx + 1], RSI_PERIOD))
        rsi_arr = np.array(rsi_series)
        price_arr = closes[-lookback:]

        for i in range(1, lookback - 1):
            if price_arr[i] < price_arr[i - 1] and price_arr[i] < price_arr[i + 1]:
                if price_arr[-1] < price_arr[i] and rsi_arr[-1] > rsi_arr[i]:
                    has_bull_div = True
            if price_arr[i] > price_arr[i - 1] and price_arr[i] > price_arr[i + 1]:
                if price_arr[-1] > price_arr[i] and rsi_arr[-1] < rsi_arr[i]:
                    has_bear_div = True

        return rsi_val, has_bull_div, has_bear_div

    def on_bar(self, bar_data, portfolio):
        signals = []
        equity = portfolio.equity if portfolio.equity > 0 else portfolio.cash
        self.bar_count += 1
        min_history = MACD_SLOW + MACD_SIGNAL + 6

        for symbol in self._symbols:
            if symbol not in bar_data:
                continue
            bd = bar_data[symbol]
            if len(bd.history) < min_history:
                continue

            closes = bd.history["close"].values
            mid = bd.close

            # Dynamic threshold based on realized volatility
            realized_vol = self._calc_vol(closes, VOL_LOOKBACK)
            vol_ratio = realized_vol / TARGET_VOL
            dyn_threshold = BASE_THRESHOLD * (0.3 + vol_ratio * 0.7)
            dyn_threshold = max(0.005, min(0.025, dyn_threshold))

            # Momentum signals
            ret_vshort = np.log(closes[-1] / closes[-SHORT_WINDOW])
            ret_short = np.log(closes[-1] / closes[-MED_WINDOW])

            mom_bull = ret_short > dyn_threshold
            mom_bear = ret_short < -dyn_threshold
            vshort_bull = ret_vshort > dyn_threshold * 0.6
            vshort_bear = ret_vshort < -dyn_threshold * 0.6

            # EMA crossover
            ema_fast_arr = ema(closes[-(EMA_SLOW+10):], EMA_FAST)
            ema_slow_arr = ema(closes[-(EMA_SLOW+10):], EMA_SLOW)
            ema_bull = ema_fast_arr[-1] > ema_slow_arr[-1]
            ema_bear = ema_fast_arr[-1] < ema_slow_arr[-1]

            # RSI entry
            rsi = calc_rsi(closes, 4)
            rsi_bull = rsi > RSI_BULL
            rsi_bear = rsi < RSI_BEAR

            # MACD histogram
            macd_hist = self._calc_macd(closes)
            macd_bull = macd_hist > 0
            macd_bear = macd_hist < 0

            # RSI divergence
            _, has_bull_div, has_bear_div = self._calc_rsi_divergence(closes)

            # Donchian breakout
            donch_high = np.max(closes[-8:-1])
            donch_low = np.min(closes[-8:-1])
            donch_range = donch_high - donch_low
            donch_bull = closes[-1] >= donch_low + donch_range * 0.60
            donch_bear = closes[-1] <= donch_low + donch_range * 0.40

            # GGOSC oscillator as 8th signal
            highs_h = bd.history["high"].values
            lows_h = bd.history["low"].values
            ggosc_val = self._calc_ggosc(highs_h, lows_h, closes)
            ggosc_bull = ggosc_val > 0
            ggosc_bear = ggosc_val < 0

            # 8-signal ensemble vote (5/8 = 62.5%)
            bull_votes = sum([mom_bull, vshort_bull, ema_bull, rsi_bull, macd_bull, has_bull_div, donch_bull, ggosc_bull])
            bear_votes = sum([mom_bear, vshort_bear, ema_bear, rsi_bear, macd_bear, has_bear_div, donch_bear, ggosc_bear])

            # Fee buffer: skip entries where expected move < fees
            if abs(ret_short) < MIN_ENTRY_MOVE:
                bull_votes = 0
                bear_votes = 0

            bullish = bull_votes >= MIN_VOTES
            bearish = bear_votes >= MIN_VOTES
            in_cooldown = (self.bar_count - self.exit_bar.get(symbol, -999)) < COOLDOWN_BARS

            size = equity * BASE_POSITION_PCT * self._weight
            current_pos = portfolio.positions.get(symbol, 0.0)
            target = current_pos

            if current_pos == 0:
                # Entry
                if not in_cooldown:
                    if bullish:
                        target = size
                    elif bearish:
                        target = -size
            else:
                # Position management: SDO-tightened trailing stop
                atr = self._calc_atr(bd.history, ATR_LOOKBACK)
                if atr is None:
                    atr = self.atr_at_entry.get(symbol, mid * 0.02)

                if symbol not in self.peak_prices:
                    self.peak_prices[symbol] = mid

                highs_h = bd.history["high"].values
                lows_h = bd.history["low"].values
                sdo_val, _ = self._calc_sdo(highs_h, lows_h, closes)
                sdo_extreme = ((current_pos > 0 and sdo_val[-1] > SDO_OVERBOUGHT) or
                               (current_pos < 0 and sdo_val[-1] < SDO_OVERSOLD))

                if current_pos > 0:
                    self.peak_prices[symbol] = max(self.peak_prices[symbol], mid)
                    if sdo_extreme:
                        stop = self.peak_prices[symbol] - SDO_TIGHT_ATR_MULT * atr
                        if mid < stop:
                            target = 0.0
                        self._current_stops[symbol] = stop
                else:
                    self.peak_prices[symbol] = min(self.peak_prices[symbol], mid)
                    if sdo_extreme:
                        stop = self.peak_prices[symbol] + SDO_TIGHT_ATR_MULT * atr
                        if mid > stop:
                            target = 0.0
                        self._current_stops[symbol] = stop

                # GGOSC TP1: take profit at 0.8× ATR from entry
                if symbol in self.entry_prices:
                    entry = self.entry_prices[symbol]
                    entry_atr = self.atr_at_entry.get(symbol, mid * 0.02)
                    if current_pos > 0 and mid >= entry + GGOSC_TP1_MULT * entry_atr:
                        target = 0.0
                    elif current_pos < 0 and mid <= entry - GGOSC_TP1_MULT * entry_atr:
                        target = 0.0

                # RSI mean-reversion exit
                if current_pos > 0 and rsi > RSI_OVERBOUGHT:
                    target = 0.0
                elif current_pos < 0 and rsi < RSI_OVERSOLD:
                    target = 0.0

                # Signal flip
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
                    self._current_stops.pop(symbol, None)
                    self.exit_bar[symbol] = self.bar_count
                elif (target > 0 and current_pos < 0) or (target < 0 and current_pos > 0):
                    self.entry_prices[symbol] = mid
                    self.peak_prices[symbol] = mid
                    self.atr_at_entry[symbol] = self._calc_atr(bd.history, ATR_LOOKBACK) or mid * 0.02

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
            self._current_stops.pop(sym, None)
            self.exit_bar[sym] = self.bar_count

    def on_stop_hit(self, symbol):
        """Called by backtester when intra-bar stop-loss triggers."""
        self.entry_prices.pop(symbol, None)
        self.peak_prices.pop(symbol, None)
        self.atr_at_entry.pop(symbol, None)
        self._current_stops.pop(symbol, None)
        self.exit_bar[symbol] = self.bar_count
