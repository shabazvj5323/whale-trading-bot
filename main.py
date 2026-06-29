# ═══════════════════════════════════════════════════════════
# main-13.py — WhaleEngine Trading Bot
# 10 Original Strategies + Strategy 11: QUANTUM SCALP
# 4-Phase Adaptive Trailing Stop Loss
# ═══════════════════════════════════════════════════════════

import os, json, time, logging, requests
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("WhaleBot")

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
HISTORY_FILE      = "history.json"
SYMBOLS           = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
LEVERAGE          = 10
MIN_SCORE         = 7
MAX_DAILY_LOSS    = 150.0
SR_PERIOD         = 20
SR_THRESH         = 0.002
BB_PERIOD         = 20
BB_STD            = 2.0
VOL_MA_PERIOD     = 20
VOL_SPIKE_MULT    = 1.5
EMA10_TIMEFRAMES  = ["5m", "15m", "30m"]
EMA10_VOL_MULT    = 1.2
SCAN_INTERVAL     = 60
DASHBOARD_FILE    = "index.html"
TRAILING_ENABLED  = True

# ★ QUANTUM SCALP CONFIG ──────────────────────────────────
QS_ENABLED              = True
QS_MIN_SCORE            = 70       # Entry threshold 0-100
QS_ATR_PERIOD           = 14
QS_ATR_LOW_PCT          = 25       # Below = too calm
QS_ATR_HIGH_PCT         = 80       # Above = too volatile
QS_VOL_DELTA_MULT       = 1.4
QS_BB_SQUEEZE_PCT       = 20
QS_TRAIL_PHASE2_ATR     = 1.5
QS_TRAIL_PHASE3_ATR     = 2.0
QS_TRAIL_PHASE4_PCT     = 0.70     # Lock 70% of profit

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════
def ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def ist_str():
    return ist_now().strftime("%Y-%m-%d %I:%M:%S %p")

def ist_short():
    return ist_now().strftime("%m-%d %H:%M")

def ema_calc(arr, period):
    k = 2.0 / (period + 1)
    r = [float(arr[0])]
    for p in arr[1:]:
        r.append(float(p) * k + r[-1] * (1 - k))
    return np.array(r)

def rsi_calc(closes, period=14):
    d = np.diff(closes.astype(float))
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = np.mean(g[-period:]) if len(g) >= period else 0.01
    al = np.mean(l[-period:]) if len(l) >= period else 0.01
    rs = ag / (al + 1e-10)
    return round(100.0 - (100.0 / (1.0 + rs)), 1)

def bollinger_calc(closes, period=20, num_std=2.0):
    c = closes.astype(float)
    sma = np.mean(c[-period:])
    std = np.std(c[-period:])
    return float(sma), float(sma + num_std * std), float(sma - num_std * std)

def atr_calc(highs, lows, closes, period=14):
    h, l, c = highs.astype(float), lows.astype(float), closes.astype(float)
    tr_list = []
    for i in range(1, len(c)):
        tr = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(c[i-1] - l[i]))
        tr_list.append(tr)
    if len(tr_list) < period:
        return float(np.mean(tr_list)) if tr_list else 0.0
    return float(np.mean(tr_list[-period:]))

# ═══════════════════════════════════════════════════════════
# ★ QUANTUM SCALP HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════
def hurst_exponent(ts, max_lag=20):
    """Hurst: <0.5 mean-reverting, >0.5 trending, =0.5 random"""
    ts = np.array(ts, dtype=float)
    lags = range(2, min(max_lag, len(ts) // 2))
    tau = []
    for lag in lags:
        diffs = ts[lag:] - ts[:-lag]
        tau.append(np.std(diffs))
    if len(tau) < 2 or min(tau) <= 0:
        return 0.5
    log_lags = np.log(list(lags))
    log_tau = np.log(tau)
    slope, _ = np.polyfit(log_lags, log_tau, 1)
    return float(np.clip(slope, 0, 1))

def atr_percentile(highs, lows, closes, period=14, lookback=100):
    """Current ATR as percentile of recent ATR values"""
    h, l, c = highs.astype(float), lows.astype(float), closes.astype(float)
    if len(c) < lookback:
        lookback = len(c) - 1
    tr_list = []
    for i in range(1, lookback):
        tr = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(c[i-1] - l[i]))
        tr_list.append(tr)
    if len(tr_list) < period:
        return 50.0
    atr_series = []
    for i in range(period, len(tr_list)):
        atr_series.append(np.mean(tr_list[i-period:i]))
    if not atr_series:
        return 50.0
    current_atr = atr_series[-1]
    percentile = (np.sum(np.array(atr_series) <= current_atr) / len(atr_series)) * 100
    return float(percentile)

def vwap_bands(highs, lows, closes, vols, period=20, num_std=2):
    """VWAP with standard deviation bands"""
    h = highs.astype(float)
    l = lows.astype(float)
    c = closes.astype(float)
    v = vols.astype(float)
    if len(c) < period:
        return None, None, None
    typical = (h[-period:] + l[-period:] + c[-period:]) / 3
    v_slice = v[-period:]
    vwap = np.sum(typical * v_slice) / (np.sum(v_slice) + 1e-9)
    squared_diff = v_slice * (typical - vwap) ** 2
    std = np.sqrt(np.sum(squared_diff) / (np.sum(v_slice) + 1e-9))
    return float(vwap), float(vwap + num_std * std), float(vwap - num_std * std)

def cumulative_volume_delta(opens, highs, lows, closes, vols, period=10):
    """Approximate CVD using candle body direction x volume"""
    o = opens.astype(float)
    h = highs.astype(float)
    l = lows.astype(float)
    c = closes.astype(float)
    v = vols.astype(float)
    if len(c) < period:
        return 0.0, 0.0
    buy_vol = 0.0
    sell_vol = 0.0
    for i in range(-period, 0):
        body = c[i] - o[i]
        candle_range = h[i] - l[i] + 1e-9
        if body > 0:
            buy_vol += v[i] * (body / candle_range)
        elif body < 0:
            sell_vol += v[i] * (abs(body) / candle_range)
        else:
            buy_vol += v[i] * 0.5
            sell_vol += v[i] * 0.5
    return float(buy_vol), float(sell_vol)

def bb_bandwidth_percentile(closes, period=20, std_mult=2.0, lookback=50):
    """Current BB bandwidth as percentile of recent history"""
    c = closes.astype(float)
    if len(c) < lookback:
        lookback = len(c)
    bw_list = []
    for end in range(period, min(lookback + 1, len(c) + 1)):
        window = c[end - period:end]
        if len(window) < period:
            continue
        sma = np.mean(window)
        std = np.std(window)
        bw = (4 * std_mult * std) / (sma + 1e-9)
        bw_list.append(bw)
    if not bw_list:
        return 50.0
    current_sma = np.mean(c[-period:])
    current_std = np.std(c[-period:])
    current_bw = (4 * std_mult * current_std) / (current_sma + 1e-9)
    pct = (np.sum(np.array(bw_list) <= current_bw) / len(bw_list)) * 100
    return float(pct)

# ═══════════════════════════════════════════════════════════
# DARVAS BOX STRATEGY
# ═══════════════════════════════════════════════════════════
def darvas_box_strategy(highs, lows, closes, vols, box_period=20):
    if len(highs) < box_period + 5:
        return "hold", 0, 0, 0, []
    box_highs = highs[-(box_period + 5):-5]
    box_lows = lows[-(box_period + 5):-5]
    resistance = float(np.max(box_highs))
    support = float(np.min(box_lows))
    price = float(closes[-1])
    current_volume = float(vols[-1])
    vol_ma = float(np.mean(vols[-VOL_MA_PERIOD:])) if len(vols) >= VOL_MA_PERIOD else current_volume
    breakout_up = price > resistance * (1 + SR_THRESH) and current_volume > vol_ma * VOL_SPIKE_MULT
    breakout_down = price < support * (1 - SR_THRESH) and current_volume > vol_ma * VOL_SPIKE_MULT
    if breakout_up:
        signal = "buy"
        trail_sl = round(support, 6)
        reasons = [
            "DARVAS BOX",
            "Break: $" + str(round(resistance, 4)),
            "VOL " + str(round(current_volume / vol_ma, 1)) + "x",
        ]
    elif breakout_down:
        signal = "sell"
        trail_sl = round(resistance, 6)
        reasons = [
            "DARVAS BOX",
            "Break: $" + str(round(support, 4)),
            "VOL " + str(round(current_volume / vol_ma, 1)) + "x",
        ]
    else:
        return "hold", round(resistance, 6), round(support, 6), 0, []
    return signal, round(resistance, 6), round(support, 6), trail_sl, reasons

# ═══════════════════════════════════════════════════════════
# ★ STRATEGY 11: QUANTUM SCALP — Main Scoring Function
# ═══════════════════════════════════════════════════════════
def quantum_scalp_score(opens, highs, lows, closes, vols,
                        closes15=None, closes1h=None):
    """
    Returns: (score, direction, regime_type, reasons)
    score: 0-100, direction: 'buy'|'sell'|None
    """
    if len(closes) < 50:
        return 0, None, 'UNKNOWN', ['Insufficient data']

    c = closes.astype(float)
    h = highs.astype(float)
    l = lows.astype(float)
    o = opens.astype(float)
    v = vols.astype(float)
    price = float(c[-1])

    score = 0
    buy_signals = 0
    sell_signals = 0
    reasons = []

    # ── FACTOR 1: Multi-Timeframe Trend Alignment (20 pts) ──
    def ema_stack(arr):
        if arr is None or len(arr) < 50:
            return 0
        a = arr.astype(float)
        e9  = np.mean(a[-9:])
        e21 = np.mean(a[-21:])
        e50 = np.mean(a[-50:])
        if e9 > e21 > e50:
            return 1
        elif e9 < e21 < e50:
            return -1
        return 0

    tf_5m  = ema_stack(c)
    tf_15m = ema_stack(closes15)
    tf_1h  = ema_stack(closes1h)
    mtf_score = tf_5m + tf_15m + tf_1h

    if mtf_score >= 3:
        score += 20; buy_signals += 3
        reasons.append("MTF BULL STACK 3/3")
    elif mtf_score <= -3:
        score += 20; sell_signals += 3
        reasons.append("MTF BEAR STACK 3/3")
    elif mtf_score >= 2:
        score += 12; buy_signals += 2
        reasons.append("MTF BULL 2/3")
    elif mtf_score <= -2:
        score += 12; sell_signals += 2
        reasons.append("MTF BEAR 2/3")
    elif mtf_score >= 1:
        score += 5; buy_signals += 1
    elif mtf_score <= -1:
        score += 5; sell_signals += 1

    # ── FACTOR 2: VWAP Position (15 pts) ──
    vwap_mid, vwap_upper, vwap_lower = vwap_bands(h, l, c, v, period=20, num_std=2)
    if vwap_mid is not None:
        if price < vwap_lower:
            score += 15; buy_signals += 2
            reasons.append("VWAP BELOW -2σ BUY")
        elif price > vwap_upper:
            score += 15; sell_signals += 2
            reasons.append("VWAP ABOVE +2σ SELL")
        elif price < vwap_mid * 0.998:
            score += 8; buy_signals += 1
            reasons.append("VWAP BELOW MID")
        elif price > vwap_mid * 1.002:
            score += 8; sell_signals += 1
            reasons.append("VWAP ABOVE MID")

    # ── FACTOR 3: Volume Delta Imbalance (15 pts) ──
    buy_vol, sell_vol = cumulative_volume_delta(o, h, l, c, v, period=10)
    total_vol = buy_vol + sell_vol + 1e-9
    delta_ratio = buy_vol / total_vol
    if delta_ratio > 0.65:
        score += 15; buy_signals += 2
        reasons.append("VOL DELTA BUY " + str(round(delta_ratio * 100)) + "%")
    elif delta_ratio < 0.35:
        score += 15; sell_signals += 2
        reasons.append("VOL DELTA SELL " + str(round((1 - delta_ratio) * 100)) + "%")
    elif delta_ratio > 0.55:
        score += 7; buy_signals += 1
    elif delta_ratio < 0.45:
        score += 7; sell_signals += 1

    # ── FACTOR 4: BB Squeeze + Expansion (12 pts) ──
    bb_pct = bb_bandwidth_percentile(c, period=20, std_mult=2.0, lookback=50)
    if bb_pct < QS_BB_SQUEEZE_PCT:
        body = c[-1] - o[-1]
        if body > 0:
            score += 12; buy_signals += 2
            reasons.append("BB SQUEEZE → EXPANSION UP")
        else:
            score += 12; sell_signals += 2
            reasons.append("BB SQUEEZE → EXPANSION DN")

    # ── FACTOR 5: Order Flow Imbalance (13 pts) ──
    of_buy = of_sell = 0.0
    for i in range(-10, 0):
        body = c[i] - o[i]
        vol_i = v[i]
        if body > 0:
            of_buy += body * vol_i
        else:
            of_sell += abs(body) * vol_i
    of_total = of_buy + of_sell + 1e-9
    of_ratio = of_buy / of_total
    if of_ratio > 0.62:
        score += 13; buy_signals += 2
        reasons.append("ORDER FLOW STRONG BUY")
    elif of_ratio < 0.38:
        score += 13; sell_signals += 2
        reasons.append("ORDER FLOW STRONG SELL")

    # ── FACTOR 6: ATR Regime Filter (12 pts) ──
    atr_pct = atr_percentile(h, l, c, period=QS_ATR_PERIOD, lookback=100)
    if QS_ATR_LOW_PCT <= atr_pct <= QS_ATR_HIGH_PCT:
        score += 12
        reasons.append("ATR SWEET SPOT " + str(round(atr_pct)) + "%")
    elif atr_pct > QS_ATR_HIGH_PCT:
        score = max(0, score - 5)
        reasons.append("⚠️ ATR HIGH " + str(round(atr_pct)) + "%")
    else:
        score = max(0, score - 3)
        reasons.append("⚠️ ATR LOW " + str(round(atr_pct)) + "%")

    # ── FACTOR 7: Momentum Divergence (13 pts) ──
    rsi = rsi_calc(c, 14)
    price_rising = c[-1] > c[-3]
    price_falling = c[-1] < c[-3]
    if rsi < 35 and price_rising:
        score += 13; buy_signals += 2
        reasons.append("RSI OVERSOLD + REVERSAL (" + str(rsi) + ")")
    elif rsi > 65 and price_falling:
        score += 13; sell_signals += 2
        reasons.append("RSI OVERBOUGHT + REVERSAL (" + str(rsi) + ")")
    elif 40 < rsi < 60 and price_rising:
        score += 6; buy_signals += 1
    elif 40 < rsi < 60 and price_falling:
        score += 6; sell_signals += 1

    # ── REGIME CLASSIFICATION ──
    hurst = hurst_exponent(c[-50:], max_lag=15)
    if hurst > 0.6:
        regime_type = 'TRENDING'
    elif hurst < 0.4:
        regime_type = 'RANGING'
    elif atr_pct > QS_ATR_HIGH_PCT:
        regime_type = 'VOLATILE'
    else:
        regime_type = 'CALM'
    reasons.insert(0, "REGIME: " + regime_type + " (H=" + str(round(hurst, 2)) + ")")

    # ── FINAL DIRECTION ──
    direction = None
    if buy_signals > sell_signals and buy_signals >= 3:
        direction = 'buy'
    elif sell_signals > buy_signals and sell_signals >= 3:
        direction = 'sell'

    score = int(min(max(score, 0), 100))
    return score, direction, regime_type, reasons

# ═══════════════════════════════════════════════════════════
# ★ 4-PHASE QUANTUM TRAILING STOP LOSS
# ═══════════════════════════════════════════════════════════
def quantum_trailing_sl(pos, current_price, recent_highs, recent_lows, atr_current):
    """
    4-Phase Trailing SL. Returns new SL value.
    pos must have: side, entry, sl, initial_sl, highest_since_entry, lowest_since_entry
    """
    side = pos['side']
    entry = float(pos['entry'])
    sl = float(pos['sl'])
    initial_sl = float(pos.get('initial_sl', sl))
    initial_risk = abs(entry - initial_sl)
    if initial_risk < 1e-9:
        initial_risk = atr_current

    if side == 'buy':
        profit = current_price - entry
        profit_r = profit / initial_risk
        highest = max(float(pos.get('highest_since_entry', entry)),
                      float(np.max(recent_highs)) if len(recent_highs) > 0 else entry)
        pos['highest_since_entry'] = highest
        new_sl = sl

        # PHASE 1: Breakeven lock (>= 0.3R)
        if profit_r >= 0.3:
            be_sl = entry + (initial_risk * 0.05)
            if be_sl > new_sl:
                new_sl = be_sl

        # PHASE 2: ATR trail (>= 1.0R)
        if profit_r >= 1.0:
            trail_sl = highest - (QS_TRAIL_PHASE2_ATR * atr_current)
            if trail_sl > new_sl:
                new_sl = trail_sl

        # PHASE 3: Chandelier (>= 2.0R)
        if profit_r >= 2.0:
            trail_sl = highest - (QS_TRAIL_PHASE3_ATR * atr_current * 0.8)
            if trail_sl > new_sl:
                new_sl = trail_sl

        # PHASE 4: Parabolic lock (>= 2.5R)
        if profit_r >= 2.5:
            max_profit = highest - entry
            locked = max_profit * QS_TRAIL_PHASE4_PCT
            parabolic_sl = entry + locked
            if parabolic_sl > new_sl:
                new_sl = parabolic_sl

        return round(new_sl, 6)

    else:  # SELL
        profit = entry - current_price
        profit_r = profit / initial_risk
        lowest = min(float(pos.get('lowest_since_entry', entry)),
                     float(np.min(recent_lows)) if len(recent_lows) > 0 else entry)
        pos['lowest_since_entry'] = lowest
        new_sl = sl

        if profit_r >= 0.3:
            be_sl = entry - (initial_risk * 0.05)
            if be_sl < new_sl:
                new_sl = be_sl

        if profit_r >= 1.0:
            trail_sl = lowest + (QS_TRAIL_PHASE2_ATR * atr_current)
            if trail_sl < new_sl:
                new_sl = trail_sl

        if profit_r >= 2.0:
            trail_sl = lowest + (QS_TRAIL_PHASE3_ATR * atr_current * 0.8)
            if trail_sl < new_sl:
                new_sl = trail_sl

        if profit_r >= 2.5:
            max_profit = entry - lowest
            locked = max_profit * QS_TRAIL_PHASE4_PCT
            parabolic_sl = entry - locked
            if parabolic_sl < new_sl:
                new_sl = parabolic_sl

        return round(new_sl, 6)

# ═══════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════
class WhaleEngine:
    def __init__(self):
        self.state = self.load_history()
        self.dashboard = []
        self._last_is_blast = False

    # ── History ──────────────────────────────
    def load_history(self):
        default = {
            "total_pnl": 0.0,
            "active_positions": {},
            "trades": [],
            "last_prices": {},
            "last_prices_history": {},
            "stats": {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "daily_pnl": {},
                "daily_trades": {},
            }
        }
        if not os.path.exists(HISTORY_FILE):
            return default
        try:
            with open(HISTORY_FILE) as f:
                data = json.load(f)
            for k, v in default.items():
                if k not in data:
                    data[k] = v
            for k, v in default["stats"].items():
                if k not in data.get("stats", {}):
                    data.setdefault("stats", {})[k] = v
            return data
        except Exception:
            return default

    def save_history(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def cleanup_stale_positions(self):
        now = ist_now()
        to_remove = []
        for symbol, pos in self.state["active_positions"].items():
            try:
                entry_dt = datetime.strptime(
                    pos.get("time", ""), "%m-%d %H:%M"
                )
                age_hours = (now - entry_dt).total_seconds() / 3600
                if age_hours > 6:
                    log.warning(f"STALE: {symbol} open {age_hours:.1f}h — force closing")
                    to_remove.append(symbol)
            except Exception as ex:
                log.warning(f"Stale check error {symbol}: {ex}")
                to_remove.append(symbol)
        for symbol in to_remove:
            if symbol in self.state["active_positions"]:
                del self.state["active_positions"][symbol]
        if to_remove:
            self.save_history()

    # ── Risk check ───────────────────────────
    def can_trade(self, score=0):
        today = ist_now().strftime("%Y-%m-%d")
        daily_loss = abs(min(self.state["stats"]["daily_pnl"].get(today, 0), 0))
        if daily_loss >= MAX_DAILY_LOSS:
            log.warning(f"Daily loss limit hit: ${daily_loss:.2f} — trading paused")
            return False
        return True

    # ── Market data ──────────────────────────
    COINGECKO_IDS = {
        "BTC/USDT": "bitcoin",
        "ETH/USDT": "ethereum",
        "SOL/USDT": "solana",
        "BNB/USDT": "binancecoin",
        "XRP/USDT": "ripple",
        "DOGE/USDT": "dogecoin",
    }

    def fetch_okx_public(self, symbol, interval, limit=120):
        clean = symbol.replace("/", "-")
        url = "https://www.okx.com/api/v5/market/candles"
        params = {
            "instId": clean + "-SWAP",
            "bar": interval,
            "limit": str(limit)
        }
        headers = {"OK-ACCESS-KEY": ""}
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "0" and data.get("data"):
                    raw = data["data"][::-1]
                    arr = np.array([[float(c[1]), float(c[2]), float(c[3]),
                                     float(c[4]), float(c[5])] for c in raw])
                    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
            log.warning(f"OKX public {resp.status_code} for {symbol}")
            return None
        except Exception as e:
            log.warning(f"OKX public failed {symbol}: {e}")
            return None

    def fetch_data(self, symbol, limit=120, vol_map=None):
        # ★ MODIFIED: Now also fetches 1h data for Quantum Scalp MTF
        for attempt in range(3):
            result_5m = self.fetch_okx_public(symbol, "5m", limit)
            result_15m = self.fetch_okx_public(symbol, "15m", 60)
            result_1h = self.fetch_okx_public(symbol, "1H", 60)  # ★ NEW
            if result_5m is not None and result_15m is not None:
                opens, highs, lows, closes, vols = result_5m
                closes15 = result_15m[3]
                closes1h = result_1h[3] if result_1h is not None else None  # ★ NEW
                log.info(f"OKX Real: {symbol} @ ${round(float(closes[-1]), 4)}")
                return opens, highs, lows, closes, vols, closes15, closes1h  # ★ NOW RETURNS 7
            log.warning(f"OKX retry {attempt+1}/3 for {symbol}")
            time.sleep(2)
        log.error(f"OKX failed after 3 retries: {symbol} — skipping")
        return None

    # ══════════════════════════════════════════
    # SCORING: Original 10 Strategies + Quantum Scalp
    # ══════════════════════════════════════════
    def score_signal(self, opens, highs, lows, closes, vols, closes15, closes1h=None):
        score, direction, reasons = 0, None, []
        is_blast = False
        price = float(closes[-1])

        # Real RSI for logging
        try:
            gains, losses = [], []
            for i in range(1, 15):
                diff = float(closes[-i]) - float(closes[-i-1])
                if diff > 0:
                    gains.append(diff)
                else:
                    losses.append(abs(diff))
            avg_gain = sum(gains) / 14 if gains else 0.01
            avg_loss = sum(losses) / 14 if losses else 0.01
            rs = avg_gain / avg_loss
            r = round(100 - (100 / (1 + rs)), 1)
        except:
            r = 50.0

        # Regime detection
        try:
            sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else price
            rsi_val = r
            if price > sma50 and rsi_val > 55:
                regime = "BULL"
            elif price < sma50 and rsi_val < 45:
                regime = "BEAR"
            else:
                regime = "SIDEWAYS"
        except:
            regime = "SIDEWAYS"

        # Vol spike
        vol_ma = float(np.mean(vols[-VOL_MA_PERIOD:])) if len(vols) >= VOL_MA_PERIOD else float(vols[-1])
        vol_spike = float(vols[-1]) > vol_ma * VOL_SPIKE_MULT

        # BB calc
        try:
            bb_mid, bb_upper, bb_lower = bollinger_calc(closes, BB_PERIOD, BB_STD)
            bb_width = bb_upper - bb_lower
        except:
            bb_mid = price; bb_upper = price * 1.01; bb_lower = price * 0.99; bb_width = 0

        adaptive_min = MIN_SCORE

        # ── Strategy 1: BB Blast ──
        try:
            bb_history = []
            for i in range(max(0, len(closes) - 30), len(closes)):
                if i >= BB_PERIOD:
                    sma_i = float(np.mean(closes[i-BB_PERIOD:i]))
                    std_i = float(np.std(closes[i-BB_PERIOD:i]))
                    bb_history.append(4 * BB_STD * std_i / (sma_i + 1e-9))
            if bb_history:
                current_bw = bb_width / (bb_mid + 1e-9)
                avg_bw = np.mean(bb_history)
                if current_bw < avg_bw * 0.5 and vol_spike:
                    is_blast = True
                    if price > bb_mid:
                        score += 4; direction = direction or "buy"
                        reasons.append("BB BLAST UP")
                    else:
                        score += 4; direction = direction or "sell"
                        reasons.append("BB BLAST DOWN")
        except:
            pass

        # ── Strategy 2: BB Bounce ──
        try:
            if price <= bb_lower and regime != "BEAR":
                score += 3; direction = direction or "buy"
                reasons.append("BB BOUNCE BUY")
            elif price >= bb_upper and regime != "BULL":
                score += 3; direction = direction or "sell"
                reasons.append("BB BOUNCE SELL")
        except:
            pass

        # ── Strategy 3-5: Regime-specific ──
        if regime == "SIDEWAYS":
            # Double BB Zone
            try:
                inner_upper = bb_mid + (bb_upper - bb_mid) * 0.5
                inner_lower = bb_mid - (bb_mid - bb_lower) * 0.5
                if price < inner_lower:
                    score += 3; direction = direction or "buy"
                    reasons.append("DOUBLE BB BUY ZONE")
                elif price > inner_upper:
                    score += 3; direction = direction or "sell"
                    reasons.append("DOUBLE BB SELL ZONE")
            except:
                pass

            # Order Flow in sideways
            try:
                buy_pressure = sell_pressure = 0.0
                for i in range(-10, 0):
                    body = float(closes[i]) - float(opens[i])
                    vol = float(vols[i])
                    if body > 0:
                        buy_pressure += abs(body) * vol
                    else:
                        sell_pressure += abs(body) * vol
                buy_ratio = buy_pressure / (buy_pressure + sell_pressure + 1e-9)
                if buy_ratio > 0.65:
                    score += 2; direction = direction or "buy"
                    reasons.append("ORDER FLOW BUY")
                elif buy_ratio < 0.35:
                    score += 2; direction = direction or "sell"
                    reasons.append("ORDER FLOW SELL")
            except:
                pass

        elif regime == "BULL":
            # EMA Ribbon Bull
            try:
                if len(closes) >= 50:
                    e9 = ema_calc(closes, 9)[-1]
                    e21 = ema_calc(closes, 21)[-1]
                    e50 = ema_calc(closes, 50)[-1]
                    if e9 > e21 > e50:
                        score += 4; direction = direction or "buy"
                        reasons.append("EMA RIBBON BULL")
            except:
                pass

            # VWAP Bull
            try:
                typical = (highs + lows + closes) / 3
                vwap = float(np.sum(typical * vols) / (np.sum(vols) + 1e-9))
                if price < vwap * 0.999:
                    score += 3; direction = direction or "buy"
                    reasons.append("VWAP BUY")
                elif price > vwap * 1.003:
                    score += 2; direction = direction or "sell"
                    reasons.append("VWAP SELL")
            except:
                pass

            # SR Breakout
            if len(highs) >= SR_PERIOD:
                resistance = float(np.max(highs[-SR_PERIOD:]))
                if price > resistance * (1 + SR_THRESH):
                    score += 4; direction = direction or "buy"
                    reasons.append("SR BREAKOUT UP")
                    if vol_spike:
                        score += 1

            # Order Flow Bull
            try:
                buy_pressure = sell_pressure = 0.0
                for i in range(-10, 0):
                    body = float(closes[i]) - float(opens[i])
                    vol = float(vols[i])
                    if body > 0:
                        buy_pressure += abs(body) * vol
                    else:
                        sell_pressure += abs(body) * vol
                buy_ratio = buy_pressure / (buy_pressure + sell_pressure + 1e-9)
                if buy_ratio > 0.60:
                    score += 2; direction = direction or "buy"
                    reasons.append("ORDER FLOW BUY")
            except:
                pass

        elif regime == "BEAR":
            # EMA Ribbon Bear
            try:
                if len(closes) >= 50:
                    e9 = ema_calc(closes, 9)[-1]
                    e21 = ema_calc(closes, 21)[-1]
                    e50 = ema_calc(closes, 50)[-1]
                    if e9 < e21 < e50:
                        score += 4; direction = direction or "sell"
                        reasons.append("EMA RIBBON BEAR")
            except:
                pass

            # VWAP Bear
            try:
                typical = (highs + lows + closes) / 3
                vwap = float(np.sum(typical * vols) / (np.sum(vols) + 1e-9))
                if price > vwap * 1.001:
                    score += 3; direction = direction or "sell"
                    reasons.append("VWAP SELL")
                elif price < vwap * 0.997:
                    score += 2; direction = direction or "buy"
                    reasons.append("VWAP BUY")
            except:
                pass

            # SR Breakout Down
            if len(lows) >= SR_PERIOD:
                support = float(np.min(lows[-SR_PERIOD:]))
                if price < support * (1 - SR_THRESH):
                    score += 4; direction = direction or "sell"
                    reasons.append("SR BREAKOUT DOWN")

            # BB Rally Sell
            if price >= bb_upper:
                score += 3; direction = direction or "sell"
                reasons.append("BB RALLY SELL")

            # Order Flow Bear
            try:
                buy_pressure = sell_pressure = 0.0
                for i in range(-10, 0):
                    body = float(closes[i]) - float(opens[i])
                    vol = float(vols[i])
                    if body > 0:
                        buy_pressure += abs(body) * vol
                    else:
                        sell_pressure += abs(body) * vol
                buy_ratio = buy_pressure / (buy_pressure + sell_pressure + 1e-9)
                if buy_ratio < 0.40:
                    score += 2; direction = direction or "sell"
                    reasons.append("ORDER FLOW SELL")
            except:
                pass

        # ── Strategy 8: Darvas Box ──
        try:
            d_signal, d_res, d_sup, d_sl, d_reasons = darvas_box_strategy(
                highs, lows, closes, vols
            )
            if d_signal != "hold":
                score += 3
                direction = direction or d_signal
                reasons.extend(d_reasons)
        except:
            pass

        # ══════════════════════════════════════════════
        # ★ STRATEGY 11: QUANTUM SCALP OVERRIDE
        # ══════════════════════════════════════════════
        qs_override = False
        if QS_ENABLED:
            try:
                qs_score, qs_dir, qs_regime, qs_reasons = quantum_scalp_score(
                    opens, highs, lows, closes, vols, closes15, closes1h
                )
                # If QS score is high enough, override original scoring
                if qs_score >= QS_MIN_SCORE and qs_dir:
                    # Convert QS 0-100 score to 0-10 scale for compatibility
                    qs_converted = min(10, max(1, int(qs_score / 10)))
                    if qs_converted >= score:
                        score = qs_converted
                        direction = qs_dir
                        reasons = qs_reasons + ["★ QUANTUM SCALP"]
                        qs_override = True
                        log.info(f"⚡ QUANTUM SCALP OVERRIDE: Score={qs_score}/100 → {qs_converted}/10 Dir={qs_dir}")
                else:
                    # Even if no override, log QS score for monitoring
                    reasons.append(f"QS:{qs_score}")
            except Exception as e:
                log.warning(f"Quantum Scalp error: {e}")

        return score, direction, r, bb_upper, bb_lower, reasons, adaptive_min, qs_override

    # ── Position management ──────────────────
    def check_positions(self, symbol, current_price, opens=None, highs=None, lows=None, closes=None):
        if symbol not in self.state["active_positions"]:
            return
        pos = self.state["active_positions"][symbol]
        side = pos["side"]
        entry = float(pos["entry"])
        tp = float(pos["tp"])
        sl = float(pos["sl"])
        qty = float(pos.get("qty", 1))
        margin = float(pos.get("margin", 10))
        is_qs = pos.get("strategy") == "QUANTUM SCALP"  # ★ NEW

        try:
            bb_mid, bb_upper, bb_lower = bollinger_calc(
                closes if closes is not None else np.array([entry] * 20),
                BB_PERIOD, BB_STD
            )
        except:
            bb_mid = entry

        profit = (current_price - entry) if side == "buy" else (entry - current_price)

        # ══════════════════════════════════════
        # ★ TRAILING SL: Quantum or Classic
        # ══════════════════════════════════════
        if is_qs and TRAILING_ENABLED and highs is not None and lows is not None:
            # ★ USE 4-PHASE QUANTUM TRAILING SL
            atr_current = atr_calc(highs, lows, closes if closes is not None else np.array([current_price] * 20))
            new_sl = quantum_trailing_sl(pos, current_price, highs[-10:], lows[-10:], atr_current)
            if side == "buy" and new_sl > sl:
                sl = new_sl
                self.state["active_positions"][symbol]["sl"] = sl
            elif side == "sell" and new_sl < sl:
                sl = new_sl
                self.state["active_positions"][symbol]["sl"] = sl
        elif TRAILING_ENABLED:
            # Classic BB-mid trailing
            last_prices_arr = list(
                self.state.get("last_prices_history", {}).get(symbol, [entry] * 20)
            )
            last_prices_arr.append(current_price)
            last_prices_arr = last_prices_arr[-20:]
            self.state.setdefault("last_prices_history", {})[symbol] = last_prices_arr
            bb_mid_trail = round(sum(last_prices_arr) / len(last_prices_arr), 6)

            trail_dist = abs(bb_upper - bb_lower) * 0.3
            if side == "buy":
                if current_price > tp:
                    tp = round(current_price + trail_dist, 6)
                if profit > 0 and bb_mid_trail > sl:
                    sl = bb_mid_trail
            else:
                if current_price < tp:
                    tp = round(current_price - trail_dist, 6)
                if profit > 0 and bb_mid_trail < sl:
                    sl = bb_mid_trail

            self.state["active_positions"][symbol]["tp"] = tp
            self.state["active_positions"][symbol]["sl"] = sl

        # ── Check TP / SL hit ──
        hit = False
        reason = ""
        exit_p = current_price
        if side == "buy":
            if current_price >= tp:
                hit, reason, exit_p = True, "TP", tp
            elif current_price <= sl:
                hit, reason, exit_p = True, "SL", sl
        else:
            if current_price <= tp:
                hit, reason, exit_p = True, "TP", tp
            elif current_price >= sl:
                hit, reason, exit_p = True, "SL", sl

        if hit:
            pnl = round(
                (exit_p - entry) * qty if side == "buy" else (entry - exit_p) * qty, 4
            )
            st = self.state["stats"]
            self.state["total_pnl"] = round(self.state.get("total_pnl", 0) + pnl, 4)
            st["total_trades"] += 1
            if pnl > 0:
                st["wins"] += 1
                st["best_trade"] = max(st["best_trade"], pnl)
            else:
                st["losses"] += 1
                st["worst_trade"] = min(st["worst_trade"], pnl)

            today = ist_now().strftime("%Y-%m-%d")
            st["daily_pnl"][today] = round(st["daily_pnl"].get(today, 0) + pnl, 4)
            st["daily_trades"][today] = st["daily_trades"].get(today, 0) + 1

            exit_time_str = ist_short()
            pos_reasons = pos.get("reasons", [])
            strategy_name = pos.get("strategy", "UNKNOWN")

            # Detect strategy from reasons if not set
            if strategy_name == "UNKNOWN":
                for strat_key in ["BB BLAST", "BB BOUNCE", "DOUBLE BB", "SR BREAKOUT",
                                  "VWAP", "EMA RIBBON", "ORDER FLOW", "DARVAS", "HEDGE",
                                  "QUANTUM SCALP"]:  # ★ Added
                    if any(strat_key in str(r) for r in pos_reasons):
                        strategy_name = strat_key
                        break

            self.state["trades"].append({
                "time": exit_time_str,
                "symbol": symbol,
                "side": side.upper(),
                "entry": round(entry, 6),
                "exit": round(exit_p, 6),
                "pnl": pnl,
                "result": reason,
                "score": pos.get("score", 8),
                "margin": margin,
                "strategy": strategy_name,
                "reasons": pos_reasons,
            })
            del self.state["active_positions"][symbol]
            icon = "PROFIT" if pnl > 0 else "LOSS"
            log.info(f"{icon}: {symbol} {side.upper()} | PnL:${pnl} | {reason} | Strategy:{strategy_name}")
            self.save_history()

    # ── Main Loop ────────────────────────────
    def run(self):
        log.info("🐋 WhaleEngine starting...")
        while True:
            try:
                self.cleanup_stale_positions()
                self.dashboard = []

                for symbol in SYMBOLS:
                    # ★ fetch_data now returns 7 values
                    data = self.fetch_data(symbol, limit=120)
                    if data is None:
                        continue

                    opens, highs, lows, closes, vols, closes15, closes1h = data
                    price = float(closes[-1])

                    # Store price history
                    self.state["last_prices"][symbol] = price
                    hist = self.state.get("last_prices_history", {}).get(symbol, [])
                    hist.append(price)
                    self.state["last_prices_history"][symbol] = hist[-30:]

                    # ★ score_signal now takes closes1h and returns 8 values
                    result = self.score_signal(opens, highs, lows, closes, vols, closes15, closes1h)
                    score, direction, rsi_val, bb_upper, bb_lower, reasons, adaptive_min, qs_override = result

                    # Check existing positions first
                    self.check_positions(symbol, price, opens, highs, lows, closes)

                    # ── Strategy 10: EMA Fresh Crossover ──
                    # (runs independently on multiple timeframes)
                    for tf in EMA10_TIMEFRAMES:
                        try:
                            limit_tf = 120 if tf == "5m" else 80 if tf == "15m" else 60
                            result_tf = self.fetch_okx_public(symbol, tf, limit_tf)
                            if result_tf is None:
                                continue
                            o_tf, h_tf, l_tf, c_tf, v_tf = result_tf
                            if len(c_tf) < 21:
                                continue
                            ema10 = ema_calc(c_tf, 10)
                            ema20 = ema_calc(c_tf, 20)
                            prev_diff = float(ema10[-2] - ema20[-2])
                            curr_diff = float(ema10[-1] - ema20[-1])
                            if prev_diff <= 0 and curr_diff > 0:
                                s10_dir = "buy"
                                s10_score = 8
                            elif prev_diff >= 0 and curr_diff < 0:
                                s10_dir = "sell"
                                s10_score = 8
                            else:
                                continue

                            s10_key = symbol + "_" + tf
                            if s10_key not in self.state["active_positions"] and self.can_trade():
                                s10_atr = atr_calc(h_tf, l_tf, c_tf, 14)
                                s10_entry = float(c_tf[-1])
                                if s10_dir == "buy":
                                    s10_tp = round(s10_entry + s10_atr * 2, 6)
                                    s10_sl = round(s10_entry - s10_atr * 1.2, 6)
                                else:
                                    s10_tp = round(s10_entry - s10_atr * 2, 6)
                                    s10_sl = round(s10_entry + s10_atr * 1.2, 6)
                                s10_margin = round(10.0, 2)
                                s10_qty = round((s10_margin * LEVERAGE) / s10_entry, 6)

                                self.state["active_positions"][s10_key] = {
                                    "side": s10_dir,
                                    "entry": s10_entry,
                                    "tp": s10_tp,
                                    "sl": s10_sl,
                                    "qty": s10_qty,
                                    "margin": s10_margin,
                                    "score": s10_score,
                                    "strategy": "EMA CROSSOVER",
                                    "reasons": ["EMA FRESH CROSS " + tf.upper()],
                                    "time": ist_short(),
                                }
                                self.save_history()
                                log.info(f"S10 NEW: {s10_key} {s10_dir.upper()} | TP:{s10_tp} SL:{s10_sl}")
                        except Exception as e:
                            log.warning(f"S10 error {symbol} {tf}: {e}")

                    # ── Open new position from main strategies ──
                    if (direction and score >= adaptive_min
                            and symbol not in self.state["active_positions"]
                            and self.can_trade(score)):

                        atr = atr_calc(highs, lows, closes, QS_ATR_PERIOD)
                        if qs_override:
                            # ★ Quantum Scalp: tighter initial SL
                            initial_risk = atr * 1.0
                            tp_dist = atr * 2.5  # Higher R:R for QS
                            margin = round(15.0, 2)  # Slightly larger margin
                        else:
                            initial_risk = atr * 1.5
                            tp_dist = atr * 2.0
                            margin = round(10.0, 2)

                        entry_price = price
                        if direction == "buy":
                            tp = round(entry_price + tp_dist, 6)
                            sl = round(entry_price - initial_risk, 6)
                        else:
                            tp = round(entry_price - tp_dist, 6)
                            sl = round(entry_price + initial_risk, 6)

                        qty = round((margin * LEVERAGE) / entry_price, 6)

                        strategy_name = "QUANTUM SCALP" if qs_override else "CLASSIC"

                        self.state["active_positions"][symbol] = {
                            "side": direction,
                            "entry": entry_price,
                            "tp": tp,
                            "sl": sl,
                            "initial_sl": sl,        # ★ For 4-phase trailing
                            "highest_since_entry": entry_price,  # ★
                            "lowest_since_entry": entry_price,   # ★
                            "qty": qty,
                            "margin": margin,
                            "score": score,
                            "strategy": strategy_name,
                            "reasons": reasons,
                            "time": ist_short(),
                        }
                        self.save_history()
                        log.info(
                            f"NEW TRADE: {symbol} {direction.upper()} | Score:{score} | "
                            f"Margin:${margin} | TP:{tp} | SL:{sl} | Strategy:{strategy_name}"
                        )

                    # ── Dashboard data ──
                    sig = "HOLD"
                    if direction == "buy":
                        sig = "BUY"
                    elif direction == "sell":
                        sig = "SELL"

                    self.dashboard.append({
                        "symbol": symbol,
                        "price": price,
                        "score": score,
                        "signal": sig,
                        "rsi": rsi_val,
                        "reasons": reasons,
                        "tp": self.state["active_positions"].get(symbol, {}).get("tp"),
                        "sl": self.state["active_positions"].get(symbol, {}).get("sl"),
                        "entry": self.state["active_positions"].get(symbol, {}).get("entry"),
                    })

                self.generate_dashboard()
                log.info(f"Scan complete. Next scan in {SCAN_INTERVAL}s")
                time.sleep(SCAN_INTERVAL)

            except Exception as e:
                log.error(f"Main loop error: {e}")
                time.sleep(30)

    # ── Dashboard ────────────────────────────
    def generate_dashboard(self):
        st = self.state["stats"]
        trading_ok = True
        today = ist_now().strftime("%Y-%m-%d")
        daily_loss = abs(min(st["daily_pnl"].get(today, 0), 0))
        if daily_loss >= MAX_DAILY_LOSS:
            trading_ok = False

        risk_badge_text = "TRADING ACTIVE" if trading_ok else "LIMIT REACHED"
        risk_badge_cls = "badge-ok" if trading_ok else "badge-limit"

        css = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        :root{
        --bg:#0b0e11;--bg1:#13161b;--bg2:#1a1d24;--bg3:#22262f;
        --line:#2b2f3a;--text:#eaecef;--muted:#848e9c;--muted2:#5a6171;
        --green:#0ecb81;--red:#f6465d;--yellow:#f0b90b;--blue:#1890ff;
        --purple:#8b5cf6;--white:#ffffff;
        --font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;
        }
        body{background:var(--bg);color:var(--text);font-family:var(--font);padding:16px;min-height:100vh;}
        .header{display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid var(--line);margin-bottom:16px;}
        .header h1{font-size:20px;font-weight:800;color:var(--white);}
        .badge-ok{background:rgba(14,203,129,.12);color:var(--green);border:1px solid rgba(14,203,129,.3);padding:4px 12px;border-radius:4px;font-size:10px;font-weight:700;}
        .badge-limit{background:rgba(246,70,93,.12);color:var(--red);border:1px solid rgba(246,70,93,.3);padding:4px 12px;border-radius:4px;font-size:10px;font-weight:700;}
        .stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:16px;}
        .stat-card{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:12px;}
        .stat-label{font-size:9px;color:var(--muted);text-transform:uppercase;font-weight:600;letter-spacing:.06em;}
        .stat-val{font-size:18px;font-weight:800;font-family:var(--mono);margin-top:4px;}
        .monitor-card{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:8px;}
        .monitor-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;}
        .monitor-sym{font-size:14px;font-weight:700;color:var(--white);}
        .monitor-price{font-size:16px;font-weight:700;font-family:var(--mono);}
        .monitor-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
        .monitor-label{font-size:9px;color:var(--muted);text-transform:uppercase;}
        .monitor-val{font-size:12px;font-weight:600;font-family:var(--mono);}
        .monitor-bottom{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px;padding-top:8px;border-top:1px solid var(--line);}
        .pill-buy{background:rgba(14,203,129,.12);color:var(--green);border:1px solid rgba(14,203,129,.3);padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;}
        .pill-sell{background:rgba(246,70,93,.12);color:var(--red);border:1px solid rgba(246,70,93,.3);padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;}
        .pill-hold{background:rgba(24,144,255,.12);color:var(--blue);border:1px solid rgba(24,144,255,.3);padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;}
        .pill-scan{background:rgba(138,92,246,.12);color:var(--purple);border:1px solid rgba(138,92,246,.3);padding:3px 10px;border-radius:4px;font-size:10px;font-weight:700;}
        .pos-card{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:13px;margin-bottom:8px;position:relative;overflow:hidden;}
        .pos-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;}
        .pos-card.buy::before{background:var(--green);}
        .pos-card.sell::before{background:var(--red);}
        .pos-card-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
        .pos-pair{font-size:14px;font-weight:700;color:var(--white);}
        .pos-side{font-size:10px;font-weight:700;padding:3px 8px;border-radius:4px;}
        .pos-details{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;background:var(--bg3);border-radius:6px;padding:10px;margin-bottom:8px;}
        .pos-d-label{font-size:9px;color:var(--muted);text-transform:uppercase;font-weight:600;margin-bottom:3px;letter-spacing:.06em;}
        .pos-d-val{font-family:var(--mono);font-size:12px;font-weight:700;}
        .pos-footer{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
        .pos-margin-tag{font-size:10px;font-weight:700;color:var(--yellow);background:rgba(240,185,11,.1);border:1px solid rgba(240,185,11,.25);padding:3px 10px;border-radius:4px;font-family:var(--mono);}
        .trade-card{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:6px;}
        .trade-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;}
        .trade-pair{font-size:13px;font-weight:700;}
        .trade-pnl{font-size:14px;font-weight:800;font-family:var(--mono);}
        .pnl-pos{color:var(--green);}
        .pnl-neg{color:var(--red);}
        .section-title{font-size:13px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin:16px 0 8px;}
        footer{text-align:center;color:var(--muted2);font-size:10px;padding:24px 12px;border-top:1px solid var(--line);margin-top:12px;}
        """

        # Stats
        total_pnl = self.state.get("total_pnl", 0)
        wins = st.get("wins", 0)
        losses = st.get("losses", 0)
        win_rate = round((wins / (wins + losses) * 100), 1) if (wins + losses) > 0 else 0
        pnl_color = "var(--green)" if total_pnl >= 0 else "var(--red)"

        stats_html = (
            '<div class="stats-grid">'
            '<div class="stat-card"><div class="stat-label">Total PnL</div>'
            '<div class="stat-val" style="color:' + pnl_color + '">$' + str(round(total_pnl, 2)) + '</div></div>'
            '<div class="stat-card"><div class="stat-label">Win Rate</div>'
            '<div class="stat-val" style="color:var(--blue)">' + str(win_rate) + '%</div></div>'
            '<div class="stat-card"><div class="stat-label">Total Trades</div>'
            '<div class="stat-val">' + str(st.get("total_trades", 0)) + '</div></div>'
            '<div class="stat-card"><div class="stat-label">Best Trade</div>'
            '<div class="stat-val" style="color:var(--green)">$' + str(round(st.get("best_trade", 0), 2)) + '</div></div>'
            '<div class="stat-card"><div class="stat-label">Worst Trade</div>'
            '<div class="stat-val" style="color:var(--red)">$' + str(round(st.get("worst_trade", 0), 2)) + '</div></div>'
            '<div class="stat-card"><div class="stat-label">Daily PnL</div>'
            '<div class="stat-val">$' + str(round(st["daily_pnl"].get(today, 0), 2)) + '</div></div>'
            '</div>'
        )

        # Monitor cards
        monitor_rows = ""
        for d in self.dashboard:
            sym = d["symbol"]
            score = d["score"]
            sig = d["signal"]
            clean = sym.replace("/", "").replace("USDT", "")
            reasons = ", ".join(d.get("reasons", [])) or "Waiting..."
            tp_val = "$" + str(d["tp"]) if d.get("tp") else "-"
            sl_val = "$" + str(d["sl"]) if d.get("sl") else "-"
            entry_val = "$" + str(d["entry"]) if d.get("entry") and "HOLD" not in sig else "-"

            if "HOLD" in sig:
                pill = "<span class='pill-hold'>" + sig + "</span>"
            elif "BUY" in sig:
                pill = "<span class='pill-buy'>" + sig + "</span>"
            elif "SELL" in sig:
                pill = "<span class='pill-sell'>" + sig + "</span>"
            else:
                pill = "<span class='pill-scan'>SCANNING</span>"

            okx_sym = sym.replace("/", "-").lower() + "-swap"
            okx_url = "https://www.okx.com/trade-swap/" + okx_sym

            monitor_rows += (
                '<div class="monitor-card">'
                '<div class="monitor-top">'
                '<div><div class="monitor-sym">' + sym + '</div>'
                '<div style="font-size:9px;color:var(--muted);margin-top:1px;">'
                '<a href="' + okx_url + '" target="_blank" style="color:var(--muted);text-decoration:none;">OKX ↗</a></div></div>'
                '<span class="monitor-price">$' + str(d.get('price', '--')) + '</span>'
                '</div>'
                '<div class="monitor-row">'
                '<div><span class="monitor-label">RSI </span><span class="monitor-val">' + str(d.get("rsi", "--")) + '</span></div>'
                '<div>' + pill + '</div>'
                '<div><span class="monitor-label">Score </span>'
                '<span class="monitor-val" style="color:var(--green);">' + str(score) + '/10</span></div>'
                '</div>'
                '<div class="monitor-bottom">'
                '<div><div class="monitor-label">Entry</div><div class="monitor-val" style="color:var(--yellow);">' + entry_val + '</div></div>'
                '<div><div class="monitor-label">Take Profit</div><div class="monitor-val" style="color:var(--green);">' + tp_val + '</div></div>'
                '<div><div class="monitor-label">Stop Loss</div><div class="monitor-val" style="color:var(--red);">' + sl_val + '</div></div>'
                '</div>'
                '<div style="font-size:9px;color:var(--muted2);margin-top:6px;">' + reasons + '</div>'
                '</div>'
            )

        # Open positions
        positions_html = ""
        for sym, pos in self.state["active_positions"].items():
            p_side = pos.get("side", "buy")
            p_entry = pos.get("entry", 0)
            p_tp = pos.get("tp", 0)
            p_sl = pos.get("sl", 0)
            p_margin = pos.get("margin", 10)
            p_strategy = pos.get("strategy", "CLASSIC")
            p_reasons = ", ".join(pos.get("reasons", []))
            card_cls = "buy" if p_side == "buy" else "sell"
            side_cls = "pill-buy" if p_side == "buy" else "pill-sell"
            cur_price = self.state["last_prices"].get(sym.replace("_5m","").replace("_15m","").replace("_30m",""), p_entry)
            unrealized = round((cur_price - p_entry) * float(pos.get("qty", 1)) if p_side == "buy"
                               else (p_entry - cur_price) * float(pos.get("qty", 1)), 4)
            u_color = "var(--green)" if unrealized >= 0 else "var(--red)"

            positions_html += (
                '<div class="pos-card ' + card_cls + '">'
                '<div class="pos-card-top">'
                '<div class="pos-pair">' + sym + '</div>'
                '<span class="pos-side ' + side_cls + '">' + p_side.upper() + '</span>'
                '</div>'
                '<div class="pos-details">'
                '<div><div class="pos-d-label">Entry</div><div class="pos-d-val">$' + str(p_entry) + '</div></div>'
                '<div><div class="pos-d-label">TP</div><div class="pos-d-val" style="color:var(--green)">$' + str(p_tp) + '</div></div>'
                '<div><div class="pos-d-label">SL</div><div class="pos-d-val" style="color:var(--red)">$' + str(p_sl) + '</div></div>'
                '<div><div class="pos-d-label">PnL</div><div class="pos-d-val" style="color:' + u_color + '">$' + str(unrealized) + '</div></div>'
                '</div>'
                '<div class="pos-footer">'
                '<span class="pos-margin-tag">Margin: $' + str(p_margin) + '</span>'
                '<span style="font-size:9px;color:var(--purple);font-weight:700;">' + p_strategy + '</span>'
                '</div>'
                '<div style="font-size:9px;color:var(--muted2);margin-top:6px;">' + p_reasons + '</div>'
                '</div>'
            )

        if not positions_html:
            positions_html = '<div style="text-align:center;color:var(--muted);padding:20px;">No open positions</div>'

        # Trade history
        history_rows = ""
        trades = self.state.get("trades", [])[-20:]
        trades.reverse()
        for t in trades:
            pv = t.get("pnl", 0)
            pc = "pnl-pos" if pv >= 0 else "pnl-neg"
            badge = "pill-buy" if t.get("side", "") == "BUY" else "pill-sell"
            prefix = "+" if pv >= 0 else ""
            history_rows += (
                '<div class="trade-card">'
                '<div class="trade-top">'
                '<div style="display:flex;align-items:center;gap:8px;">'
                '<div class="trade-pair">' + str(t.get("symbol", "")) + '</div>'
                '<span class="' + badge + '">' + str(t.get("side", "")) + '</span>'
                '</div>'
                '<div class="trade-pnl ' + pc + '">' + prefix + '$' + str(round(pv, 2)) + '</div>'
                '</div>'
                '<div style="font-size:9px;color:var(--muted2);">'
                + str(t.get("time", "")) + ' | ' + str(t.get("result", ""))
                + ' | ' + str(t.get("strategy", ""))
                + '</div>'
                '</div>'
            )

        if not history_rows:
            history_rows = '<div class="trade-card" style="text-align:center;color:var(--muted);">No trades yet — bot is scanning...</div>'

        html = (
            '<!DOCTYPE html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>WhaleEngine Dashboard</title>'
            '<style>' + css + '</style></head><body>'
            '<div class="header"><h1>🐋 WhaleEngine</h1>'
            '<span class="' + risk_badge_cls + '">' + risk_badge_text + '</span></div>'
            + stats_html
            + '<div class="section-title">📡 Live Monitor</div>'
            + monitor_rows
            + '<div class="section-title">📊 Open Positions</div>'
            + positions_html
            + '<div class="section-title">📜 Trade History</div>'
            + history_rows
            + '<footer>WhaleEngine v13 + Quantum Scalp | Updated: ' + ist_str() + '</footer>'
            '</body></html>'
        )

        with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Dashboard saved — index.html")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    engine = WhaleEngine()
    engine.run()
