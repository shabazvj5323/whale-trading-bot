# main.py – GitHub 15-min Version (Advanced Liquidation Strategy)
import time
import logging
import os
import json
import ccxt
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================================
# CONFIG
# =====================================================================
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
INITIAL_CAPITAL = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE = 10
HISTORY_FILE = "history.json"

MAX_POSITIONS = 10
MIN_SCORE = 4
VOL_MULT = 1.5
BB_PERIOD = 20
BB_STD = 2.0
BB_FAST_P = 10
BB_FAST_S = 1.5
SR_PERIOD = 15
SR_THRESH = 0.003
SR_RETEST_THRESH = 0.002
ATR_PERIOD = 14

TP_MULT = 2.0
SL_MULT = 0.8
TRAIL_LOOKBACK = 3

# ---- Liquidation Hunting Parameters ----
LIQ_LOOKBACK = 20          # Candles for swing high/low
LIQ_ATR_MULT = 1.5         # Level 2 multiplier
LIQ_CASCADE_MULT = 2.0     # Level 3 (cascade)
LIQ_SWEEP_CONFIRM_BARS = 1 # Reversal candle count
ENABLE_OI_FUNDING = False  # Set True if Binance API key available

SCAN_INTERVAL = 900        # 15 minutes for GitHub

# =====================================================================
# LOGGING (unchanged)
# =====================================================================
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        return ist.strftime("%Y-%m-%d %H:%M:%S IST")

_handler = logging.StreamHandler()
_handler.setFormatter(ISTFormatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_handler]
log = logging.getLogger("WhaleTrader_Pro")

# =====================================================================
# HELPERS (same as before)
# =====================================================================
def ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def ist_short():
    return ist_now().strftime("%m-%d %H:%M")

def ema_calc(arr, period):
    k = 2.0 / (period + 1)
    r = [float(arr[0])]
    for p in arr[1:]:
        r.append(float(p) * k + r[-1] * (1 - k))
    return np.array(r)

def rsi_calc(closes, period=14):
    closes = closes.astype(float)
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[:period]) if len(gain) >= period else 0
    avg_loss = np.mean(loss[:period]) if len(loss) >= period else 0
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def atr_calc(highs, lows, closes, period):
    h, l, c = highs.astype(float), lows.astype(float), closes.astype(float)
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(c[:-1] - l[1:])))
    return float(np.mean(tr[-period:])) if len(tr) >= period else float(np.mean(tr))

def bollinger_calc(closes, period, std_mult):
    c = closes.astype(float)
    recent = c[-period:] if len(c) >= period else c
    sma = float(np.mean(recent))
    std = float(np.std(recent))
    return sma + std_mult * std, sma, sma - std_mult * std

def market_regime(closes, highs, lows):
    closes = closes.astype(float)
    if len(closes) < 50:
        return "SIDEWAYS"
    ema50 = ema_calc(closes, 50)[-1]
    ema200 = ema_calc(closes, min(100, len(closes)-1))[-1]
    if ema50 > ema200 * 1.002:
        regime = "BULL"
    elif ema50 < ema200 * 0.998:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS"
    return regime

def support_resistance_range(price, highs, lows, threshold=0.002):
    recent_high = np.max(highs[-10:])
    recent_low = np.min(lows[-10:])
    if abs(price - recent_high) / price < threshold:
        return True
    if abs(price - recent_low) / price < threshold:
        return True
    return False

def detect_engulfing(opens, closes):
    if len(opens) < 2: return None
    prev_body = closes[-2] - opens[-2]
    curr_body = closes[-1] - opens[-1]
    if prev_body < 0 and curr_body > 0 and curr_body > abs(prev_body):
        return "bullish"
    if prev_body > 0 and curr_body < 0 and abs(curr_body) > prev_body:
        return "bearish"
    return None

def detect_pin_bar(opens, highs, lows, closes):
    if len(highs) < 2: return None
    body = abs(closes[-1] - opens[-1])
    lower_wick = min(opens[-1], closes[-1]) - lows[-1]
    upper_wick = highs[-1] - max(opens[-1], closes[-1])
    if lower_wick > 2 * body and lower_wick > upper_wick:
        return "bullish"
    if upper_wick > 2 * body and upper_wick > lower_wick:
        return "bearish"
    return None

def detect_choch(highs, lows, closes, lookback=10):
    if len(highs) < lookback + 1:
        return None, None
    recent_high = np.max(highs[-lookback:-1])
    recent_low = np.min(lows[-lookback:-1])
    current_price = closes[-1]
    if current_price > recent_high:
        return "bullish", recent_high
    if current_price < recent_low:
        return "bearish", recent_low
    return None, None

# ---- Liquidation Hunting: Core Functions ----
def detect_liquidation_sweep(highs, lows, closes, opens, lookback=10, threshold=0.002):
    """
    Detect if price swept a recent swing high or low with a wick (wick = high > swing_high, close < swing_high).
    Returns direction ('bullish'/'bearish') and the swept level.
    """
    if len(highs) < lookback + 1:
        return None, None
    swing_high = np.max(highs[-lookback:-1])
    swing_low = np.min(lows[-lookback:-1])
    current_high = highs[-1]
    current_close = closes[-1]
    current_low = lows[-1]

    # Bullish sweep: price broke above resistance (wick) but closed back inside/ below
    if current_high > swing_high * (1 + threshold) and current_close < swing_high:
        return "bullish", swing_high
    # Bearish sweep: price broke below support (wick) but closed back inside/ above
    if current_low < swing_low * (1 - threshold) and current_close > swing_low:
        return "bearish", swing_low
    return None, None

def estimate_liquidation_levels(highs, lows, closes, atr, lookback=20):
    """Return list of (level, type) where type is 'short' or 'long'."""
    if len(highs) < lookback:
        return []
    swing_high = np.max(highs[-lookback:])
    swing_low = np.min(lows[-lookback:])
    levels = []
    # Level 1: swing extremes
    levels.append((swing_high, "short"))
    levels.append((swing_low, "long"))
    # Level 2: swing + ATR * 1.5
    levels.append((swing_high + atr * LIQ_ATR_MULT, "short"))
    levels.append((swing_low - atr * LIQ_ATR_MULT, "long"))
    # Level 3: swing + ATR * 2.0 (cascade)
    levels.append((swing_high + atr * LIQ_CASCADE_MULT, "short"))
    levels.append((swing_low - atr * LIQ_CASCADE_MULT, "long"))
    return levels

def get_binance_funding_rate(symbol):
    """Fetch funding rate from Binance futures (optional)."""
    try:
        clean = symbol.replace("/", "")
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={clean}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return float(data.get("lastFundingRate", 0))
    except:
        pass
    return None

# =====================================================================
# DATA FETCHING (same as before)
# =====================================================================
def fetch_binance_public(symbol, interval="5m", limit=120):
    try:
        clean = symbol.replace("/", "")
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": clean, "interval": interval, "limit": limit}
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            arr = np.array([[float(c[1]), float(c[2]), float(c[3]),
                             float(c[4]), float(c[5])] for c in data], dtype=float)
            return arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        return None
    except Exception as e:
        log.warning(f"Binance public failed {symbol}: {e}")
        return None

def fetch_okx_public(symbol, interval="5m", limit=120):
    try:
        clean = symbol.replace("/", "-")
        url = "https://www.okx.com/api/v5/market/candles"
        params = {"instId": clean + "-SWAP", "bar": interval, "limit": str(limit)}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                raw = data["data"][::-1]
                arr = np.array([[float(c[1]), float(c[2]), float(c[3]),
                                 float(c[4]), float(c[5])] for c in raw])
                return arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        return None
    except:
        return None

def fetch_candles(symbol, interval="5m", limit=120):
    result = fetch_binance_public(symbol, interval, limit)
    if result is not None:
        return result
    return fetch_okx_public(symbol, interval, limit)

def fetch_symbol_data_enhanced(symbol):
    with ThreadPoolExecutor(max_workers=4) as executor:
        f1h = executor.submit(fetch_candles, symbol, "1h", 100)
        f15m = executor.submit(fetch_candles, symbol, "15m", 120)
        f5m = executor.submit(fetch_candles, symbol, "5m", 120)
        f1m = executor.submit(fetch_candles, symbol, "1m", 60)
        r1h = f1h.result(); r15m = f15m.result(); r5m = f5m.result(); r1m = f1m.result()
    if r1h and r15m and r5m and r1m:
        return {
            '1h': {'o': r1h[0], 'h': r1h[1], 'l': r1h[2], 'c': r1h[3], 'v': r1h[4]},
            '15m': {'o': r15m[0], 'h': r15m[1], 'l': r15m[2], 'c': r15m[3], 'v': r15m[4]},
            '5m': {'o': r5m[0], 'h': r5m[1], 'l': r5m[2], 'c': r5m[3], 'v': r5m[4]},
            '1m': {'o': r1m[0], 'h': r1m[1], 'l': r1m[2], 'c': r1m[3], 'v': r1m[4]}
        }
    return None

# =====================================================================
# ENGINE
# =====================================================================
class WhaleEngine:
    def __init__(self):
        self.dashboard = []
        self._last_is_blast = False
        self.state = self.load_history()

    def load_history(self):
        default = {
            "total_pnl": 0.0,
            "active_positions": {},
            "trades": [],
            "last_prices": {},
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
            if not data["stats"]["daily_pnl"] or isinstance(data["stats"]["daily_pnl"], str):
                data["stats"]["daily_pnl"] = {}
                data["stats"]["daily_trades"] = {}
                for t in data.get("trades", []):
                    try:
                        time_str = t.get("time", "")
                        month_day = time_str.split()[0]
                        year = datetime.now().year
                        date_str = f"{year}-{month_day}"
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                        day_key = date_obj.strftime("%Y-%m-%d")
                    except:
                        continue
                    pnl = float(t.get("pnl", 0.0))
                    data["stats"]["daily_pnl"][day_key] = round(data["stats"]["daily_pnl"].get(day_key, 0.0) + pnl, 4)
                    data["stats"]["daily_trades"][day_key] = data["stats"]["daily_trades"].get(day_key, 0) + 1
                total = round(sum(t.get("pnl", 0.0) for t in data.get("trades", [])), 4)
                data["total_pnl"] = total
                log.info("History auto-repaired.")
            return data
        except Exception as e:
            log.warning(f"History load failed, using default: {e}")
            return default

    def save_history(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def cleanup_stale_positions(self):
        now = ist_now()
        to_remove = []
        for symbol, pos in list(self.state["active_positions"].items()):
            try:
                entry_time_str = pos.get("time", "")
                if not entry_time_str:
                    to_remove.append(symbol)
                    continue
                entry_dt = datetime.strptime(
                    f"{now.year}-{entry_time_str}", "%Y-%m-%d %H:%M"
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

    def can_trade(self, score=0):
        return True

    # ---- SCORING WITH LIQUIDATION HUNTING ----
    def score_signal(self, data, symbol):
        opens1 = data['1m']['o']; highs1 = data['1m']['h']; lows1 = data['1m']['l']; closes1 = data['1m']['c']; vols1 = data['1m']['v']
        opens5 = data['5m']['o']; highs5 = data['5m']['h']; lows5 = data['5m']['l']; closes5 = data['5m']['c']; vols5 = data['5m']['v']
        closes15 = data['15m']['c']
        price_1h = data['1h']['c'][-1]
        
        score, direction, reasons = 0, None, []
        is_blast = False
        price = float(closes1[-1])

        # ---- 1. VOLATILITY FILTER ----
        atr_val = atr_calc(highs5, lows5, closes5, ATR_PERIOD)
        atr_pct = (atr_val / price) * 100
        if atr_pct < 0.25:
            return 0, None, ["LOW VOL"], 0, 0, 4

        # ---- 2. MARKET REGIME ----
        regime = market_regime(closes5, highs5, lows5)
        reasons.append(regime)

        # ---- 3. BOLLINGER BANDS ----
        upper, mid, lower = bollinger_calc(closes5, BB_PERIOD, BB_STD)
        u_fast, m_fast, l_fast = bollinger_calc(closes5, BB_FAST_P, BB_FAST_S)

        vol_avg = float(np.mean(vols5[-15:-1])) if len(vols5) > 15 else float(np.mean(vols5))
        vol_now = float(vols5[-1])
        vol_ratio = vol_now / (vol_avg + 1e-9)
        vol_spike = vol_ratio >= VOL_MULT

        atr = atr_calc(highs5, lows5, closes5, ATR_PERIOD)
        body_now = float(closes1[-1]) - float(opens1[-1])
        avg_body = float(np.mean([abs(float(closes1[i])-float(opens1[i])) for i in range(-6,-1)]))

        # ---- 4. BB BLAST ----
        band_widths = []
        for i in range(-20, -1):
            try:
                u_i, m_i, l_i = bollinger_calc(closes5[:i], BB_PERIOD, BB_STD)
                band_widths.append(u_i - l_i)
            except:
                pass
        avg_band_width = float(np.mean(band_widths)) if band_widths else (upper - lower)
        band_width_now = upper - lower
        squeeze = band_width_now < avg_band_width * 0.70
        big_candle = abs(body_now) > avg_body * 2.5

        if squeeze and big_candle and vol_spike:
            if body_now > 0 and price > mid:
                score += 7; direction = "buy"; is_blast = True; reasons.append("BB BLAST UP")
            elif body_now < 0 and price < mid:
                score += 7; direction = "sell"; is_blast = True; reasons.append("BB BLAST DN")

        # ---- 5. REGIME SPECIFIC ----
        if regime in ("SIDEWAYS", "RANGING"):
            rsi = rsi_calc(closes5, 14)
            if not is_blast:
                if l_fast <= price <= lower and rsi < 30:
                    score += 5; direction = "buy"; reasons.append("DOUBLE BB + RSI<30")
                elif upper <= price <= u_fast and rsi > 70:
                    score += 5; direction = "sell"; reasons.append("DOUBLE BB + RSI>70")
            # CVD (cumulative volume delta)
            cvd = 0
            for i in range(-10, 0):
                if closes1[i] > opens1[i]:
                    cvd += vols1[i]
                else:
                    cvd -= vols1[i]
            range_high = np.max(highs5[-10:])
            range_low = np.min(lows5[-10:])
            in_range = range_low < price < range_high
            absorption = abs(cvd) < 200 and vol_spike and in_range

            if cvd > 300 and not absorption:
                score += 4; direction = direction or "buy"; reasons.append("CVD BUY")
            elif cvd < -300 and not absorption:
                score += 4; direction = direction or "sell"; reasons.append("CVD SELL")
            if absorption:
                reasons.append("ABSORPTION")

        elif regime == "BULL":
            if len(closes5) >= 30:
                e9 = ema_calc(closes5, 9)[-1]; e21 = ema_calc(closes5, 21)[-1]; e50 = ema_calc(closes5, 50)[-1]
                if e9 > e21 > e50:
                    score += 5; direction = direction or "buy"; reasons.append("EMA RIBBON BULL")
            try:
                typical = (highs5 + lows5 + closes5) / 3
                vwap = float(np.sum(typical * vols5) / (np.sum(vols5) + 1e-9))
                if price < vwap * 0.998 and price > vwap * 0.995:
                    score += 3; direction = "buy"; reasons.append("VWAP DIP")
            except:
                pass
            # ---- SR BREAKOUT (UPDATED) ----
            if len(highs5) >= SR_PERIOD:
                resistance = float(np.max(highs5[-SR_PERIOD:]))
                support = float(np.min(lows5[-SR_PERIOD:]))
                strong_candle = abs(closes1[-1] - opens1[-1]) > avg_body * 1.5
                retest_res = abs(price - resistance) / price < SR_RETEST_THRESH
                retest_sup = abs(price - support) / price < SR_RETEST_THRESH
                if price > resistance * (1 + SR_THRESH) and strong_candle and vol_spike:
                    score += 6; direction = direction or "buy"; reasons.append("SR BREAKOUT BULL (Strong+Vol)")
                elif price < support * (1 - SR_THRESH) and strong_candle and vol_spike:
                    score += 6; direction = direction or "sell"; reasons.append("SR BREAKOUT BEAR (Strong+Vol)")
                elif retest_res and strong_candle and vol_spike:
                    score += 4; direction = direction or "buy"; reasons.append("SR RETEST RES (Conf)")
                elif retest_sup and strong_candle and vol_spike:
                    score += 4; direction = direction or "sell"; reasons.append("SR RETEST SUP (Conf)")
                elif price > resistance * (1 + SR_THRESH) and vol_spike:
                    score += 3; direction = direction or "buy"; reasons.append("SR BREAKOUT UP (Vol)")
                elif price < support * (1 - SR_THRESH) and vol_spike:
                    score += 3; direction = direction or "sell"; reasons.append("SR BREAKOUT DN (Vol)")
            if price <= lower and vol_spike:
                score += 4; direction = direction or "buy"; reasons.append("BB DIP + VOL")

        elif regime == "BEAR":
            if len(closes5) >= 30:
                e9 = ema_calc(closes5, 9)[-1]; e21 = ema_calc(closes5, 21)[-1]; e50 = ema_calc(closes5, 50)[-1]
                if e9 < e21 < e50:
                    score += 5; direction = direction or "sell"; reasons.append("EMA RIBBON BEAR")
            try:
                typical = (highs5 + lows5 + closes5) / 3
                vwap = float(np.sum(typical * vols5) / (np.sum(vols5) + 1e-9))
                if price > vwap * 1.002 and price < vwap * 1.005:
                    score += 3; direction = "sell"; reasons.append("VWAP RALLY")
            except:
                pass
            if len(highs5) >= SR_PERIOD:
                resistance = float(np.max(highs5[-SR_PERIOD:]))
                support = float(np.min(lows5[-SR_PERIOD:]))
                strong_candle = abs(closes1[-1] - opens1[-1]) > avg_body * 1.5
                retest_res = abs(price - resistance) / price < SR_RETEST_THRESH
                retest_sup = abs(price - support) / price < SR_RETEST_THRESH
                if price < support * (1 - SR_THRESH) and strong_candle and vol_spike:
                    score += 6; direction = direction or "sell"; reasons.append("SR BREAKOUT BEAR (Strong+Vol)")
                elif price > resistance * (1 + SR_THRESH) and strong_candle and vol_spike:
                    score += 6; direction = direction or "buy"; reasons.append("SR BREAKOUT BULL (Strong+Vol)")
                elif retest_sup and strong_candle and vol_spike:
                    score += 4; direction = direction or "sell"; reasons.append("SR RETEST SUP (Conf)")
                elif retest_res and strong_candle and vol_spike:
                    score += 4; direction = direction or "buy"; reasons.append("SR RETEST RES (Conf)")
                elif price < support * (1 - SR_THRESH) and vol_spike:
                    score += 3; direction = direction or "sell"; reasons.append("SR BREAKOUT DN (Vol)")
                elif price > resistance * (1 + SR_THRESH) and vol_spike:
                    score += 3; direction = direction or "buy"; reasons.append("SR BREAKOUT UP (Vol)")
            if price >= upper and vol_spike:
                score += 4; direction = direction or "sell"; reasons.append("BB RALLY + VOL")

        # ---- 6. PRICE ACTION @ S/R ----
        if support_resistance_range(price, highs5, lows5, 0.002):
            engulf = detect_engulfing(opens1, closes1)
            if engulf == "bullish":
                score += 4; direction = direction or "buy"; reasons.append("ENGULFING @ S/R")
            elif engulf == "bearish":
                score += 4; direction = direction or "sell"; reasons.append("ENGULFING @ S/R")
            pin = detect_pin_bar(opens1, highs1, lows1, closes1)
            if pin == "bullish":
                score += 3; direction = direction or "buy"; reasons.append("PIN BAR @ S/R")
            elif pin == "bearish":
                score += 3; direction = direction or "sell"; reasons.append("PIN BAR @ S/R")

        # ---- 7. VOLUME BONUS ----
        if direction and vol_spike:
            score += 2; reasons.append("VOL " + str(round(vol_ratio, 1)) + "x")

        # ---- 8. COUNTER-TREND PENALTY ----
        if regime == "BULL" and direction == "sell": score -= 2; reasons.append("COUNTER-TREND")
        if regime == "BEAR" and direction == "buy": score -= 2; reasons.append("COUNTER-TREND")

        # ---- 9. LIQUIDATION HUNTING (NEW) ----
        # 9a. Sweep Detection
        sweep_dir, sweep_level = detect_liquidation_sweep(highs5, lows5, closes5, opens5, lookback=10, threshold=0.002)
        if sweep_dir == "bullish" and direction != "sell":
            score += 5
            direction = direction or "buy"
            reasons.append(f"LIQ SWEEP BULL @ {sweep_level:.2f}")
            # Check reversal candle confirmation
            if len(closes5) >= 2:
                prev_body = closes5[-2] - opens5[-2]
                curr_body = closes5[-1] - opens5[-1]
                if curr_body > 0 and abs(curr_body) > abs(prev_body) * 1.2:
                    score += 2
                    reasons.append("REVERSAL CANDLE CONFIRM")
        elif sweep_dir == "bearish" and direction != "buy":
            score += 5
            direction = direction or "sell"
            reasons.append(f"LIQ SWEEP BEAR @ {sweep_level:.2f}")
            if len(closes5) >= 2:
                prev_body = closes5[-2] - opens5[-2]
                curr_body = closes5[-1] - opens5[-1]
                if curr_body < 0 and abs(curr_body) > abs(prev_body) * 1.2:
                    score += 2
                    reasons.append("REVERSAL CANDLE CONFIRM")

        # 9b. Multiple Liquidation Levels (Cascade)
        liq_levels = estimate_liquidation_levels(highs5, lows5, closes5, atr, lookback=LIQ_LOOKBACK)
        for level, ltype in liq_levels:
            if abs(price - level) / price < 0.005:  # within 0.5%
                if ltype == "short" and direction != "sell":
                    score += 3
                    direction = direction or "buy"
                    reasons.append(f"LIQ SHORT ZONE @ {level:.2f}")
                elif ltype == "long" and direction != "buy":
                    score += 3
                    direction = direction or "sell"
                    reasons.append(f"LIQ LONG ZONE @ {level:.2f}")
                # Cascade: if level is beyond 1.5x ATR, extra bonus
                if abs(level - swing_high) > atr * LIQ_CASCADE_MULT:
                    score += 2
                    reasons.append("CASCADE POTENTIAL")

        # 9c. Optional OI + Funding (if enabled)
        if ENABLE_OI_FUNDING:
            funding = get_binance_funding_rate(symbol)
            if funding and funding > 0.001:  # high positive funding
                if direction == "sell":
                    score += 2
                    reasons.append("FUNDING HIGH (SHORT)")
                elif direction == "buy":
                    score -= 2  # avoid buying when funding high
            elif funding and funding < -0.001:
                if direction == "buy":
                    score += 2
                    reasons.append("FUNDING LOW (LONG)")

        # ---- 10. MACRO TREND ----
        ema20_1h = ema_calc(data['1h']['c'], 20)[-1]
        ema50_1h = ema_calc(data['1h']['c'], 50)[-1]
        macro_bull = price_1h > ema50_1h and ema20_1h > ema50_1h
        macro_bear = price_1h < ema50_1h and ema20_1h < ema50_1h
        if direction == "buy" and macro_bull:
            score += 2; reasons.append("MACRO BULL")
        elif direction == "sell" and macro_bear:
            score += 2; reasons.append("MACRO BEAR")

        self._last_is_blast = is_blast
        
        # ---- MIN_SCORE DYNAMIC ----
        if atr_pct > 0.8: min_score = 5
        elif atr_pct > 0.5: min_score = 4
        else: min_score = 3

        log.info(f"SCORE: {score} | DIR: {direction} | REASONS: {', '.join(reasons)}")
        return score, direction, reasons, upper, lower, min_score

    # ---- TP/SL ----
    def calculate_tp_sl(self, entry, direction, atr, price, lows1=None, highs1=None):
        if direction == "buy":
            tp = round(entry + atr * TP_MULT, 6)
            if lows1 is not None:
                swing_low = float(np.min(lows1[-5:]))
                sl = round(min(swing_low - atr * 0.2, entry - atr * SL_MULT), 6)
            else:
                sl = round(entry - atr * SL_MULT, 6)
        else:
            tp = round(entry - atr * TP_MULT, 6)
            if highs1 is not None:
                swing_high = float(np.max(highs1[-5:]))
                sl = round(max(swing_high + atr * 0.2, entry + atr * SL_MULT), 6)
            else:
                sl = round(entry + atr * SL_MULT, 6)
        return tp, sl

    # ---- POSITION MANAGEMENT ----
    def check_positions(self, symbol, current_price, closes_1m=None, highs_1m=None, lows_1m=None):
        if symbol not in self.state["active_positions"]:
            return
        pos = self.state["active_positions"][symbol]
        side = pos["side"]
        entry = float(pos["entry"])
        tp = float(pos["tp"])
        sl = float(pos["sl"])
        margin = float(pos.get("margin", MARGIN_PER_TRADE))
        qty = (margin * LEVERAGE) / entry
        price = float(current_price)

        if closes_1m is not None and len(closes_1m) >= TRAIL_LOOKBACK:
            atr = atr_calc(highs_1m, lows_1m, closes_1m, 14)
            if side == "buy":
                recent_lows = lows_1m[-TRAIL_LOOKBACK:]
                new_sl = float(np.min(recent_lows)) - atr * 0.2
                if new_sl > sl:
                    sl = new_sl
                    self.state["active_positions"][symbol]["sl"] = round(sl, 6)
            else:
                recent_highs = highs_1m[-TRAIL_LOOKBACK:]
                new_sl = float(np.max(recent_highs)) + atr * 0.2
                if new_sl < sl:
                    sl = new_sl
                    self.state["active_positions"][symbol]["sl"] = round(sl, 6)

        hit, reason, exit_p = False, "", price
        if side == "buy":
            if price >= tp: hit, reason, exit_p = True, "TP", tp
            elif price <= sl: hit, reason, exit_p = True, "TSL", sl
        else:
            if price <= tp: hit, reason, exit_p = True, "TP", tp
            elif price >= sl: hit, reason, exit_p = True, "TSL", sl

        if hit:
            if side == "buy": pnl = (exit_p - entry) * qty
            else: pnl = (entry - exit_p) * qty
            pnl = round(pnl, 4)
            self.state["total_pnl"] = round(self.state["total_pnl"] + pnl, 4)
            st = self.state["stats"]
            st["total_trades"] += 1
            if pnl > 0:
                st["wins"] += 1
                st["best_trade"] = round(max(st["best_trade"], pnl), 4)
            else:
                st["losses"] += 1
                st["worst_trade"] = round(min(st["worst_trade"], pnl), 4)
            today = ist_now().strftime("%Y-%m-%d")
            st["daily_pnl"][today] = round(st["daily_pnl"].get(today, 0) + pnl, 4)
            st["daily_trades"][today] = st["daily_trades"].get(today, 0) + 1
            exit_time_str = ist_short()
            pos_reasons = pos.get("reasons", [])
            strategy_name = "UNKNOWN"
            for strat_key in ["BB BLAST", "BB BOUNCE", "DOUBLE BB", "SR BREAKOUT",
                               "VWAP", "EMA RIBBON", "ORDER FLOW", "DARVAS", "HEDGE",
                               "ENGULFING", "PIN BAR", "INSIDE BAR", "OB", "FVG", "LIQ SWEEP",
                               "CHoCH", "MACRO", "CVD", "LIQ SWEEP", "CASCADE"]:
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
                "score": pos.get("score", 0),
                "margin": margin,
                "strategy": strategy_name,
                "reasons": pos_reasons,
            })
            del self.state["active_positions"][symbol]
            icon = "PROFIT" if pnl > 0 else "LOSS"
            log.info(f"{icon} {symbol} {side.upper()} | Entry:${entry} Exit:${exit_p} PnL:${pnl}")
            self.save_history()

    # ---- STRATEGY 10 (unchanged) ----
    def strategy10_ema_crossover(self, symbol, opens, highs, lows, closes, vols, timeframe="5m"):
        try:
            if len(closes) < 30: return None, 0, []
            e10 = ema_calc(closes, 10)[-1]
            e20 = ema_calc(closes, 20)[-1]
            e30 = ema_calc(closes, 30)[-1]
            e10_prev = ema_calc(closes[:-1], 10)[-1]
            e20_prev = ema_calc(closes[:-1], 20)[-1]
            e30_prev = ema_calc(closes[:-1], 30)[-1]
            bull_now = e10 > e20 > e30
            bull_prev = e10_prev > e20_prev > e30_prev
            bear_now = e10 < e20 < e30
            bear_prev = e10_prev < e20_prev < e30_prev
            fresh_bull = bull_now and not bull_prev
            fresh_bear = bear_now and not bear_prev
            if not fresh_bull and not fresh_bear:
                return None, 0, []
            body_now = float(closes[-1]) - float(opens[-1])
            green_candle = body_now > 0
            red_candle = body_now < 0
            vol_now = float(vols[-1])
            vol_prev = float(vols[-2]) if len(vols) >= 2 else vol_now
            vol_spike = vol_now > vol_prev * 4.0
            if fresh_bull and green_candle and vol_spike:
                return "buy", 8, [f"S10 BULL CROSS {timeframe}"]
            if fresh_bear and red_candle and vol_spike:
                return "sell", 8, [f"S10 BEAR CROSS {timeframe}"]
        except Exception as ex:
            log.warning(f"S10 error {symbol}: {ex}")
        return None, 0, []

    def run_strategy10(self, symbol):
        s10_key = symbol + "_S10"
        if s10_key in self.state["active_positions"]:
            pos = self.state["active_positions"][s10_key]
            price = self.state["last_prices"].get(symbol, 0)
            if price:
                side = pos["side"]
                entry = float(pos["entry"])
                tp = float(pos["tp"])
                sl = float(pos["sl"])
                margin = float(pos.get("margin", MARGIN_PER_TRADE))
                qty = (margin * LEVERAGE) / entry
                hit, reason, exit_p = False, "", price
                if side == "buy":
                    if price >= tp: hit, reason, exit_p = True, "TP", tp
                    elif price <= sl: hit, reason, exit_p = True, "SL", sl
                else:
                    if price <= tp: hit, reason, exit_p = True, "TP", tp
                    elif price >= sl: hit, reason, exit_p = True, "SL", sl
                if hit:
                    pnl = round((exit_p - entry) * qty if side == "buy" else (entry - exit_p) * qty, 4)
                    self.state["total_pnl"] = round(self.state["total_pnl"] + pnl, 4)
                    st = self.state["stats"]
                    st["total_trades"] += 1
                    if pnl > 0: st["wins"] += 1
                    else: st["losses"] += 1
                    today = ist_now().strftime("%Y-%m-%d")
                    st["daily_pnl"][today] = round(st["daily_pnl"].get(today, 0) + pnl, 4)
                    st["daily_trades"][today] = st["daily_trades"].get(today, 0) + 1
                    self.state["trades"].append({
                        "time": ist_short(),
                        "symbol": s10_key,
                        "side": side.upper(),
                        "entry": round(entry, 6),
                        "exit": round(exit_p, 6),
                        "pnl": pnl,
                        "result": reason,
                        "score": pos.get("score", 8),
                        "margin": margin,
                        "strategy": "EMA CROSSOVER",
                        "reasons": pos.get("reasons", []),
                    })
                    del self.state["active_positions"][s10_key]
                    icon = "PROFIT" if pnl > 0 else "LOSS"
                    log.info(f"S10 {icon}: {s10_key} | PnL:${pnl}")
                    self.save_history()
            return
        for tf in ["5m", "15m"]:
            try:
                limit = 80 if tf == "5m" else 60
                result = fetch_candles(symbol, tf, limit)
                if result is None:
                    continue
                opens, highs, lows, closes, vols = result
                direction, score, reasons = self.strategy10_ema_crossover(
                    symbol, opens, highs, lows, closes, vols, tf
                )
                if direction and score >= 6 and self.can_trade(score):
                    price = float(closes[-1])
                    av = atr_calc(highs, lows, closes, ATR_PERIOD)
                    if direction == "buy":
                        tp = round(price + av * 2.5, 6)
                        swing_low = float(np.min(lows[-10:]))
                        sl = round(min(swing_low - av * 0.2, price - av * 0.8), 6)
                    else:
                        tp = round(price - av * 2.5, 6)
                        swing_high = float(np.max(highs[-10:]))
                        sl = round(max(swing_high + av * 0.2, price + av * 0.8), 6)
                    self.state["active_positions"][s10_key] = {
                        "side": direction,
                        "entry": price,
                        "tp": tp,
                        "sl": sl,
                        "margin": MARGIN_PER_TRADE,
                        "score": score,
                        "reasons": reasons,
                        "time": ist_short(),
                        "strategy10": True,
                        "timeframe": tf,
                    }
                    self.state["last_prices"][symbol] = price
                    self.save_history()
                    log.info(f"S10 NEW TRADE: {s10_key} {direction.upper()} [{tf}] | Score:{score} | TP:{tp} | SL:{sl}")
                    break
            except Exception as ex:
                log.warning(f"S10 run error {symbol} [{tf}]: {ex}")

    def check_hedge_positions(self):
        pass

    # ---- MAIN SCAN ----
    def scan_market(self):
        log.info("=== Starting market scan ===")
        self.dashboard = []
        self.cleanup_stale_positions()
        self.check_hedge_positions()

        for sym10 in SYMBOLS:
            try:
                self.run_strategy10(sym10)
            except Exception as ex:
                log.warning(f"Strategy 10 outer error {sym10}: {ex}")

        symbol_data = {}
        with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as executor:
            future_to_symbol = {executor.submit(fetch_symbol_data_enhanced, sym): sym for sym in SYMBOLS}
            for future in as_completed(future_to_symbol):
                sym = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        symbol_data[sym] = data
                    else:
                        log.warning(f"Failed to fetch data for {sym}")
                except Exception as e:
                    log.error(f"Error fetching {sym}: {e}")

        for symbol in SYMBOLS:
            try:
                if symbol not in symbol_data:
                    self.dashboard.append({"symbol": symbol, "price": self.state["last_prices"].get(symbol, 0),
                                           "signal": "SCANNING", "score": 0,
                                           "entry": None, "tp": None, "sl": None,
                                           "reasons": ["Data unavailable"]})
                    continue

                data = symbol_data[symbol]
                opens1 = data['1m']['o']; highs1 = data['1m']['h']; lows1 = data['1m']['l']; closes1 = data['1m']['c']; vols1 = data['1m']['v']
                opens5 = data['5m']['o']; highs5 = data['5m']['h']; lows5 = data['5m']['l']; closes5 = data['5m']['c']; vols5 = data['5m']['v']
                closes15 = data['15m']['c']

                current_price = float(closes1[-1])
                self.check_positions(symbol, current_price, closes1, highs1, lows1)

                price = round(float(closes1[-1]), 6)
                self.state["last_prices"][symbol] = price
                if "last_prices_history" not in self.state:
                    self.state["last_prices_history"] = {}
                hist = self.state["last_prices_history"].get(symbol, [])
                hist.append(price)
                self.state["last_prices_history"][symbol] = hist[-20:]

                if symbol in self.state["active_positions"]:
                    pos = self.state["active_positions"][symbol]
                    self.dashboard.append({"symbol": symbol, "price": price,
                                           "signal": "HOLDING " + pos["side"].upper(),
                                           "score": pos.get("score", 0),
                                           "entry": pos["entry"],
                                           "tp": pos["tp"],
                                           "sl": pos["sl"],
                                           "reasons": pos.get("reasons", [])})
                    continue

                score, direction, reasons, upper, lower, adaptive_min = self.score_signal(data, symbol)

                log.info(f"{symbol} | ${price} | Score:{score}/10 | Regime:{reasons[0] if reasons else '?'} | Min:{adaptive_min}")

                # ---- DARVAS (unchanged) ----
                darvas_sig, box_top, box_bot, darvas_sl, darvas_reasons = darvas_box_strategy(
                    opens5, highs5, lows5, closes5, vols5)
                if darvas_sig and darvas_sig == direction:
                    score = min(score + 2, 10)
                    reasons = reasons + darvas_reasons
                    log.info(f"DARVAS CONFIRM {symbol} {darvas_sig.upper()}")
                elif darvas_sig and not direction and darvas_sl:
                    direction = darvas_sig
                    score = max(score, 5)
                    reasons = darvas_reasons
                    log.info(f"DARVAS SIGNAL {symbol} {darvas_sig.upper()}")

                # ---- ENTRY ----
                if direction and score >= adaptive_min and self.can_trade(score):
                    is_blast = getattr(self, "_last_is_blast", False)
                    margin = MARGIN_BLAST if is_blast else MARGIN_PER_TRADE
                    av = atr_calc(highs5, lows5, closes5, ATR_PERIOD)

                    if direction == "buy":
                        tp, sl = self.calculate_tp_sl(price, direction, av, price, lows1, highs1)
                    else:
                        tp, sl = self.calculate_tp_sl(price, direction, av, price, lows1, highs1)

                    self.state["active_positions"][symbol] = {
                        "side": direction,
                        "entry": price,
                        "tp": tp,
                        "sl": sl,
                        "margin": margin,
                        "score": score,
                        "reasons": reasons,
                        "time": ist_short(),
                    }
                    self.save_history()
                    log.info(f"NEW TRADE: {symbol} {direction.upper()} | Score:{score} | Margin:${margin} | TP:{tp} | SL:{sl}")

                # ---- DASHBOARD ENTRIES ----
                s10_key = symbol + "_S10"
                if s10_key in self.state["active_positions"]:
                    sp = self.state["active_positions"][s10_key]
                    self.dashboard.append({"symbol": s10_key, "price": price,
                                           "signal": "S10 " + sp["side"].upper(),
                                           "score": sp.get("score", 8),
                                           "entry": sp["entry"],
                                           "tp": sp["tp"],
                                           "sl": sp["sl"],
                                           "margin": sp.get("margin", MARGIN_PER_TRADE),
                                           "reasons": sp.get("reasons", [])})

                if symbol in self.state["active_positions"]:
                    pos = self.state["active_positions"][symbol]
                    self.dashboard.append({"symbol": symbol, "price": price,
                                           "signal": "HOLDING " + pos["side"].upper(),
                                           "score": pos.get("score", score),
                                           "entry": pos["entry"],
                                           "tp": pos["tp"],
                                           "sl": pos["sl"],
                                           "margin": pos.get("margin", MARGIN_PER_TRADE),
                                           "reasons": pos.get("reasons", reasons)})
                else:
                    self.dashboard.append({"symbol": symbol,
                                           "price": price,
                                           "signal": direction.upper() + " " + str(score) + "/10" if direction else "SCANNING",
                                           "score": score,
                                           "entry": None,
                                           "tp": None,
                                           "sl": None,
                                           "reasons": reasons})
            except Exception as e:
                log.error(f"Error {symbol}: {e}")
                self.dashboard.append({"symbol": symbol, "price": 0, "signal": "ERROR",
                                       "score": 0, "entry": 0, "tp": None, "sl": None, "reasons": [str(e)]})

        self.save_history()
        self.build_dashboard()
        log.info(f"Scan complete. Next scan in {SCAN_INTERVAL//60} minutes.")

    # ---- DASHBOARD (truncated for brevity – reuse your full version) ----
    def build_dashboard(self):
        # Placeholder – you already have the full build_dashboard function from previous code.
        # To keep the answer concise, I'm assuming you'll copy it from your existing file.
        # If not, let me know and I'll paste it.
        pass

    # ---- MAIN LOOP ----
    def run(self):
        log.info(f"WhaleTrader Pro (Liquidation Hunting) – Scanning every {SCAN_INTERVAL//60} minutes.")
        while True:
            try:
                self.scan_market()
                time.sleep(SCAN_INTERVAL)
            except KeyboardInterrupt:
                log.info("Bot stopped by user. Exiting...")
                break
            except Exception as e:
                log.error(f"Unexpected error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    engine = WhaleEngine()
    engine.run()
