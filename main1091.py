# main.py – Har 15 minute par auto-run (Ctrl+C se stop)
import time
import logging
import os
import json
import ccxt
import numpy as np
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# ── Logging ──
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        return ist.strftime("%Y-%m-%d %H:%M:%S IST")

_handler = logging.StreamHandler()
_handler.setFormatter(ISTFormatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_handler]
log = logging.getLogger("WhaleTrader_Pro")

# ═══════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════
SYMBOLS          = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
INITIAL_CAPITAL  = 1000.0
MARGIN_PER_TRADE = 100.0
MARGIN_BLAST     = 100.0
LEVERAGE         = 10
HISTORY_FILE     = "history.json"

MIN_SCORE   = 4
VOL_MULT    = 1.3
BB_PERIOD   = 20
BB_STD      = 2.0
BB_FAST_P   = 10
BB_FAST_S   = 1.5
SR_PERIOD   = 20
SR_THRESH   = 0.003
ATR_PERIOD  = 14

TP_MULT     = 1.5
SL_MULT     = 0.5
TRAIL_LOOKBACK = 5

HEDGE_ENABLED   = True
HEDGE_SUFFIX    = "_HEDGE"

EMA10_FAST      = 20
EMA10_MID       = 30
EMA10_SLOW      = 50
EMA10_VOL_MULT  = 5.0
EMA10_TIMEFRAMES = ["1m", "5m", "15m"]
MAX_POSITIONS   = 10

# ── LOOP INTERVAL (seconds) ──
LOOP_INTERVAL = 900   # 15 minutes

# ═══════════════════════════════════════════════
#  HELPERS (same as before)
# ═══════════════════════════════════════════════
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
    ema50  = ema_calc(closes, 50)[-1]
    ema200 = ema_calc(closes, min(100, len(closes)-1))[-1]
    if ema50 > ema200 * 1.002:
        regime = "BULL"
    elif ema50 < ema200 * 0.998:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS"
    return regime, 0.0

# ── PRICE ACTION & SMC (corrected) ──
def detect_engulfing(opens, closes):
    if len(opens) < 2 or len(closes) < 2:
        return None
    prev_body = closes[-2] - opens[-2]
    curr_body = closes[-1] - opens[-1]
    if prev_body < 0 and curr_body > 0 and curr_body > abs(prev_body) and closes[-1] > opens[-2]:
        return "bullish"
    if prev_body > 0 and curr_body < 0 and abs(curr_body) > prev_body and closes[-1] < opens[-2]:
        return "bearish"
    return None

def detect_pin_bar(opens, highs, lows, closes):
    if len(highs) < 2:
        return None
    body = abs(closes[-1] - highs[-2])
    lower_wick = min(opens[-1], closes[-1]) - lows[-1]
    upper_wick = highs[-1] - max(opens[-1], closes[-1])
    if lower_wick > 2 * body and lower_wick > upper_wick:
        return "bullish"
    if upper_wick > 2 * body and upper_wick > lower_wick:
        return "bearish"
    return None

def detect_inside_bar(highs, lows):
    if len(highs) < 2:
        return False
    return highs[-1] < highs[-2] and lows[-1] > lows[-2]

def detect_order_block(highs, lows, closes):
    if len(highs) < 6:
        return None
    range_high = np.max(highs[-6:-1])
    range_low = np.min(lows[-6:-1])
    price = closes[-1]
    if price > range_high:
        return "bullish"
    if price < range_low:
        return "bearish"
    return None

def detect_fair_value_gap(highs, lows, closes):
    if len(highs) < 3:
        return None
    if lows[-1] > highs[-2]:
        return "bullish"
    if highs[-1] < lows[-2]:
        return "bearish"
    return None

def detect_liquidity_sweep(opens, highs, lows, closes):
    if len(highs) < 10:
        return None
    prev_high = np.max(highs[-10:-1])
    prev_low = np.min(lows[-10:-1])
    price = closes[-1]
    if price > prev_high and (closes[-1] - opens[-1]) > 0:
        return "bullish"
    if price < prev_low and (closes[-1] - opens[-1]) < 0:
        return "bearish"
    return None

def darvas_box_strategy(opens, highs, lows, closes, vols, box_period=20, vol_period=20):
    if len(closes) < box_period + vol_period:
        return None, None, None, None, []
    closes = closes.astype(float)
    highs  = highs.astype(float)
    lows   = lows.astype(float)
    vols   = vols.astype(float)
    box_highs = highs[-box_period:]
    box_lows  = lows[-box_period:]
    resistance = float(np.max(box_highs))
    support    = float(np.min(box_lows))
    current_close  = float(closes[-1])
    current_volume = float(vols[-1])
    vol_ma = float(np.mean(vols[-vol_period:]))
    breakout_up   = current_close > resistance
    breakout_down = current_close < support
    trail_sl = None
    signal   = None
    reasons  = []
    if breakout_up:
        signal   = "buy"
        trail_sl = round(support, 6)
        reasons  = ["DARVAS BOX", "Break: $" + str(round(resistance, 4)), "VOL " + str(round(current_volume / vol_ma, 1)) + "x"]
    elif breakout_down:
        signal   = "sell"
        trail_sl = round(resistance, 6)
        reasons  = ["DARVAS BOX", "Break: $" + str(round(support, 4)), "VOL " + str(round(current_volume / vol_ma, 1)) + "x"]
    return signal, round(resistance, 6), round(support, 6), trail_sl, reasons

# ═══════════════════════════════════════════════
#  ENGINE (with loop)
# ═══════════════════════════════════════════════
class WhaleEngine:
    def __init__(self):
        self.dashboard = []
        self._last_is_blast = False
        api_key    = os.getenv("BINANCE_API_KEY")
        secret_key = os.getenv("BINANCE_SECRET_KEY")
        if not api_key or not secret_key:
            self.mock = True
            log.warning("Mock mode — no API keys")
        else:
            self.mock = False
            self.exchange = ccxt.binance({
                "apiKey": api_key,
                "secret": secret_key,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            self.exchange.set_sandbox_mode(False)
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
            # Auto-repair daily_pnl if corrupted
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

    # ── Market data ──
    COINGECKO_IDS = {
        "BTC/USDT":  "bitcoin",
        "ETH/USDT":  "ethereum",
        "SOL/USDT":  "solana",
        "BNB/USDT":  "binancecoin",
        "XRP/USDT":  "ripple",
        "DOGE/USDT": "dogecoin",
    }
    CG_API_KEY = os.environ.get("COINGECKO_API_KEY", "")

    def _cg_params(self, extra=None):
        p = extra.copy() if extra else {}
        if self.CG_API_KEY:
            p["x_cg_demo_api_key"] = self.CG_API_KEY
        return p

    def prefetch_all_volumes(self):
        try:
            ids = ",".join(self.COINGECKO_IDS.values())
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = self._cg_params({
                "vs_currency": "usd",
                "ids": ids,
                "per_page": 20,
                "page": 1,
            })
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            vol_map = {}
            for item in data:
                vol_map[item["id"]] = float(item.get("total_volume", 1000))
            return vol_map
        except Exception as e:
            log.warning(f"Volume prefetch failed: {e}")
            return {}

    def fetch_okx_public(self, symbol, interval="5m", limit=120):
        try:
            clean = symbol.replace("/", "-")
            url = "https://www.okx.com/api/v5/market/candles"
            params = {
                "instId": clean + "-SWAP",
                "bar": interval,
                "limit": str(limit)
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "0" and data.get("data"):
                    raw = data["data"][::-1]
                    arr = np.array([[float(c[1]), float(c[2]), float(c[3]),
                                     float(c[4]), float(c[5])] for c in raw])
                    return arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
            log.warning(f"OKX public {resp.status_code} for {symbol}")
            return None
        except Exception as e:
            log.warning(f"OKX public failed {symbol}: {e}")
            return None

    def fetch_symbol_data(self, symbol):
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_1m = executor.submit(self.fetch_okx_public, symbol, "1m", 60)
            future_5m = executor.submit(self.fetch_okx_public, symbol, "5m", 120)
            future_15m = executor.submit(self.fetch_okx_public, symbol, "15m", 60)
            result_1m = future_1m.result()
            result_5m = future_5m.result()
            result_15m = future_15m.result()
        if result_1m and result_5m and result_15m:
            opens1, highs1, lows1, closes1, vols1 = result_1m
            opens5, highs5, lows5, closes5, vols5 = result_5m
            closes15 = result_15m[3]
            return opens1, highs1, lows1, closes1, vols1, opens5, highs5, lows5, closes5, vols5, closes15
        return None

    # ── SCORING ──
    def score_signal(self, opens1, highs1, lows1, closes1, vols1,
                           opens5, highs5, lows5, closes5, vols5, closes15):
        score, direction, reasons = 0, None, []
        is_blast = False
        price  = float(closes1[-1])
        regime, _ = market_regime(closes5, highs5, lows5)
        reasons.append(regime)
        upper, mid, lower = bollinger_calc(closes5, BB_PERIOD, BB_STD)
        u_fast, m_fast, l_fast = bollinger_calc(closes5, BB_FAST_P, BB_FAST_S)
        vol_avg   = float(np.mean(vols5[-15:-1])) if len(vols5) > 15 else float(np.mean(vols5))
        vol_now   = float(vols5[-1])
        vol_ratio = vol_now / (vol_avg + 1e-9)
        vol_spike = vol_ratio >= VOL_MULT
        atr       = atr_calc(highs5, lows5, closes5, ATR_PERIOD)
        body_now  = float(closes1[-1]) - float(opens1[-1])
        avg_body  = float(np.mean([abs(float(closes1[i])-float(opens1[i])) for i in range(-6,-1)]))
        # BB Blast
        band_widths = []
        for i in range(-20, -1):
            try:
                u_i, m_i, l_i = bollinger_calc(closes5[:i], BB_PERIOD, BB_STD)
                band_widths.append(u_i - l_i)
            except:
                pass
        avg_band_width = float(np.mean(band_widths)) if band_widths else (upper - lower)
        band_width_now = upper - lower
        squeeze    = band_width_now < avg_band_width * 0.75
        big_candle = abs(body_now) > avg_body * 2.0
        if squeeze and big_candle:
            if body_now > 0 and price > mid:
                score += 6; direction = "buy"; is_blast = True; reasons.append("BB BLAST UP 🚀")
            elif body_now < 0 and price < mid:
                score += 6; direction = "sell"; is_blast = True; reasons.append("BB BLAST DN 🚀")
        # Regime specific
        if regime in ("SIDEWAYS", "RANGING"):
            if not is_blast:
                if l_fast <= price <= lower:
                    score += 4; direction = "buy"; reasons.append("DOUBLE BB BUY ZONE")
                elif upper <= price <= u_fast:
                    score += 4; direction = "sell"; reasons.append("DOUBLE BB SELL ZONE")
            try:
                buy_pressure = sell_pressure = 0.0
                for i in range(-10, 0):
                    body = float(closes1[i]) - float(opens1[i])
                    vol  = float(vols1[i])
                    if body > 0: buy_pressure  += abs(body) * vol
                    else:        sell_pressure += abs(body) * vol
                buy_ratio = buy_pressure / (buy_pressure + sell_pressure + 1e-9)
                if buy_ratio > 0.60:
                    score += 2; direction = direction or "buy"; reasons.append("ORDER FLOW BUY")
                elif buy_ratio < 0.40:
                    score += 2; direction = direction or "sell"; reasons.append("ORDER FLOW SELL")
            except: pass
        elif regime == "BULL":
            try:
                if len(closes5) >= 50:
                    e9 = ema_calc(closes5,9)[-1]; e21=ema_calc(closes5,21)[-1]; e50=ema_calc(closes5,50)[-1]
                    if e9 > e21 > e50:
                        score += 4; direction = direction or "buy"; reasons.append("EMA RIBBON BULL")
            except: pass
            try:
                typical = (highs5 + lows5 + closes5) / 3
                vwap = float(np.sum(typical * vols5) / (np.sum(vols5) + 1e-9))
                if price < vwap * 0.999:
                    score += 3; direction = "buy"; reasons.append("VWAP BUY")
                elif price > vwap * 1.003:
                    if direction == "sell":
                        direction = None; score = max(0, score - 4)
            except: pass
            if len(highs5) >= SR_PERIOD:
                resistance = float(np.max(highs5[-SR_PERIOD:]))
                support    = float(np.min(lows5[-SR_PERIOD:]))
                if price > resistance * (1 + SR_THRESH):
                    score += 4; direction = direction or "buy"; reasons.append("SR BREAKOUT UP")
                    if vol_spike: score += 2; reasons.append("VOL CONFIRM")
                elif price < support * (1 - SR_THRESH):
                    score += 2; direction = direction or "sell"; reasons.append("SR BREAKOUT DN")
            if price <= lower:
                score += 3; direction = direction or "buy"; reasons.append("BB DIP BUY")
        elif regime == "BEAR":
            try:
                if len(closes5) >= 50:
                    e9 = ema_calc(closes5,9)[-1]; e21=ema_calc(closes5,21)[-1]; e50=ema_calc(closes5,50)[-1]
                    if e9 < e21 < e50:
                        score += 4; direction = direction or "sell"; reasons.append("EMA RIBBON BEAR")
            except: pass
            try:
                typical = (highs5 + lows5 + closes5) / 3
                vwap = float(np.sum(typical * vols5) / (np.sum(vols5) + 1e-9))
                if price > vwap * 1.001:
                    score += 3; direction = "sell"; reasons.append("VWAP SELL")
                elif price < vwap * 0.997:
                    if direction == "buy":
                        direction = None; score = max(0, score - 4)
            except: pass
            if len(highs5) >= SR_PERIOD:
                resistance = float(np.max(highs5[-SR_PERIOD:]))
                support    = float(np.min(lows5[-SR_PERIOD:]))
                if price < support * (1 - SR_THRESH):
                    score += 4; direction = direction or "sell"; reasons.append("SR BREAKOUT DN")
                    if vol_spike: score += 2; reasons.append("VOL CONFIRM")
                elif price > resistance * (1 + SR_THRESH):
                    score += 2; direction = direction or "buy"; reasons.append("SR BREAKOUT UP")
            if price >= upper:
                score += 3; direction = direction or "sell"; reasons.append("BB RALLY SELL")
        # PA / SMC
        engulf = detect_engulfing(opens1, closes1)
        if engulf == "bullish": score += 3; direction = direction or "buy"; reasons.append("ENGULFING BULL")
        elif engulf == "bearish": score += 3; direction = direction or "sell"; reasons.append("ENGULFING BEAR")
        pin = detect_pin_bar(opens1, highs1, lows1, closes1)
        if pin == "bullish": score += 2; direction = direction or "buy"; reasons.append("PIN BAR BULL")
        elif pin == "bearish": score += 2; direction = direction or "sell"; reasons.append("PIN BAR BEAR")
        if detect_inside_bar(highs1, lows1):
            if price > highs1[-2]:
                score += 2; direction = direction or "buy"; reasons.append("INSIDE BAR UP")
            elif price < lows1[-2]:
                score += 2; direction = direction or "sell"; reasons.append("INSIDE BAR DN")
        ob = detect_order_block(highs1, lows1, closes1)
        if ob == "bullish": score += 4; direction = direction or "buy"; reasons.append("OB BREAKOUT")
        elif ob == "bearish": score += 4; direction = direction or "sell"; reasons.append("OB BREAKDOWN")
        fvg = detect_fair_value_gap(highs1, lows1, closes1)
        if fvg == "bullish": score += 3; direction = direction or "buy"; reasons.append("FVG BUY")
        elif fvg == "bearish": score += 3; direction = direction or "sell"; reasons.append("FVG SELL")
        sweep = detect_liquidity_sweep(opens1, highs1, lows1, closes1)
        if sweep == "bullish": score += 4; direction = direction or "buy"; reasons.append("LIQ SWEEP UP")
        elif sweep == "bearish": score += 4; direction = direction or "sell"; reasons.append("LIQ SWEEP DN")
        if direction and vol_spike and "VOL" not in str(reasons):
            score += 2; reasons.append("VOL " + str(round(vol_ratio, 1)) + "x")
        self._last_is_blast = is_blast
        return score, direction, 0, upper, lower, reasons, MIN_SCORE

    # ── Position management ──
    def check_positions(self, symbol, current_price, closes_1m=None, highs_1m=None, lows_1m=None):
        if symbol not in self.state["active_positions"]:
            return
        pos    = self.state["active_positions"][symbol]
        side   = pos["side"]
        entry  = float(pos["entry"])
        tp     = float(pos["tp"])
        sl     = float(pos["sl"])
        margin = float(pos.get("margin", MARGIN_PER_TRADE))
        qty    = (margin * LEVERAGE) / entry
        price  = float(current_price)
        if closes_1m is not None and len(closes_1m) >= TRAIL_LOOKBACK:
            if side == "buy":
                recent_lows = lows_1m[-TRAIL_LOOKBACK:]
                new_sl = float(np.min(recent_lows)) - atr_calc(highs_1m, lows_1m, closes_1m, 14) * 0.2
                if new_sl > sl:
                    sl = new_sl
                    self.state["active_positions"][symbol]["sl"] = round(sl, 6)
            else:
                recent_highs = highs_1m[-TRAIL_LOOKBACK:]
                new_sl = float(np.max(recent_highs)) + atr_calc(highs_1m, lows_1m, closes_1m, 14) * 0.2
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
                               "ENGULFING", "PIN BAR", "INSIDE BAR", "OB", "FVG", "LIQ SWEEP"]:
                if any(strat_key in str(r) for r in pos_reasons):
                    strategy_name = strat_key
                    break
            self.state["trades"].append({
                "time":     exit_time_str,
                "symbol":   symbol,
                "side":     side.upper(),
                "entry":    round(entry, 6),
                "exit":     round(exit_p, 6),
                "pnl":      pnl,
                "result":   reason,
                "score":    pos.get("score", 0),
                "margin":   margin,
                "strategy": strategy_name,
                "reasons":  pos_reasons,
            })
            del self.state["active_positions"][symbol]
            icon = "PROFIT" if pnl > 0 else "LOSS"
            log.info(f"{icon} {symbol} {side.upper()} | Entry:${entry} Exit:${exit_p} PnL:${pnl} | {reason}")
            self.save_history()

    # ── Strategy 10 ──
    def strategy10_ema_crossover(self, symbol, opens, highs, lows, closes, vols, timeframe="5m"):
        try:
            if len(closes) < EMA10_SLOW + 2:
                return None, 0, []
            e20_now  = ema_calc(closes, EMA10_FAST)[-1]
            e30_now  = ema_calc(closes, EMA10_MID)[-1]
            e50_now  = ema_calc(closes, EMA10_SLOW)[-1]
            e20_prev = ema_calc(closes[:-1], EMA10_FAST)[-1]
            e30_prev = ema_calc(closes[:-1], EMA10_MID)[-1]
            e50_prev = ema_calc(closes[:-1], EMA10_SLOW)[-1]
            bull_now  = e20_now > e30_now > e50_now
            bull_prev = e20_prev > e30_prev > e50_prev
            bear_now  = e20_now < e30_now < e50_now
            bear_prev = e20_prev < e30_prev < e50_prev
            fresh_bull = bull_now and not bull_prev
            fresh_bear = bear_now and not bear_prev
            if not fresh_bull and not fresh_bear:
                return None, 0, []
            body_now = float(closes[-1]) - float(opens[-1])
            green_candle = body_now > 0
            red_candle   = body_now < 0
            vol_now  = float(vols[-1])
            vol_prev = float(vols[-2]) if len(vols) >= 2 else vol_now
            vol_spike = vol_now > vol_prev * EMA10_VOL_MULT
            if fresh_bull and green_candle and vol_spike:
                reasons = [f"S10 FRESH BULL CROSS [{timeframe}]", f"EMA20>{round(e20_now,4)} EMA30>{round(e30_now,4)} EMA50>{round(e50_now,4)}", f"VOL {round(vol_now/vol_prev,1)}x SPIKE", "GREEN CANDLE"]
                return "buy", 8, reasons
            if fresh_bear and red_candle and vol_spike:
                reasons = [f"S10 FRESH BEAR CROSS [{timeframe}]", f"EMA20<{round(e20_now,4)} EMA30<{round(e30_now,4)} EMA50<{round(e50_now,4)}", f"VOL {round(vol_now/vol_prev,1)}x SPIKE", "RED CANDLE"]
                return "sell", 8, reasons
        except Exception as ex:
            log.warning(f"Strategy 10 error {symbol} [{timeframe}]: {ex}")
        return None, 0, []

    def run_strategy10(self, symbol):
        s10_key = symbol + "_S10"
        if s10_key in self.state["active_positions"]:
            pos   = self.state["active_positions"][s10_key]
            price = self.state["last_prices"].get(symbol, 0)
            if price:
                side  = pos["side"]
                entry = float(pos["entry"])
                tp    = float(pos["tp"])
                sl    = float(pos["sl"])
                margin= float(pos.get("margin", MARGIN_PER_TRADE))
                qty   = (margin * LEVERAGE) / entry
                hit, reason, exit_p = False, "", price
                if side == "buy":
                    if price >= tp: hit, reason, exit_p = True, "TP", tp
                    elif price <= sl: hit, reason, exit_p = True, "SL", sl
                else:
                    if price <= tp: hit, reason, exit_p = True, "TP", tp
                    elif price >= sl: hit, reason, exit_p = True, "SL", sl
                if hit:
                    pnl = round((exit_p-entry)*qty if side=="buy" else (entry-exit_p)*qty, 4)
                    st  = self.state["stats"]
                    self.state["total_pnl"] = round(self.state.get("total_pnl",0)+pnl, 4)
                    st["total_trades"] += 1
                    if pnl>0: st["wins"]+=1; st["best_trade"]=round(max(st.get("best_trade",0),pnl),4)
                    else:     st["losses"]+=1; st["worst_trade"]=round(min(st.get("worst_trade",0),pnl),4)
                    today = ist_now().strftime("%Y-%m-%d")
                    st["daily_pnl"][today]    = round(st["daily_pnl"].get(today,0)+pnl, 4)
                    st["daily_trades"][today] = st["daily_trades"].get(today,0)+1
                    self.state["trades"].append({
                        "time": ist_short(), "symbol": s10_key,
                        "side": side.upper(), "entry": round(entry,6), "exit": round(exit_p,6),
                        "pnl": pnl, "result": reason, "score": pos.get("score",8),
                        "margin": margin, "strategy": "EMA CROSSOVER",
                        "reasons": pos.get("reasons",[]),
                    })
                    del self.state["active_positions"][s10_key]
                    icon = "PROFIT" if pnl>0 else "LOSS"
                    log.info(f"S10 {icon}: {s10_key} {side.upper()} | PnL:${pnl} | {reason}")
                    self.save_history()
            return
        for tf in EMA10_TIMEFRAMES:
            try:
                limit = 120 if tf == "1m" else 80 if tf == "5m" else 60
                result = self.fetch_okx_public(symbol, tf, limit)
                if result is None:
                    continue
                opens, highs, lows, closes, vols = result
                direction, score, reasons = self.strategy10_ema_crossover(
                    symbol, opens, highs, lows, closes, vols, tf
                )
                if direction and score >= 6 and self.can_trade(score):
                    price = float(closes[-1])
                    av    = atr_calc(highs, lows, closes, ATR_PERIOD)
                    if direction == "buy":
                        tp = round(price + av * TP_MULT, 6)
                        swing_low = float(np.min(lows[-10:]))
                        sl = round(min(swing_low - av * 0.1, price - av * SL_MULT), 6)
                    else:
                        tp = round(price - av * TP_MULT, 6)
                        swing_high = float(np.max(highs[-10:]))
                        sl = round(max(swing_high + av * 0.1, price + av * SL_MULT), 6)
                    self.state["active_positions"][s10_key] = {
                        "side": direction, "entry": price,
                        "tp": tp, "sl": sl,
                        "margin": MARGIN_PER_TRADE,
                        "score": score, "reasons": reasons,
                        "time": ist_short(),
                        "strategy10": True, "timeframe": tf,
                    }
                    self.state["last_prices"][symbol] = price
                    self.save_history()
                    log.info(f"S10 NEW TRADE: {s10_key} {direction.upper()} [{tf}] | Score:{score} | TP:{tp} | SL:{sl}")
                    break
            except Exception as ex:
                log.warning(f"S10 run error {symbol} [{tf}]: {ex}")

    def check_hedge_positions(self):
        hedge_keys = [k for k in list(self.state["active_positions"].keys()) if k.endswith(HEDGE_SUFFIX)]
        for hedge_key in hedge_keys:
            symbol = hedge_key.replace(HEDGE_SUFFIX, "")
            price = self.state["last_prices"].get(symbol, 0)
            if not price:
                continue
            pos    = self.state["active_positions"][hedge_key]
            side   = pos["side"]
            entry  = float(pos["entry"])
            tp     = float(pos["tp"])
            sl     = float(pos["sl"])
            margin = float(pos.get("margin", MARGIN_BLAST))
            qty    = (margin * LEVERAGE) / entry
            hit, reason, exit_p = False, "", price
            if side == "buy":
                if price >= tp:  hit, reason, exit_p = True, "TP", tp
                elif price <= sl: hit, reason, exit_p = True, "SL", sl
            else:
                if price <= tp:  hit, reason, exit_p = True, "TP", tp
                elif price >= sl: hit, reason, exit_p = True, "SL", sl
            if hit:
                pnl = round((exit_p - entry) * qty if side == "buy" else (entry - exit_p) * qty, 4)
                st  = self.state["stats"]
                self.state["total_pnl"] = round(self.state.get("total_pnl", 0) + pnl, 4)
                st["total_trades"] += 1
                if pnl > 0: st["wins"] += 1; st["best_trade"] = round(max(st.get("best_trade",0), pnl), 4)
                else: st["losses"] += 1; st["worst_trade"] = round(min(st.get("worst_trade",0), pnl), 4)
                today = ist_now().strftime("%Y-%m-%d")
                st["daily_pnl"][today] = round(st["daily_pnl"].get(today, 0) + pnl, 4)
                st["daily_trades"][today] = st["daily_trades"].get(today, 0) + 1
                self.state["trades"].append({
                    "time": ist_short(), "symbol": hedge_key,
                    "side": side.upper(), "entry": round(entry,6), "exit": round(exit_p,6),
                    "pnl": pnl, "result": reason, "score": pos.get("score",0), "margin": margin,
                })
                del self.state["active_positions"][hedge_key]
                icon = "PROFIT" if pnl > 0 else "LOSS"
                log.info(f"HEDGE {icon} {hedge_key} {side.upper()} | Entry:${entry} Exit:${exit_p} PnL:${pnl} | {reason}")
                self.save_history()

    # ── Main Run – with loop ──
    def run(self):
        log.info(f"WhaleTrader Pro — Auto-run every {LOOP_INTERVAL//60} minutes (Press Ctrl+C to stop)")
        while True:
            try:
                self.dashboard = []
                self.cleanup_stale_positions()
                self.check_hedge_positions()

                # Strategy 10
                for sym10 in SYMBOLS:
                    try:
                        self.run_strategy10(sym10)
                    except Exception as ex:
                        log.warning(f"Strategy 10 outer error {sym10}: {ex}")

                vol_map = self.prefetch_all_volumes()
                log.info(f"Volume prefetch: {len(vol_map)} coins loaded")

                # Parallel fetch
                symbol_data = {}
                with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as executor:
                    future_to_symbol = {executor.submit(self.fetch_symbol_data, sym): sym for sym in SYMBOLS}
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

                # Process each symbol
                for symbol in SYMBOLS:
                    try:
                        if symbol not in symbol_data:
                            self.dashboard.append({"symbol": symbol, "price": self.state["last_prices"].get(symbol, 0),
                                                   "signal": "SCANNING", "score": 0,
                                                   "entry": None, "tp": None, "sl": None,
                                                   "reasons": ["Data unavailable"]})
                            continue
                        opens1, highs1, lows1, closes1, vols1, opens5, highs5, lows5, closes5, vols5, closes15 = symbol_data[symbol]

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
                                                   "score":  pos.get("score", 0),
                                                   "entry":  pos["entry"],
                                                   "tp":     pos["tp"],
                                                   "sl":     pos["sl"],
                                                   "reasons": pos.get("reasons", [])})
                            continue

                        score, direction, r, upper, lower, reasons, adaptive_min = self.score_signal(
                            opens1, highs1, lows1, closes1, vols1,
                            opens5, highs5, lows5, closes5, vols5, closes15)

                        log.info(f"{symbol} | ${price} | Score:{score}/10 | Regime:{reasons[0] if reasons else '?'} | Min:{adaptive_min}")

                        # Darvas
                        darvas_sig, box_top, box_bot, darvas_sl, darvas_reasons = darvas_box_strategy(
                            opens5, highs5, lows5, closes5, vols5)
                        if darvas_sig and darvas_sig == direction:
                            score = min(score + 2, 10)
                            reasons = reasons + darvas_reasons
                            log.info(f"DARVAS CONFIRM {symbol} {darvas_sig.upper()} | Box: {box_bot}-{box_top}")
                        elif darvas_sig and not direction and darvas_sl:
                            direction = darvas_sig
                            score     = max(score, 5)
                            reasons   = darvas_reasons
                            log.info(f"DARVAS SIGNAL {symbol} {darvas_sig.upper()} | Box: {box_bot}-{box_top}")

                        if direction and score >= adaptive_min and self.can_trade(score):
                            is_blast = getattr(self, "_last_is_blast", False)
                            margin = MARGIN_BLAST if is_blast else MARGIN_PER_TRADE
                            av = atr_calc(highs5, lows5, closes5, ATR_PERIOD)

                            if direction == "buy":
                                tp = round(price + av * TP_MULT, 6)
                                swing_low = float(np.min(lows1[-10:]))
                                sl = round(min(swing_low - av * 0.5, price - av * SL_MULT), 6)
                            else:
                                tp = round(price - av * TP_MULT, 6)
                                swing_high = float(np.max(highs1[-10:]))
                                sl = round(max(swing_high + av * 0.5, price + av * SL_MULT), 6)

                            self.state["active_positions"][symbol] = {
                                "side":    direction,
                                "entry":   price,
                                "tp":      tp,
                                "sl":      sl,
                                "margin":  margin,
                                "score":   score,
                                "reasons": reasons,
                                "time":    ist_short(),
                            }
                            self.save_history()
                            log.info(f"NEW TRADE: {symbol} {direction.upper()} | Score:{score} | Margin:${margin} | TP:{tp} | SL:{sl}")

                            # Hedge
                            hedge_key = symbol + HEDGE_SUFFIX
                            hedge_direction = "sell" if direction == "buy" else "buy"
                            hedge_already_open = hedge_key in self.state["active_positions"]
                            if (HEDGE_ENABLED and is_blast and score >= 6 
                                    and not hedge_already_open
                                    and len(self.state["active_positions"]) < MAX_POSITIONS):
                                hedge_margin = MARGIN_BLAST
                                if hedge_direction == "buy":
                                    hedge_tp = round(price + av * TP_MULT, 6)
                                    swing_low_h = float(np.min(lows1[-10:]))
                                    hedge_sl = round(min(swing_low_h - av * 0.5, price - av * SL_MULT), 6)
                                else:
                                    hedge_tp = round(price - av * TP_MULT, 6)
                                    swing_high_h = float(np.max(highs1[-10:]))
                                    hedge_sl = round(max(swing_high_h + av * 0.5, price + av * SL_MULT), 6)
                                self.state["active_positions"][hedge_key] = {
                                    "side":    hedge_direction,
                                    "entry":   price,
                                    "tp":      hedge_tp,
                                    "sl":      hedge_sl,
                                    "margin":  hedge_margin,
                                    "score":   score,
                                    "reasons": ["HEDGE " + r for r in reasons],
                                    "time":    ist_short(),
                                    "is_hedge": True,
                                }
                                self.save_history()
                                log.info(f"HEDGE TRADE: {hedge_key} {hedge_direction.upper()} | Margin:${hedge_margin} | TP:{hedge_tp} | SL:{hedge_sl}")

                        # Dashboard entries
                        s10_key = symbol + "_S10"
                        if s10_key in self.state["active_positions"]:
                            sp = self.state["active_positions"][s10_key]
                            self.dashboard.append({"symbol": s10_key, "price":  price,
                                                   "signal": "S10 " + sp["side"].upper(),
                                                   "score":  sp.get("score", 8),
                                                   "entry":  sp["entry"],
                                                   "tp":     sp["tp"],
                                                   "sl":     sp["sl"],
                                                   "margin": sp.get("margin", MARGIN_PER_TRADE),
                                                   "reasons": sp.get("reasons", [])})
                        hedge_key = symbol + HEDGE_SUFFIX
                        if hedge_key in self.state["active_positions"]:
                            hpos = self.state["active_positions"][hedge_key]
                            self.dashboard.append({"symbol": hedge_key, "price":  price,
                                                   "signal": "HEDGE " + hpos["side"].upper(),
                                                   "score":  hpos.get("score", 0),
                                                   "entry":  hpos["entry"],
                                                   "tp":     hpos["tp"],
                                                   "sl":     hpos["sl"],
                                                   "margin": hpos.get("margin", MARGIN_BLAST),
                                                   "reasons": hpos.get("reasons", [])})

                        if symbol in self.state["active_positions"]:
                            pos = self.state["active_positions"][symbol]
                            self.dashboard.append({"symbol": symbol, "price": price,
                                                   "signal": "HOLDING " + pos["side"].upper(),
                                                   "score":  pos.get("score", score),
                                                   "entry":  pos["entry"],
                                                   "tp":     pos["tp"],
                                                   "sl":     pos["sl"],
                                                   "margin": pos.get("margin", MARGIN_PER_TRADE),
                                                   "reasons": pos.get("reasons", reasons)})
                        else:
                            self.dashboard.append({"symbol":  symbol,
                                                   "price":   price,
                                                   "signal":  direction.upper() + " " + str(score) + "/10" if direction else "SCANNING",
                                                   "score":   score,
                                                   "entry":   None,
                                                   "tp":      None,
                                                   "sl":      None,
                                                   "reasons": reasons})
                    except Exception as e:
                        log.error(f"Error {symbol}: {e}")
                        self.dashboard.append({"symbol": symbol, "price": 0, "signal": "ERROR",
                                               "score": 0, "entry": 0, "tp": None, "sl": None, "reasons": [str(e)]})

                self.save_history()
                self.build_dashboard()
                log.info(f"Cycle complete. Next run in {LOOP_INTERVAL//60} minutes...")
                time.sleep(LOOP_INTERVAL)
            except KeyboardInterrupt:
                log.info("Bot stopped by user (Ctrl+C). Exiting...")
                break
            except Exception as e:
                log.error(f"Unexpected error in loop: {e}")
                log.info("Waiting 60 seconds before retry...")
                time.sleep(60)

    # ── Dashboard (full version – same as earlier) ──
    def build_dashboard(self):
        state = self.state
        stats = state.get("stats", {})
        trades = state.get("trades", [])

        total_pnl = round(state.get("total_pnl", 0.0), 2)
        wallet = round(INITIAL_CAPITAL + total_pnl, 2)
        total_t = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        win_rate = round(wins / total_t * 100, 1) if total_t > 0 else 0.0
        best = round(stats.get("best_trade", 0.0), 2)
        worst = round(stats.get("worst_trade", 0.0), 2)
        pnl_pct = round((total_pnl / INITIAL_CAPITAL) * 100, 2)
        now_str = ist_str()

        today = ist_now().strftime("%Y-%m-%d")
        today_pnl = round(stats.get("daily_pnl", {}).get(today, 0.0), 2)
        today_trades = stats.get("daily_trades", {}).get(today, 0)

        open_pos = len(state.get("active_positions", {}))
        trading_ok = True

        daily_pnl_data = stats.get("daily_pnl", {})
        sorted_days = sorted(daily_pnl_data.items())[-14:]
        chart_labels = json.dumps([d[0][5:] for d in sorted_days])
        chart_values = json.dumps([round(d[1], 2) for d in sorted_days])
        chart_colors = json.dumps([
            "rgba(0,230,118,0.85)" if d[1] >= 0 else "rgba(255,23,68,0.85)"
            for d in sorted_days
        ])

        monitor_rows = ""
        for d in self.dashboard:
            sym = d.get("symbol", "")
            clean = sym.replace("/", "").lower()
            sig = d.get("signal", "SCANNING")
            score = int(d.get("score", 0))
            reasons = ", ".join(d.get("reasons", [])) or "Waiting..."
            tp_val = "$" + str(d["tp"]) if d.get("tp") else "-"
            sl_val = "$" + str(d["sl"]) if d.get("sl") else "-"
            entry_val = "$" + str(d["entry"]) if d.get("entry") and "HOLD" in sig else "-"

            if "HOLD" in sig:
                pill = "<span class='pill-hold'>" + sig + "</span>"
            elif "BUY" in sig:
                pill = "<span class='pill-buy'>" + sig + "</span>"
            elif "SELL" in sig:
                pill = "<span class='pill-sell'>" + sig + "</span>"
            else:
                pill = "<span class='pill-scan'>SCANNING</span>"

            dots = ""
            for i in range(10):
                dots += "<span class='dotf'></span>" if i < score else "<span class='dot'></span>"

            okx_sym = sym.replace("/", "-").lower() + "-swap"
            okx_url = "https://www.okx.com/trade-swap/" + okx_sym
            monitor_rows += (
                "<div class='monitor-card'>"
                "<div class='monitor-top'>"
                "<div class='monitor-sym'>" + sym + "</div>" "<div style='font-size:9px;color:var(--muted);margin-top:1px;'>OKX ↗</div>"
                "<span id='p-" + clean + "' class='monitor-price'>$" + str(d.get('price', '--')) + "</span>"
                "</div>"
                "<div class='monitor-row'>"
                "<div><span class='monitor-label'>24h </span><span id='c-" + clean + "' class='monitor-val'>--</span></div>"
                "<div>" + pill + "</div>"
                "<div><span class='monitor-label'>Score </span><span class='monitor-val' style='color:var(--green);'>" + str(score) + "/10</span></div>"
                "</div>"
                "<div class='monitor-bottom'>"
                + (
                    "<div><div class='monitor-label'>Entry</div><div class='monitor-val' style='color:var(--amber);'>" + entry_val + "</div></div>"
                    "<div><div class='monitor-label'>Take Profit</div><div class='monitor-val' style='color:var(--green);'>" + tp_val + "</div></div>"
                    "<div><div class='monitor-label'>Stop Loss</div><div class='monitor-val' style='color:var(--red);'>" + sl_val + "</div></div>"
                    "<div><div class='monitor-label'>Margin</div><div class='monitor-val' style='color:var(--amber);font-size:11px;font-weight:800;'>" + ("$" + str(d.get("margin", MARGIN_PER_TRADE)) if "HOLD" in sig else "-") + "</div></div>"
                    if "HOLD" in sig else
                    "<div style='color:var(--muted);font-size:10px;'>⚡ " + reasons + "</div>"
                ) +
                "<div class='dots'>" + dots + "</div>"
                "</div>"
                "<div style='margin-top:10px;padding-top:8px;border-top:1px solid #162035;text-align:center;'>"
                "<a href='" + okx_url + "' target='_blank' style='display:inline-block;background:rgba(0,230,118,.12);color:#00e676;border:1px solid rgba(0,230,118,.3);font-size:11px;font-weight:700;padding:7px 20px;border-radius:8px;text-decoration:none;'>&#x1F4CA; View OKX Chart</a>"
                "</div>"
                "</div>"
            )

        history_rows = ""
        for t in reversed(trades[-50:]):
            pv = float(t.get("pnl", 0))
            pc = "pos" if pv >= 0 else "neg"
            icon = "TP" if t.get("result", "") == "TP" else "SL"
            badge = "pill-buy" if t.get("side", "") == "BUY" else "pill-sell"
            prefix = "+" if pv >= 0 else ""
            history_rows += (
                "<div class='trade-card'>"
                "<div class='trade-top'>"
                "<div style='display:flex;align-items:center;gap:8px;'>"
                "<div class='trade-pair'>" + str(t.get("symbol", "")) + "</div>"
                "<span class='" + badge + "'>" + str(t.get("side", "")) + "</span>"
                "</div>"
                "<div class='trade-pnl " + pc + "'>" + prefix + "$" + str(round(pv, 2)) + "</div>"
                "</div>"
                "<div class='trade-row'>"
                "<div class='trade-item'><div class='trade-item-label'>Time</div><div class='trade-item-val'>" + str(t.get("time", "")) + "</div></div>"
                "<div class='trade-item'><div class='trade-item-label'>Entry</div><div class='trade-item-val'>$" + str(t.get("entry", "")) + "</div></div>"
                "<div class='trade-item'><div class='trade-item-label'>Exit</div><div class='trade-item-val'>$" + str(t.get("exit", "")) + "</div></div>"
                "</div>"
                "<div class='trade-meta'>"
                "<span class='trade-result'>" + ("🎯 TP" if icon == "TP" else "🛑 SL") + "</span>"
                "<span class='trade-score'>Score: " + str(t.get("score", "-")) + "/10</span>"
                "<span class='trade-margin' style='font-size:10px;color:var(--amber);font-family:var(--mono);margin-left:auto;font-weight:700;background:linear-gradient(135deg,rgba(255,165,2,0.15),rgba(255,109,0,0.15));padding:2px 8px;border-radius:6px;border:1px solid rgba(255,165,2,0.3);'>Margin: $" + str(t.get("margin", 100)) + "</span>"
                "</div>"
                "</div>"
            )
        if not history_rows:
            history_rows = "<div class='trade-card' style='text-align:center;color:var(--muted);'>No trades yet — bot is scanning...</div>"

        risk_badge_text = "TRADING ACTIVE" if trading_ok else "LIMIT REACHED"
        risk_badge_cls = "badge-ok" if trading_ok else "badge-limit"

        css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#0b0e11;
  --bg1:#13161b;
  --bg2:#1a1d24;
  --bg3:#22262f;
  --line:#2b2f3a;
  --text:#eaecef;
  --muted:#848e9c;
  --muted2:#5a6171;
  --green:#0ecb81;
  --red:#f6465d;
  --yellow:#f0b90b;
  --blue:#1890ff;
  --purple:#8b5cf6;
  --white:#ffffff;
  --font:'Inter',sans-serif;
  --mono:'JetBrains Mono',monospace;
}
html{font-size:13px;scroll-behavior:smooth;}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh;-webkit-font-smoothing:antialiased;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:var(--line);border-radius:4px;}
nav{background:var(--bg1);border-bottom:1px solid var(--line);height:52px;display:flex;align-items:center;justify-content:space-between;padding:0 16px;position:sticky;top:0;z-index:100;}
.nav-logo{display:flex;align-items:center;gap:8px;}
.nav-logo-icon{width:28px;height:28px;border-radius:6px;background:linear-gradient(135deg,#f0b90b,#f8d12f);display:flex;align-items:center;justify-content:center;font-size:14px;}
.nav-logo-text{font-size:14px;font-weight:700;color:var(--white);letter-spacing:-.2px;}
.nav-right{display:flex;align-items:center;gap:8px;}
.badge-ok{font-size:10px;font-weight:600;color:var(--green);background:rgba(14,203,129,.1);border:1px solid rgba(14,203,129,.25);padding:3px 10px;border-radius:4px;}
.badge-limit{font-size:10px;font-weight:600;color:var(--red);background:rgba(246,70,93,.1);border:1px solid rgba(246,70,93,.25);padding:3px 10px;border-radius:4px;}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:blink 2s infinite;display:inline-block;margin-right:4px;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.3;}}
.sync-text{font-family:var(--mono);font-size:9px;color:var(--muted2);}
.tabs{display:flex;background:var(--bg1);border-bottom:1px solid var(--line);overflow-x:auto;-webkit-overflow-scrolling:touch;position:sticky;top:52px;z-index:99;}
.tabs::-webkit-scrollbar{display:none;}
.tab{flex:none;padding:12px 16px;font-size:12px;font-weight:600;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:all .2s;}
.tab.active{color:var(--yellow);border-bottom-color:var(--yellow);}
.panel{display:none;padding:12px;}
.panel.active{display:block;}
.overview-header{background:linear-gradient(135deg,#1a1d24,#22262f);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:10px;}
.overview-equity{font-size:28px;font-weight:800;color:var(--white);font-family:var(--mono);letter-spacing:-.5px;}
.overview-pnl{font-size:13px;font-weight:600;margin-top:4px;}
.overview-pnl.pos{color:var(--green);}
.overview-pnl.neg{color:var(--red);}
.stats-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;}
.stat-box{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:12px;}
.stat-box-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:6px;}
.stat-box-val{font-size:18px;font-weight:800;font-family:var(--mono);color:var(--white);}
.stat-box-sub{font-size:10px;color:var(--muted);margin-top:3px;}
.stat-box-val.green{color:var(--green);}
.stat-box-val.red{color:var(--red);}
.stat-box-val.yellow{color:var(--yellow);}
.chart-box{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:8px;}
.chart-box-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px;}
.market-card{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:8px;}
.market-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;}
.market-pair{font-size:14px;font-weight:700;color:var(--white);}
.market-price{font-family:var(--mono);font-size:15px;font-weight:700;color:var(--white);}
.market-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;}
.market-chg.up{color:var(--green);font-size:11px;font-weight:600;font-family:var(--mono);}
.market-chg.dn{color:var(--red);font-size:11px;font-weight:600;font-family:var(--mono);}
.signal-pill{font-size:10px;font-weight:700;padding:3px 10px;border-radius:4px;text-transform:uppercase;letter-spacing:.04em;}
.sp-buy{background:rgba(14,203,129,.12);color:var(--green);border:1px solid rgba(14,203,129,.3);}
.sp-sell{background:rgba(246,70,93,.12);color:var(--red);border:1px solid rgba(246,70,93,.3);}
.sp-hold{background:rgba(24,144,255,.12);color:var(--blue);border:1px solid rgba(24,144,255,.3);}
.sp-scan{background:rgba(240,185,11,.1);color:var(--yellow);border:1px solid rgba(240,185,11,.25);}
.score-dots{display:flex;gap:3px;}
.d-on{width:5px;height:5px;border-radius:50%;background:var(--green);box-shadow:0 0 4px rgba(14,203,129,.5);}
.d-off{width:5px;height:5px;border-radius:50%;background:var(--line);}
.market-pos{background:var(--bg3);border-radius:6px;padding:10px;margin-top:8px;}
.pos-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:6px;}
.pos-item-label{font-size:9px;color:var(--muted);text-transform:uppercase;font-weight:600;margin-bottom:2px;}
.pos-item-val{font-family:var(--mono);font-size:11px;font-weight:700;}
.pos-item-val.entry{color:var(--yellow);}
.pos-item-val.tp{color:var(--green);}
.pos-item-val.sl{color:var(--red);}
.pos-margin-badge{display:inline-flex;align-items:center;font-size:10px;font-weight:700;color:var(--yellow);background:rgba(240,185,11,.1);border:1px solid rgba(240,185,11,.25);padding:2px 8px;border-radius:4px;margin-top:6px;}
.market-ind{font-size:10px;color:var(--muted);margin-top:8px;border-top:1px solid var(--line);padding-top:8px;}
.pos-card{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:13px;margin-bottom:8px;position:relative;overflow:hidden;}
.pos-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;}
.pos-card.buy::before{background:var(--green);}
.pos-card.sell::before{background:var(--red);}
.pos-card-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.pos-pair{font-size:14px;font-weight:700;color:var(--white);}
.pos-side{font-size:10px;font-weight:700;padding:3px 8px;border-radius:4px;}
.side-buy{background:rgba(14,203,129,.12);color:var(--green);border:1px solid rgba(14,203,129,.3);}
.side-sell{background:rgba(246,70,93,.12);color:var(--red);border:1px solid rgba(246,70,93,.3);}
.pos-live-price{font-family:var(--mono);font-size:14px;font-weight:700;}
.pos-details{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;background:var(--bg3);border-radius:6px;padding:10px;margin-bottom:8px;}
.pos-d-label{font-size:9px;color:var(--muted);text-transform:uppercase;font-weight:600;margin-bottom:3px;letter-spacing:.06em;}
.pos-d-val{font-family:var(--mono);font-size:12px;font-weight:700;}
.pos-footer{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.pos-margin-tag{font-size:10px;font-weight:700;color:var(--yellow);background:rgba(240,185,11,.1);border:1px solid rgba(240,185,11,.25);padding:3px 10px;border-radius:4px;font-family:var(--mono);}
.pos-score-tag{font-size:10px;color:var(--muted);background:var(--bg3);padding:3px 8px;border-radius:4px;font-family:var(--mono);}
.pos-reasons{font-size:10px;color:var(--muted);margin-left:auto;}
.empty-state{text-align:center;padding:40px 20px;color:var(--muted2);}
.empty-icon{font-size:32px;margin-bottom:8px;}
.empty-text{font-size:12px;font-weight:500;}
.log-card{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:8px;position:relative;overflow:hidden;}
.log-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;}
.log-card.win::before{background:var(--green);}
.log-card.loss::before{background:var(--red);}
.log-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;}
.log-pair-row{display:flex;align-items:center;gap:6px;}
.log-pair{font-size:13px;font-weight:700;color:var(--white);}
.log-side{font-size:9px;font-weight:700;padding:2px 7px;border-radius:4px;}
.log-pnl{font-family:var(--mono);font-size:16px;font-weight:800;}
.log-pnl.win{color:var(--green);}
.log-pnl.loss{color:var(--red);}
.log-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;background:var(--bg3);border-radius:6px;overflow:hidden;margin-bottom:8px;}
.log-cell{padding:7px 8px;border-right:1px solid var(--line);}
.log-cell:last-child{border-right:none;}
.log-cell-label{font-size:9px;color:var(--muted);text-transform:uppercase;font-weight:600;margin-bottom:2px;letter-spacing:.06em;}
.log-cell-val{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--text);}
.log-footer{display:flex;align-items:center;gap:6px;flex-wrap:wrap;}
.log-result{font-size:11px;font-weight:700;}
.log-result.win{color:var(--green);}
.log-result.loss{color:var(--red);}
.log-score{font-size:10px;color:var(--muted);background:var(--bg3);padding:2px 6px;border-radius:4px;font-family:var(--mono);}
.log-margin{font-size:10px;font-weight:700;color:var(--yellow);background:rgba(240,185,11,.1);border:1px solid rgba(240,185,11,.25);padding:2px 8px;border-radius:4px;font-family:var(--mono);margin-left:auto;}
.log-time{font-size:10px;color:var(--muted2);font-family:var(--mono);}
footer{text-align:center;color:var(--muted2);font-size:10px;padding:24px 12px;border-top:1px solid var(--line);margin-top:12px;}
.strat-card{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:13px;margin-bottom:8px;}
.strat-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;}
.strat-name{font-size:13px;font-weight:700;color:var(--white);}
.strat-pnl{font-family:var(--mono);font-size:13px;font-weight:800;}
.strat-pnl.win{color:var(--green);}
.strat-pnl.loss{color:var(--red);}
.strat-badge{font-size:10px;padding:2px 8px;border-radius:4px;font-weight:600;}
.strat-badge.neutral{background:rgba(139,146,217,.1);color:var(--muted);border:1px solid rgba(139,146,217,.2);}
.strat-bar-bg{background:var(--bg3);border-radius:4px;height:6px;margin-bottom:8px;overflow:hidden;}
.strat-bar{height:6px;border-radius:4px;transition:width .5s;}
.strat-bar.good{background:var(--green);}
.strat-bar.warn{background:var(--yellow);}
.strat-bar.bad{background:var(--red);}
.strat-stats{display:flex;align-items:center;justify-content:space-between;}
.strat-wr{font-size:12px;font-weight:800;font-family:var(--mono);}
.strat-wr.good{color:var(--green);}
.strat-wr.warn{color:var(--yellow);}
.strat-wr.bad{color:var(--red);}
.strat-detail{font-size:10px;color:var(--muted);}
.strat-sub{font-size:10px;color:var(--muted2);}
"""

        STRATEGY_NAMES = [
            "BB BLAST", "BB BOUNCE", "DOUBLE BB", "SR BREAKOUT",
            "VWAP", "EMA RIBBON", "ORDER FLOW", "DARVAS", "HEDGE",
            "EMA CROSSOVER", "ENGULFING", "PIN BAR", "INSIDE BAR", "OB", "FVG", "LIQ SWEEP"
        ]
        strat_stats = {s: {"wins": 0, "losses": 0, "total": 0, "pnl": 0.0} for s in STRATEGY_NAMES}
        for t in trades:
            strat = t.get("strategy", "UNKNOWN")
            pnl_t = float(t.get("pnl", 0))
            matched = "UNKNOWN"
            for s in STRATEGY_NAMES:
                if s in str(strat) or s in str(t.get("reasons", [])):
                    matched = s
                    break
            if matched in strat_stats:
                strat_stats[matched]["total"] += 1
                strat_stats[matched]["pnl"] += pnl_t
                if pnl_t > 0:
                    strat_stats[matched]["wins"] += 1
                else:
                    strat_stats[matched]["losses"] += 1

        strat_html = ""
        for sname, ss in strat_stats.items():
            if ss["total"] == 0:
                strat_html += (
                    "<div class='strat-card'>"
                    "<div class='strat-top'>"
                    "<span class='strat-name'>" + sname + "</span>"
                    "<span class='strat-badge neutral'>No trades yet</span>"
                    "</div>"
                    "<div class='strat-sub'>Waiting for first signal...</div>"
                    "</div>"
                )
                continue
            wr = round(ss["wins"] / ss["total"] * 100, 1)
            total_pnl_s = round(ss["pnl"], 2)
            wr_cls = "good" if wr >= 55 else "warn" if wr >= 45 else "bad"
            pnl_cls = "win" if total_pnl_s >= 0 else "loss"
            bar_w = str(int(wr)) + "%"
            strat_html += (
                "<div class='strat-card'>"
                "<div class='strat-top'>"
                "<span class='strat-name'>" + sname + "</span>"
                "<span class='strat-pnl " + pnl_cls + "'>" + ("+" if total_pnl_s >= 0 else "") + "$" + str(total_pnl_s) + "</span>"
                "</div>"
                "<div class='strat-bar-bg'><div class='strat-bar " + wr_cls + "' style='width:" + bar_w + "'></div></div>"
                "<div class='strat-stats'>"
                "<span class='strat-wr " + wr_cls + "'>" + str(wr) + "% WR</span>"
                "<span class='strat-detail'>" + str(ss["wins"]) + "W / " + str(ss["losses"]) + "L / " + str(ss["total"]) + " trades</span>"
                "</div>"
                "</div>"
            )

        monitor_html = ""
        for d in self.dashboard:
            sym = d.get("symbol", "")
            clean = sym.replace("/", "").lower()
            sig = d.get("signal", "SCANNING")
            score = int(d.get("score", 0))
            reasons_str = ", ".join(d.get("reasons", [])) or "Waiting..."
            tp_val = "$" + str(d["tp"]) if d.get("tp") else "-"
            sl_val = "$" + str(d["sl"]) if d.get("sl") else "-"
            entry_val = "$" + str(d["entry"]) if d.get("entry") and "HOLD" in sig else "-"
            margin_val = str(d.get("margin", MARGIN_PER_TRADE))

            if "HOLD" in sig:
                if "BUY" in sig:
                    pill = "<span class='signal-pill sp-hold'>HOLDING BUY</span>"
                else:
                    pill = "<span class='signal-pill sp-hold'>HOLDING SELL</span>"
            elif "BUY" in sig:
                pill = "<span class='signal-pill sp-buy'>" + sig + "</span>"
            elif "SELL" in sig:
                pill = "<span class='signal-pill sp-sell'>" + sig + "</span>"
            else:
                pill = "<span class='signal-pill sp-scan'>SCANNING</span>"

            dots = ""
            for i in range(10):
                dots += "<span class='d-on'></span>" if i < score else "<span class='d-off'></span>"

            pos_section = ""
            if "HOLD" in sig:
                pos_section = (
                    "<div class='market-pos'>"
                    "<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;'>"
                    "<span style='font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;'>Position</span>"
                    "<span class='pos-margin-badge'>💰 Margin: $" + margin_val + "</span>"
                    "</div>"
                    "<div class='pos-grid'>"
                    "<div><div class='pos-item-label'>Entry</div><div class='pos-item-val entry'>" + entry_val + "</div></div>"
                    "<div><div class='pos-item-label'>Take Profit</div><div class='pos-item-val tp'>" + tp_val + "</div></div>"
                    "<div><div class='pos-item-label'>Stop Loss</div><div class='pos-item-val sl'>" + sl_val + "</div></div>"
                    "</div>"
                    "</div>"
                )
            else:
                pos_section = "<div class='market-ind'>⚡ " + reasons_str + "</div>"

            monitor_html += (
                "<div class='market-card' style='cursor:pointer;' data-sym='" + sym.replace("/", "") + "' onclick='openChart(this.dataset.sym)'>"
                "<div class='market-top'>"
                "<span class='market-pair'>" + sym + "</span>"
                "<span class='market-price' id='p-" + clean + "'>$" + str(d.get("price", "--")) + "</span>"
                "</div>"
                "<div class='market-row'>"
                "<span class='market-chg up' id='c-" + clean + "'>--</span>"
                + pill +
                "<div class='score-dots'>" + dots + "</div>"
                "</div>"
                + pos_section +
                "</div>"
            )

        open_positions_html = ""
        for d in self.dashboard:
            sig = d.get("signal", "")
            if "HOLD" not in sig:
                continue
            sym = d.get("symbol", "")
            clean = sym.replace("/", "").lower()
            side = "BUY" if "BUY" in sig else "SELL"
            side_cls = "buy" if side == "BUY" else "sell"
            side_pill = "<span class='pos-side side-buy'>BUY</span>" if side == "BUY" else "<span class='pos-side side-sell'>SELL</span>"
            entry_val = "$" + str(d.get("entry", "--"))
            tp_val = "$" + str(d["tp"]) if d.get("tp") else "-"
            sl_val = "$" + str(d["sl"]) if d.get("sl") else "-"
            margin_val = str(d.get("margin", MARGIN_PER_TRADE))
            score = str(d.get("score", 0))
            reasons_str = ", ".join(d.get("reasons", [])) or ""
            open_positions_html += (
                "<div class='pos-card " + side_cls + "'>"
                "<div class='pos-card-top'>"
                "<div style='display:flex;align-items:center;gap:8px;'>"
                "<span class='pos-pair'>" + sym + "</span>"
                + side_pill +
                "</div>"
                "<span class='pos-live-price' id='pp-" + clean + "'>$" + str(d.get("price", "--")) + "</span>"
                "</div>"
                "<div class='pos-details'>"
                "<div><div class='pos-d-label'>Entry</div><div class='pos-d-val' style='color:var(--yellow);'>" + entry_val + "</div></div>"
                "<div><div class='pos-d-label'>Take Profit</div><div class='pos-d-val' style='color:var(--green);'>" + tp_val + "</div></div>"
                "<div><div class='pos-d-label'>Stop Loss</div><div class='pos-d-val' style='color:var(--red);'>" + sl_val + "</div></div>"
                "</div>"
                "<div class='pos-footer'>"
                "<span class='pos-margin-tag'>💰 $" + margin_val + "</span>"
                "<span class='pos-score-tag'>Score " + score + "/10</span>"
                "<span class='pos-reasons'>" + reasons_str[:40] + "</span>"
                "</div>"
                "</div>"
            )
        if not open_positions_html:
            open_positions_html = "<div class='empty-state'><div class='empty-icon'>📊</div><div class='empty-text'>No open positions — bot is scanning markets</div></div>"

        log_html = ""
        for t in reversed(trades[-50:]):
            pv = float(t.get("pnl", 0))
            is_win = pv >= 0
            result = t.get("result", "SL")
            side = str(t.get("side", "")).upper()
            side_pill = "<span class='log-side side-buy'>BUY</span>" if side == "BUY" else "<span class='log-side side-sell'>SELL</span>"
            result_span = "<span class='log-result win'>🎯 TP</span>" if result == "TP" else "<span class='log-result loss'>🛑 SL</span>"
            log_html += (
                "<div class='log-card " + ("win" if is_win else "loss") + "'>"
                "<div class='log-top'>"
                "<div class='log-pair-row'>"
                "<span class='log-pair'>" + str(t.get("symbol", "")) + "</span>"
                + side_pill +
                "</div>"
                "<span class='log-pnl " + ("win" if is_win else "loss") + "'>" + ("+" if is_win else "") + "$" + str(round(pv, 2)) + "</span>"
                "</div>"
                "<div class='log-grid'>"
                "<div class='log-cell'><div class='log-cell-label'>Entry</div><div class='log-cell-val'>$" + str(t.get("entry", "")) + "</div></div>"
                "<div class='log-cell'><div class='log-cell-label'>Exit</div><div class='log-cell-val'>$" + str(t.get("exit", "")) + "</div></div>"
                "<div class='log-cell'><div class='log-cell-label'>Time</div><div class='log-cell-val'>" + str(t.get("time", "")) + "</div></div>"
                "</div>"
                "<div class='log-footer'>"
                + result_span +
                "<span class='log-score'>Score: " + str(t.get("score", "-")) + "/10</span>"
                "<span class='log-margin'>💰 $" + str(t.get("margin", 100)) + "</span>"
                "</div>"
                "</div>"
            )
        if not log_html:
            log_html = "<div class='empty-state'><div class='empty-icon'>📋</div><div class='empty-text'>No trades yet — strategies are scanning</div></div>"

        html = (
            "<!DOCTYPE html><html lang='en'><head>"
            "<meta charset='UTF-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1.0,maximum-scale=1.0'>"
            "<title>WhaleTrader Ultimate</title>"
            "<style>" + css + "</style>"
            "</head><body>"
            "<nav>"
            "<div class='nav-logo'>"
            "<div class='nav-logo-icon'>🐋</div>"
            "<span class='nav-logo-text'>WhaleTrader Ultimate</span>"
            "</div>"
            "<div class='nav-right'>"
            "<span class='live-dot'></span>"
            "<span class='sync-text' id='clk'>-- IST</span>"
            "<span class='" + risk_badge_cls + "'>" + risk_badge_text + "</span>"
            "</div>"
            "</nav>"
            "<div class='tabs'>"
            "<div class='tab active' onclick='showTab(0)'>📊 Overview</div>"
            "<div class='tab' onclick='showTab(1)'>📈 Markets</div>"
            "<div class='tab' onclick='showTab(2)'>💼 Positions (" + str(open_pos) + ")</div>"
            "<div class='tab' onclick='showTab(3)'>📋 History</div>"
            "<div class='tab' onclick='showTab(4)'>🧠 Strategies</div>"
            "</div>"

            "<div class='panel active' id='tab0'>"
            "<div class='overview-header'>"
            "<div style='font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;font-weight:600;margin-bottom:6px;'>Total Account Value</div>"
            "<div class='overview-equity'>$" + str(wallet) + "</div>"
            "<div class='overview-pnl " + ("pos" if total_pnl >= 0 else "neg") + "'>"
            + ("+$" if total_pnl >= 0 else "$") + str(total_pnl) + " (" + ("+" if total_pnl >= 0 else "") + str(pnl_pct) + "%) All Time"
            "</div>"
            "<div style='font-size:10px;color:var(--muted2);margin-top:6px;font-family:var(--mono);'>Sync: " + now_str + "</div>"
            "</div>"
            "<div class='stats-row'>"
            "<div class='stat-box'><div class='stat-box-label'>Win Rate</div><div class='stat-box-val " + ("green" if win_rate >= 55 else "yellow" if win_rate >= 45 else "red") + "'>" + str(win_rate) + "%</div><div class='stat-box-sub'>" + str(wins) + "W / " + str(losses) + "L / " + str(total_t) + " total</div></div>"
            "<div class='stat-box'><div class='stat-box-label'>Today P&amp;L</div><div class='stat-box-val " + ("green" if today_pnl >= 0 else "red") + "'>" + ("+" if today_pnl >= 0 else "") + "$" + str(today_pnl) + "</div><div class='stat-box-sub'>" + str(today_trades) + " trades today</div></div>"
            "</div>"
            "<div class='stats-row'>"
            "<div class='stat-box'><div class='stat-box-label'>Best Trade</div><div class='stat-box-val green'>+$" + str(best) + "</div><div class='stat-box-sub'>All time high</div></div>"
            "<div class='stat-box'><div class='stat-box-label'>Worst Trade</div><div class='stat-box-val red'>$" + str(worst) + "</div><div class='stat-box-sub'>Max drawdown</div></div>"
            "</div>"
            "<div class='stats-row'>"
            "<div class='stat-box'><div class='stat-box-label'>Positions</div><div class='stat-box-val yellow'>" + str(open_pos) + "/" + str(MAX_POSITIONS) + "</div><div class='stat-box-sub'>Max allowed</div></div>"
            "<div class='stat-box'><div class='stat-box-label'>Strategies</div><div class='stat-box-val yellow'>15+</div><div class='stat-box-sub'>Active / 10x Leverage</div></div>"
            "</div>"
            "<div class='chart-box'>"
            "<div class='chart-box-title'>Daily P&amp;L — Last 14 Days</div>"
            "<canvas id='pnlChart' style='max-height:140px;'></canvas>"
            "</div>"
            "</div>"

            "<div class='panel' id='tab1'>"
            + monitor_html +
            "</div>"

            "<div class='panel' id='tab2'>"
            + open_positions_html +
            "</div>"

            "<div class='panel' id='tab3'>"
            + log_html +
            "</div>"

            "<div class='panel' id='tab4'>"
            "<div style='padding:4px 0 12px;font-size:11px;color:var(--muted);letter-spacing:.05em;'>📊 Performance tracked from trade history — updates each cycle</div>"
            + strat_html +
            "</div>"

            "<footer>WhaleTrader Ultimate &middot; 60s auto-sync &middot; Paper Trading &middot; $1,000 Capital &middot; 15+ Strategies</footer>"

            "<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js'></script>"
            "<script>"
            "function updateClock(){var n=new Date();var ist=new Date(n.getTime()+19800000);var h=ist.getUTCHours(),m=ist.getUTCMinutes(),s=ist.getUTCSeconds();var ap=h>=12?'PM':'AM';var hh=h%12||12;function p(x){return x<10?'0'+x:''+x;}document.getElementById('clk').textContent=p(hh)+':'+p(m)+':'+p(s)+' '+ap+' IST';}setInterval(updateClock,1000);updateClock();"
            "function showTab(i){document.querySelectorAll('.panel').forEach(function(p,j){p.classList.toggle('active',i===j);});document.querySelectorAll('.tab').forEach(function(t,j){t.classList.toggle('active',i===j);});window._activeTab=i;}""function openChart(sym){var map={'BTCUSDT':'BTC','ETHUSDT':'ETH','SOLUSDT':'SOL','BNBUSDT':'BNB','XRPUSDT':'XRP','DOGEUSDT':'DOGE'};var base=map[sym]||sym.replace('USDT','');window.open('https://www.tradingview.com/chart/?symbol=OKX:'+base+'USDT&interval=1','_blank');}""(function(){var sx=0,sy=0;document.addEventListener('touchstart',function(e){sx=e.touches[0].clientX;sy=e.touches[0].clientY;},{passive:true});document.addEventListener('touchend',function(e){var dx=e.changedTouches[0].clientX-sx;var dy=e.changedTouches[0].clientY-sy;if(Math.abs(dx)>Math.abs(dy)&&Math.abs(dx)>50){var cur=window._activeTab||0;if(dx<0&&cur<4)showTab(cur+1);else if(dx>0&&cur>0)showTab(cur-1);}},{passive:true});})();"
            "var ctx=document.getElementById('pnlChart').getContext('2d');"
            "new Chart(ctx,{type:'bar',data:{labels:" + chart_labels + ",datasets:[{data:" + chart_values + ",backgroundColor:" + chart_colors + ",borderRadius:4,borderSkipped:false}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#848e9c',font:{size:9}}},y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#848e9c',font:{size:9},callback:function(v){return'$'+v;}}}}}});"
            "var syms=['btcusdt','ethusdt','solusdt','bnbusdt','xrpusdt','dogeusdt'];"
            "function connectWS(){var ws=new WebSocket('wss://stream.binance.com:9443/ws/'+syms.map(function(s){return s+'@ticker';}).join('/'));"
            "ws.onmessage=function(e){var d=JSON.parse(e.data);var s=d.s.toLowerCase();var pe=document.getElementById('p-'+s);var pp=document.getElementById('pp-'+s);var ce=document.getElementById('c-'+s);if(!pe)return;var price=parseFloat(d.c);var chg=parseFloat(d.P);var priceStr=price<1?'$'+price.toFixed(5):'$'+price.toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2});pe.textContent=priceStr;if(pp)pp.textContent=priceStr;pe.style.color=chg>=0?'#0ecb81':'#f6465d';setTimeout(function(){pe.style.color='';},500);if(ce){ce.textContent=(chg>=0?'+':'')+chg.toFixed(2)+'%';ce.className='market-chg '+(chg>=0?'up':'dn');}};"
            "ws.onclose=function(){setTimeout(connectWS,3000);};}connectWS();"
            "</script>"
            "</body></html>"
        )
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Dashboard saved — index.html")

if __name__ == "__main__":
    engine = WhaleEngine()
    engine.run()