import time
import logging
import os
import json
import ccxt
import numpy as np
import requests
from datetime import datetime, timedelta
from collections import deque

# ═══════════════════════════════════════════════
# CUSTOM IST LOGGER
# ═══════════════════════════════════════════════
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
        return ist.strftime("%Y-%m-%d %H:%M:%S IST")

_handler = logging.StreamHandler()
_handler.setFormatter(ISTFormatter("%(asctime)s | %(levelname)s | %(message)s"))
logging.root.setLevel(logging.INFO)
logging.root.handlers = [_handler]
log = logging.getLogger("WhaleTrader_Ultimate")

# ═══════════════════════════════════════════════
# CONFIG — PROFESSIONAL SCALPER SETTINGS
# ═══════════════════════════════════════════════
SYMBOLS          = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
INITIAL_CAPITAL  = 1000.0
MARGIN_PER_TRADE = 100.0
MARGIN_BLAST     = 200.0
LEVERAGE         = 5
HISTORY_FILE     = "history.json"

# Risk Management
MIN_SCORE        = 6
VOL_MULT         = 1.3
FEE_RATE         = 0.0004
SLIPPAGE_RATE    = 0.0005
MAX_DAILY_LOSS   = 50.0

# BB Settings
BB_PERIOD   = 20
BB_STD      = 2.0
BB_FAST_P   = 10
BB_FAST_S   = 1.5

# S/R Consolidation
SR_PERIOD   = 20
SR_THRESH   = 0.003
ATR_PERIOD  = 14

# TP/SL
TP_MULT     = 2.0
SL_MULT     = 1.0
TRAIL_STEP  = 0.15

# Risk management
MAX_POSITIONS   = 4

# ═══════════════════════════════════════════════
# NEW: VWAP CALCULATION (Institutional Strategy)
# ═══════════════════════════════════════════════
def calculate_vwap(highs, lows, closes, volumes):
    """Volume Weighted Average Price — institutional level"""
    typical_price = (highs + lows + closes) / 3
    cumulative_tp_vol = np.cumsum(typical_price * volumes)
    cumulative_vol = np.cumsum(volumes)
    vwap = cumulative_tp_vol / (cumulative_vol + 1e-9)
    return vwap

def vwap_strategy(closes, vwap):
    """VWAP scalping: price below VWAP = buy, above = sell"""
    price = closes[-1]
    vwap_val = vwap[-1]
    
    if price < vwap_val * 0.998:  # 0.2% below VWAP
        return "buy", 3, "VWAP BUY"
    elif price > vwap_val * 1.002:  # 0.2% above VWAP
        return "sell", 3, "VWAP SELL"
    return None, 0, ""

# ═══════════════════════════════════════════════
# NEW: EMA RIBBON (Trend Confirmation)
# ═══════════════════════════════════════════════
def ema_ribbon(closes):
    """EMA 9, 21, 50 ribbon — trend strength"""
    ema9 = ema_calc(closes, 9)[-1]
    ema21 = ema_calc(closes, 21)[-1]
    ema50 = ema_calc(closes, 50)[-1] if len(closes) >= 50 else ema21
    
    # Bullish: 9 > 21 > 50
    if ema9 > ema21 > ema50:
        return "BULL", 2
    # Bearish: 9 < 21 < 50
    elif ema9 < ema21 < ema50:
        return "BEAR", 2
    return "NEUTRAL", 0

# ═══════════════════════════════════════════════
# NEW: ORDER FLOW IMBALANCE
# ═══════════════════════════════════════════════
def order_flow_imbalance(opens, closes, volumes):
    """Detect buying/selling pressure from candle data"""
    # Green candles with high volume = buying pressure
    # Red candles with high volume = selling pressure
    
    recent_bodies = closes[-10:] - opens[-10:]
    recent_vols = volumes[-10:]
    
    buy_pressure = np.sum(np.where(recent_bodies > 0, recent_bodies * recent_vols, 0))
    sell_pressure = np.sum(np.where(recent_bodies < 0, abs(recent_bodies) * recent_vols, 0))
    
    total = buy_pressure + sell_pressure + 1e-9
    buy_ratio = buy_pressure / total
    
    if buy_ratio > 0.65:  # 65%+ buying pressure
        return "buy", 2, "ORDER FLOW BUY"
    elif buy_ratio < 0.35:  # 65%+ selling pressure
        return "sell", 2, "ORDER FLOW SELL"
    return None, 0, ""

# ═══════════════════════════════════════════════
# HELPERS
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

def rsi_calc(closes, period):
    d = np.diff(closes.astype(float))
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = np.mean(g[:period]) if len(g) >= period else 0.0
    al = np.mean(l[:period]) if len(l) >= period else 0.0
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
    return 100.0 - (100.0 / (1.0 + ag / (al + 1e-10)))

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
        return "SIDEWAYS", 0
    
    ema50  = ema_calc(closes, 50)[-1]
    ema200 = ema_calc(closes, min(100, len(closes)-1))[-1]
    
    atr = atr_calc(highs, lows, closes, 14)
    price = closes[-1]
    atr_pct = (atr / price) * 100
    
    if ema50 > ema200 * 1.002:
        regime = "BULL"
    elif ema50 < ema200 * 0.998:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS"
    
    return regime, atr_pct

# ═══════════════════════════════════════════════
# DARVAS BOX STRATEGY
# ═══════════════════════════════════════════════
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
    
    breakout_up   = current_close  > resistance
    breakout_down = current_close  < support
    
    trail_sl = None
    signal   = None
    reasons  = []
    
    if breakout_up:
        signal   = "buy"
        trail_sl = round(support, 6)
        reasons  = [
            "DARVAS BOX",
            "Break: $" + str(round(resistance, 4)),
            "VOL " + str(round(current_volume / (vol_ma + 1e-9), 1)) + "x",
        ]
    elif breakout_down:
        signal   = "sell"
        trail_sl = round(resistance, 6)
        reasons  = [
            "DARVAS BOX",
            "Break: $" + str(round(support, 4)),
            "VOL " + str(round(current_volume / (vol_ma + 1e-9), 1)) + "x",
        ]
    
    return signal, round(resistance, 6), round(support, 6), trail_sl, reasons

# ═══════════════════════════════════════════════
# MULTI-TIMEFRAME CONFLUENCE ENGINE
# ═══════════════════════════════════════════════
def mtf_confluence_score(closes_1m, closes_5m, closes_15m):
    def trend_dir(closes, period=20):
        if len(closes) < period:
            return 0
        sma = np.mean(closes[-period:])
        slope = np.polyfit(np.arange(period), closes[-period:], 1)[0]
        if closes[-1] > sma * 1.001 and slope > 0:
            return 1
        elif closes[-1] < sma * 0.999 and slope < 0:
            return -1
        return 0
    
    d1 = trend_dir(closes_1m)
    d5 = trend_dir(closes_5m)
    d15 = trend_dir(closes_15m)
    
    if d1 == d5 == d15 and d1 != 0:
        return 3, d1
    elif (d1 == d5 or d1 == d15) and d1 != 0:
        return 2, d1
    elif d1 != 0:
        return 1, d1
    return 0, 0

# ═══════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════
class WhaleEngine:
    def __init__(self):
        self.state = self.load_history()
        self.dashboard = []
        self._last_is_blast = False
        self.live_prices = {}
        
        api_key = os.getenv("BINANCE_API_KEY")
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
            return data
        except Exception:
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
        today = ist_now().strftime("%Y-%m-%d")
        daily_loss = abs(min(self.state["stats"]["daily_pnl"].get(today, 0), 0))
        open_pos = len(self.state["active_positions"])
        
        if daily_loss >= MAX_DAILY_LOSS:
            log.warning(f"Daily loss limit hit: ${daily_loss:.2f} — trading paused")
            return False
        
        if score >= 8:
            log.info(f"High confidence {score}/10 — no position limit!")
            return True
        
        if open_pos >= MAX_POSITIONS:
            log.info(f"Max positions: {open_pos}/{MAX_POSITIONS} (score {score}/10)")
            return False
        
        return True
    
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
    
    def fetch_data(self, symbol, limit=120, vol_map=None):
        for attempt in range(3):
            try:
                result_1m  = self.fetch_okx_public(symbol, "1m",  100)
                result_5m  = self.fetch_okx_public(symbol, "5m",  limit)
                result_15m = self.fetch_okx_public(symbol, "15m", 60)
                
                if result_1m and result_5m and result_15m:
                    opens_1m, highs_1m, lows_1m, closes_1m, vols_1m = result_1m
                    opens_5m, highs_5m, lows_5m, closes_5m, vols_5m = result_5m
                    closes_15m = result_15m[3]
                    
                    log.info(f"OKX Real: {symbol} @ ${round(float(closes_5m[-1]),4)} | Vol: {vols_5m[-1]:.0f}")
                    
                    return (opens_5m, highs_5m, lows_5m, closes_5m, vols_5m, closes_15m,
                            closes_1m, opens_1m, highs_1m, lows_1m, vols_1m)
                
                log.warning(f"OKX retry {attempt+1}/3 for {symbol}")
                time.sleep(2)
            except Exception as e:
                log.warning(f"Fetch error {symbol}: {e}")
        
        log.error(f"OKX failed after 3 retries: {symbol}")
        return None
    
    def score_signal(self, opens, highs, lows, closes, vols, closes15, 
                     closes_1m=None, opens_1m=None, highs_1m=None, lows_1m=None, vols_1m=None):
        score, direction, reasons = 0, None, []
        is_blast = False
        price = float(closes[-1])
        
        try:
            r = rsi_calc(closes, 14)
        except:
            r = 50.0
        
        try:
            regime_result = market_regime(closes, highs, lows)
            regime = regime_result[0] if isinstance(regime_result, tuple) else regime_result
        except:
            regime = "SIDEWAYS"
        
        reasons.append(regime)
        
        upper, mid, lower = bollinger_calc(closes, BB_PERIOD, BB_STD)
        u_fast, m_fast, l_fast = bollinger_calc(closes, BB_FAST_P, BB_FAST_S)
        
        vol_avg = float(np.mean(vols[-15:-1])) if len(vols) > 15 else float(np.mean(vols))
        vol_now = float(vols[-1])
        vol_ratio = vol_now / (vol_avg + 1e-9)
        vol_spike = vol_ratio >= VOL_MULT
        
        atr = atr_calc(highs, lows, closes, ATR_PERIOD)
        
        # Multi-Timeframe Confluence
        mtf_bonus = 0
        if closes_1m is not None:
            mtf_score, mtf_dir = mtf_confluence_score(closes_1m, closes, closes15)
            mtf_bonus = mtf_score * 2
            if mtf_score >= 2:
                reasons.append(f"MTF:{'BULL' if mtf_dir==1 else 'BEAR'}")
        
        # ════════════════════════════════════════
        # STRATEGY 1: BB BOUNCE
        # ════════════════════════════════════════
        body_now = float(closes[-1]) - float(opens[-1])
        avg_body = float(np.mean([abs(float(closes[i])-float(opens[i])) for i in range(-6,-1)]))
        
        if price <= lower:
            score += 4
            direction = "buy"
            reasons.append("BB BOUNCE LOW")
            if body_now > 0:
                score += 2
                reasons.append("REVERSAL CANDLE")
        elif price >= upper:
            score += 4
            direction = "sell"
            reasons.append("BB BOUNCE HIGH")
            if body_now < 0:
                score += 2
                reasons.append("REVERSAL CANDLE")
        
        # ════════════════════════════════════════
        # STRATEGY 2: DOUBLE BB ZONE
        # ════════════════════════════════════════
        if direction is None:
            in_buy_zone  = l_fast <= price <= lower
            in_sell_zone = upper <= price <= u_fast
            if in_buy_zone:
                score += 4
                direction = "buy"
                reasons.append("DOUBLE BB BUY ZONE")
            elif in_sell_zone:
                score += 4
                direction = "sell"
                reasons.append("DOUBLE BB SELL ZONE")
        
        # ════════════════════════════════════════
        # STRATEGY 3: BB BLAST
        # ════════════════════════════════════════
        band_widths = []
        for i in range(-20, -1):
            try:
                u_i, m_i, l_i = bollinger_calc(closes[:i], BB_PERIOD, BB_STD)
                band_widths.append(u_i - l_i)
            except:
                pass
        avg_band_width = float(np.mean(band_widths)) if band_widths else (upper - lower)
        band_width_now = upper - lower
        squeeze = band_width_now < avg_band_width * 0.75
        
        big_candle = abs(body_now) > avg_body * 2.0
        
        if squeeze and big_candle:
            if body_now > 0 and price > mid:
                score += 6
                direction = "buy"
                is_blast = True
                reasons.append("BB BLAST UP 🚀")
            elif body_now < 0 and price < mid:
                score += 6
                direction = "sell"
                is_blast = True
                reasons.append("BB BLAST DN 🚀")
        
        # ════════════════════════════════════════
        # STRATEGY 4: S/R CONSOLIDATION BREAKOUT
        # ════════════════════════════════════════
        if direction is None and len(highs) >= SR_PERIOD:
            resistance = float(np.max(highs[-SR_PERIOD:]))
            support    = float(np.min(lows[-SR_PERIOD:]))
            
            broke_resistance = price > resistance * (1 + SR_THRESH)
            broke_support    = price < support * (1 - SR_THRESH)
            
            if broke_resistance:
                score += 4
                direction = "buy"
                reasons.append("SR BREAKOUT UP")
                if vol_spike:
                    score += 2
                    reasons.append("VOL CONFIRM")
            elif broke_support:
                score += 4
                direction = "sell"
                reasons.append("SR BREAKOUT DN")
                if vol_spike:
                    score += 2
                    reasons.append("VOL CONFIRM")
        
        # ════════════════════════════════════════
        # NEW STRATEGY 5: VWAP SCALPING
        # ════════════════════════════════════════
        if vols_1m is not None and len(vols_1m) > 0:
            vwap = calculate_vwap(highs_1m, lows_1m, closes_1m, vols_1m)
            vwap_sig, vwap_score, vwap_reason = vwap_strategy(closes_1m, vwap)
            
            if vwap_sig and vwap_sig == direction:
                score += vwap_score
                reasons.append(vwap_reason)
            elif vwap_sig and direction is None:
                direction = vwap_sig
                score += vwap_score
                reasons.append(vwap_reason)
        
        # ════════════════════════════════════════
        # NEW STRATEGY 6: EMA RIBBON
        # ════════════════════════════════════════
        ema_sig, ema_score = ema_ribbon(closes)
        if ema_sig == "BULL" and direction == "buy":
            score += ema_score
            reasons.append("EMA RIBBON BULL")
        elif ema_sig == "BEAR" and direction == "sell":
            score += ema_score
            reasons.append("EMA RIBBON BEAR")
        
        # ════════════════════════════════════════
        # NEW STRATEGY 7: ORDER FLOW
        # ════════════════════════════════════════
        if opens_1m is not None:
            of_sig, of_score, of_reason = order_flow_imbalance(opens_1m, closes_1m, vols_1m)
            if of_sig and of_sig == direction:
                score += of_score
                reasons.append(of_reason)
        
        # Volume bonus
        if direction and vol_spike and "VOL" not in str(reasons):
            score += 2
            reasons.append("VOL " + str(round(vol_ratio, 1)) + "x")
        
        # MTF confluence bonus
        if direction and mtf_bonus > 0:
            score += mtf_bonus
        
        self._last_is_blast = is_blast
        
        return score, direction, r, upper, lower, reasons, MIN_SCORE
    
    def check_positions(self, symbol, current_price):
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
        
        sl_dist    = abs(entry - sl)
        trail_dist = sl_dist * TRAIL_STEP
        
        last_prices_arr = list(self.state.get("last_prices_history", {}).get(symbol, [entry] * 20))
        last_prices_arr.append(price)
        last_prices_arr = last_prices_arr[-20:]
        bb_mid = round(sum(last_prices_arr) / len(last_prices_arr), 6)
        
        profit = (price - entry) if side == "buy" else (entry - price)
        
        if side == "buy":
            if price > tp:
                tp = round(price + trail_dist, 6)
                self.state["active_positions"][symbol]["tp"] = tp
            if profit > sl_dist:
                new_sl = round(bb_mid - trail_dist, 6)
                if new_sl > sl:
                    sl = new_sl
                    self.state["active_positions"][symbol]["sl"] = sl
        else:
            if price < tp:
                tp = round(price - trail_dist, 6)
                self.state["active_positions"][symbol]["tp"] = tp
            if profit > sl_dist:
                new_sl = round(bb_mid + trail_dist, 6)
                if new_sl < sl:
                    sl = new_sl
                    self.state["active_positions"][symbol]["sl"] = sl
        
        hit, reason, exit_p = False, "", price
        if side == "buy":
            if price >= tp:
                hit, reason, exit_p = True, "TP", tp
            elif price <= sl:
                hit, reason, exit_p = True, "TSL", sl
        else:
            if price <= tp:
                hit, reason, exit_p = True, "TP", tp
            elif price >= sl:
                hit, reason, exit_p = True, "TSL", sl
        
        if hit:
            if side == "buy":
                gross_pnl = (exit_p - entry) * qty
            else:
                gross_pnl = (entry - exit_p) * qty
            
            entry_fee = entry * qty * FEE_RATE
            exit_fee  = exit_p * qty * FEE_RATE
            fees = entry_fee + exit_fee
            
            slippage_cost = entry * qty * SLIPPAGE_RATE
            
            pnl = gross_pnl - fees - slippage_cost
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
            
            self.state["trades"].append({
                "time":   exit_time_str,
                "symbol": symbol,
                "side":   side.upper(),
                "entry":  round(entry, 6),
                "exit":   round(exit_p, 6),
                "pnl":    pnl,
                "fees":   round(fees, 4),
                "result": reason,
                "score":  pos.get("score", 0),
                "margin": margin,
            })
            del self.state["active_positions"][symbol]
            icon = "PROFIT" if pnl > 0 else "LOSS"
            log.info(f"{icon} {symbol} {side.upper()} | Entry:${entry} Exit:${exit_p} Gross:${gross_pnl:.2f} Fees:${fees:.2f} Net:${pnl} | {reason}")
            self.save_history()
    
    def run(self):
        log.info("WhaleTrader Ultimate — Cycle Start")
        self.cleanup_stale_positions()
        
        import random
        symbols_this_run = SYMBOLS.copy()
        random.shuffle(symbols_this_run)
        
        for symbol in symbols_this_run:
            try:
                time.sleep(3)
                result = self.fetch_data(symbol)
                if result is None:
                    last_price = self.state["last_prices"].get(symbol, 0)
                    self.dashboard.append({
                        "symbol": symbol, "price": last_price,
                        "signal": "SCANNING", "score": 0,
                        "entry": None, "tp": None, "sl": None,
                        "reasons": ["Data unavailable"],
                    })
                    continue
                
                (opens, highs, lows, closes, vols, closes15, 
                 closes_1m, opens_1m, highs_1m, lows_1m, vols_1m) = result
                
                current_price = float(closes[-1])
                self.check_positions(symbol, current_price)
                
                price = round(float(closes[-1]), 6)
                self.state["last_prices"][symbol] = price
                self.live_prices[symbol] = price
                
                if "last_prices_history" not in self.state:
                    self.state["last_prices_history"] = {}
                hist = self.state["last_prices_history"].get(symbol, [])
                hist.append(price)
                self.state["last_prices_history"][symbol] = hist[-20:]
                
                if symbol in self.state["active_positions"]:
                    pos = self.state["active_positions"][symbol]
                    self.dashboard.append({
                        "symbol": symbol, "price": price,
                        "signal": "HOLDING " + pos["side"].upper(),
                        "score": pos.get("score", 0),
                        "entry": pos["entry"],
                        "tp": pos["tp"],
                        "sl": pos["sl"],
                        "reasons": pos.get("reasons", []),
                    })
                    continue
                
                score, direction, r, upper, lower, reasons, adaptive_min = self.score_signal(
                    opens, highs, lows, closes, vols, closes15,
                    closes_1m, opens_1m, highs_1m, lows_1m, vols_1m)
                
                log.info(f"{symbol} | ${price} | RSI:{round(r,1)} | Score:{score}/10 | Regime:{reasons[0] if reasons else '?'} | Min:{adaptive_min}")
                
                darvas_sig, box_top, box_bot, darvas_sl, darvas_reasons = darvas_box_strategy(
                    opens, highs, lows, closes, vols)
                
                if darvas_sig and darvas_sig == direction:
                    score = min(score + 2, 10)
                    reasons = reasons + darvas_reasons
                    log.info(f"DARVAS CONFIRM {symbol} {darvas_sig.upper()}")
                elif darvas_sig and not direction and darvas_sl:
                    direction = darvas_sig
                    score = max(score, 5)
                    reasons = darvas_reasons
                
                if direction and score >= adaptive_min and self.can_trade(score):
                    is_blast = getattr(self, "_last_is_blast", False)
                    margin = MARGIN_BLAST if is_blast else MARGIN_PER_TRADE
                    av = atr_calc(highs, lows, closes, ATR_PERIOD)
                    
                    if direction == "buy":
                        tp = round(price + av * TP_MULT, 6)
                        swing_low = float(np.min(lows[-10:]))
                        sl = round(min(swing_low - av * 0.5, price - av * SL_MULT), 6)
                    else:
                        tp = round(price - av * TP_MULT, 6)
                        swing_high = float(np.max(highs[-10:]))
                        sl = round(max(swing_high + av * 0.5, price + av * SL_MULT), 6)
                    
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
                
                if symbol in self.state["active_positions"]:
                    pos = self.state["active_positions"][symbol]
                    self.dashboard.append({
                        "symbol": symbol, "price": price,
                        "signal": "HOLDING " + pos["side"].upper(),
                        "score": pos.get("score", score),
                        "entry": pos["entry"],
                        "tp": pos["tp"],
                        "sl": pos["sl"],
                        "reasons": pos.get("reasons", reasons),
                    })
                else:
                    self.dashboard.append({
                        "symbol": symbol,
                        "price": price,
                        "signal": direction.upper() + "  " + str(score) + "/10" if direction else "SCANNING",
                        "score": score,
                        "entry": None,
                        "tp": None,
                        "sl": None,
                        "reasons": reasons,
                    })
            
            except Exception as e:
                log.error(f"Error {symbol}: {e}")
                self.dashboard.append({
                    "symbol": symbol, "price": 0, "signal": "ERROR",
                    "score": 0, "entry": 0, "tp": None, "sl": None, "reasons": [str(e)],
                })
        
        self.save_history()
        self.build_dashboard()
    
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
        pnl_color = "#00e676" if total_pnl >= 0 else "#ff1744"
        pnl_prefix = "+" if total_pnl >= 0 else ""
        wr_color = "#00e676" if win_rate >= 55 else "#ffab00" if win_rate >= 45 else "#ff1744"
        now_str = ist_str()
        
        today = ist_now().strftime("%Y-%m-%d")
        today_pnl = round(stats.get("daily_pnl", {}).get(today, 0.0), 2)
        today_trades = stats.get("daily_trades", {}).get(today, 0)
        today_color = "#00e676" if today_pnl >= 0 else "#ff1744"
        today_prefix = "+" if today_pnl >= 0 else ""
        
        open_pos = len(state.get("active_positions", {}))
        daily_loss = abs(min(stats.get("daily_pnl", {}).get(today, 0), 0))
        trading_ok = daily_loss < MAX_DAILY_LOSS and open_pos < MAX_POSITIONS
        
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
            clean = sym.replace("/", " ").lower().replace(" ", "")
            sig = d.get("signal", "SCANNING")
            score = int(d.get("score", 0))
            reasons = ", ".join(d.get("reasons", [])) or "Waiting..."
            tp_val = "$" + str(d["tp"]) if d.get("tp") else "-"
            sl_val = "$" + str(d["sl"]) if d.get("sl") else "-"
            entry_val = "$" + str(d["entry"]) if d.get("entry") and "HOLD" in sig else "-"
            
            if "HOLD" in sig:   pill = "<span class='pill-hold'>" + sig + "</span>"
            elif "BUY" in sig:  pill = "<span class='pill-buy'>" + sig + "</span>"
            elif "SELL" in sig: pill = "<span class='pill-sell'>" + sig + "</span>"
            else:               pill = "<span class='pill-scan'>SCANNING</span>"
            
            dots = ""
            for i in range(10):
                dots += "<span class='dotf'></span>" if i < score else "<span class='dot'></span>"
            
            monitor_rows += (
                "<div class='monitor-card'>"
                "<div class='monitor-top'>"
                "<div class='monitor-sym'>" + sym + "</div>"
                "<span id='p-" + clean + "' class='monitor-price'>$" + str(d.get('price','--')) + "</span>"
                "</div>"
                "<div class='monitor-row'>"
                "<div><span class='monitor-label'>24h</span> <span id='c-" + clean + "' class='monitor-val'>--</span></div>"
                "<div>" + pill + "</div>"
                "<div><span class='monitor-label'>Score</span> <span class='monitor-val' style='color:var(--green);'>" + str(score) + "/10</span></div>"
                "</div>"
                "<div class='monitor-bottom'>"
                + (
                    "<div><div class='monitor-label'>Entry</div><div class='monitor-val' style='color:var(--amber);'>" + entry_val + "</div></div>"
                    "<div><div class='monitor-label'>Take Profit</div><div class='monitor-val' style='color:var(--green);'>" + tp_val + "</div></div>"
                    "<div><div class='monitor-label'>Stop Loss</div><div class='monitor-val' style='color:var(--red);'>" + sl_val + "</div></div>"
                    if "HOLD" in sig else
                    "<div style='color:var(--muted);font-size:10px;'>⚡ " + reasons + "</div>"
                ) +
                "<div class='dots'>" + dots + "</div>"
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
            fees = t.get("fees", 0)
            history_rows += (
                "<div class='trade-card'>"
                "<div class='trade-top'>"
                "<div style='display:flex;align-items:center;gap:8px;'>"
                "<div class='trade-pair'>" + str(t.get("symbol", "")) + "</div>"
                "<span class='" + badge + "'>" + str(t.get("side", "")) + "</span>"
                "</div>"
                "<div class='trade-pnl " + pc + "'>" + prefix + "$" + str(round(pv,2)) + "</div>"
                "</div>"
                "<div class='trade-row'>"
                "<div class='trade-item'><div class='trade-item-label'>Time</div><div class='trade-item-val'>" + str(t.get("time", "")) + "</div></div>"
                "<div class='trade-item'><div class='trade-item-label'>Entry</div><div class='trade-item-val'>$" + str(t.get("entry", "")) + "</div></div>"
                "<div class='trade-item'><div class='trade-item-label'>Exit</div><div class='trade-item-val'>$" + str(t.get("exit", "")) + "</div></div>"
                "</div>"
                "<div class='trade-meta'>"
                "<span class='trade-result'>" + ("🎯 TP" if icon == "TP" else "🛑 SL") + "</span>"
                "<span class='trade-score'>Score: " + str(t.get("score", "-")) + "/10</span>"
                "<span class='trade-margin'>Fee: $" + str(round(fees, 2)) + "</span>"
                "</div>"
                "</div>"
            )
        
        if not history_rows:
            history_rows = "<div class='trade-card' style='text-align:center;color:var(--muted);'>No trades yet — bot is scanning...</div>"
        
        risk_badge = "<span class='badge-status status-ok'>TRADING ACTIVE</span>" if trading_ok else "<span class='badge-status status-limit'>LIMIT REACHED</span>"
        
        mobile_css = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root{--bg0:#03050a;--bg1:#07090f;--bg2:#0b0e17;--bg3:#0f1420;--border:#162035;--border2:#1e2d47;--text:#e2e8f5;--muted:#3d5470;--muted2:#2a3a52;--green:#00e676;--green2:#00c853;--red:#ff1744;--red2:#d50000;--blue:#2979ff;--amber:#ffab00;--purple:#7c4dff;--cyan:#00e5ff;--font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg0);color:var(--text);font-family:var(--font);font-size:13px;min-height:100vh;}
nav{background:rgba(3,5,10,0.98);backdrop-filter:blur(24px);border-bottom:1px solid var(--border);padding:0 16px;height:54px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;}
.logo{display:flex;align-items:center;gap:9px;font-size:15px;font-weight:900;color:#fff;}
.logo-icon{width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,#1a56ff,#00e676);display:flex;align-items:center;justify-content:center;font-size:15px;}
.badge-live{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.3);font-size:10px;font-weight:800;padding:3px 9px;border-radius:20px;display:flex;align-items:center;gap:5px;}
.pulse-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(0,230,118,.6);}70%{box-shadow:0 0 0 6px rgba(0,230,118,0);}100%{box-shadow:0 0 0 0 rgba(0,230,118,0);}}
.nav-right{display:flex;align-items:center;gap:6px;}
.badge-status{font-size:9px;font-weight:800;padding:4px 9px;border-radius:20px;}
.status-ok{background:rgba(0,230,118,.12);color:var(--green);border:1px solid rgba(0,230,118,.3);}
.status-limit{background:rgba(255,23,68,.12);color:var(--red);border:1px solid rgba(255,23,68,.3);}
.main{padding:14px 12px;max-width:640px;margin:0 auto;}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:18px;}
.stat-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:13px;}
.stat-label{font-size:9px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:700;margin-bottom:6px;}
.stat-value{font-size:20px;font-weight:900;font-family:var(--mono);}
.stat-sub{font-size:10px;color:var(--muted);margin-top:5px;}
.sec-head{display:flex;align-items:center;gap:8px;margin:18px 0 10px;}
.sec-title{font-size:9px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:800;}
.sec-line{flex:1;height:1px;background:var(--border);}
.chart-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:14px;margin-bottom:4px;}
.monitor-cards{display:flex;flex-direction:column;gap:9px;}
.monitor-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:13px;}
.monitor-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.monitor-sym{font-weight:900;color:#fff;font-size:16px;}
.monitor-price{font-family:var(--mono);font-size:17px;font-weight:800;color:#fff;}
.monitor-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.monitor-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:700;}
.monitor-val{font-family:var(--mono);font-size:13px;font-weight:700;}
.dots{display:flex;gap:3px;align-items:center;}
.dot{width:6px;height:6px;border-radius:50%;background:var(--border2);}
.dotf{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 5px rgba(0,230,118,.6);}
.pill-buy,.pill-sell,.pill-hold,.pill-scan{font-size:10px;font-weight:800;padding:4px 10px;border-radius:6px;text-transform:uppercase;white-space:nowrap;}
.pill-buy{background:rgba(0,230,118,.12);color:var(--green);border:1px solid rgba(0,230,118,.3);}
.pill-sell{background:rgba(255,23,68,.12);color:var(--red);border:1px solid rgba(255,23,68,.3);}
.pill-hold{background:rgba(41,121,255,.12);color:var(--blue);border:1px solid rgba(41,121,255,.3);}
.pill-scan{background:rgba(255,171,0,.1);color:var(--amber);border:1px solid rgba(255,171,0,.25);}
.trade-cards{display:flex;flex-direction:column;gap:8px;}
.trade-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:13px;}
.trade-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;}
.trade-pair{font-weight:900;color:#fff;font-size:15px;}
.trade-pnl{font-family:var(--mono);font-size:18px;font-weight:900;}
.trade-pnl.pos{color:var(--green);}
.trade-pnl.neg{color:var(--red);}
.trade-row{display:flex;gap:0;margin-bottom:9px;background:var(--bg2);border-radius:8px;overflow:hidden;}
.trade-item{flex:1;padding:7px 8px;border-right:1px solid var(--border);}
.trade-item:last-child{border-right:none;}
.trade-item-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:700;margin-bottom:3px;}
.trade-item-val{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--text);}
.trade-meta{display:flex;align-items:center;gap:8px;padding-top:9px;border-top:1px solid var(--border);}
.trade-result{font-size:12px;font-weight:800;}
.trade-score{font-size:10px;color:var(--muted);font-family:var(--mono);background:var(--bg2);padding:2px 6px;border-radius:4px;}
.trade-margin{font-size:10px;color:var(--amber);font-family:var(--mono);margin-left:auto;font-weight:700;}
footer{text-align:center;color:var(--muted2);font-size:10px;padding:20px;}"""
        
        html = (
            "<!DOCTYPE html><html lang='en'><head>"
            "<meta charset='UTF-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
            "<title>WhaleTrader Ultimate</title>"
            "<style>" + mobile_css + "</style>"
            "</head><body>"
            "<nav>"
            "<div class='logo'><div class='logo-icon'>🐋</div>WhaleTrader Ultimate</div>"
            "<div class='nav-right'>" + risk_badge +
            "<div class='badge-live'><div class='pulse-dot'></div>LIVE</div>"
            "</div></nav>"
            "<div class='main'>"
            "<div style='text-align:right;font-size:9px;color:var(--muted);font-family:var(--mono);margin-bottom:10px;'>"
            "Sync: " + now_str + " | <span id='clk'>-- IST</span>"
            "</div>"
            "<div class='stat-grid'>"
            "<div class='stat-card'><div class='stat-label'>Equity</div><div class='stat-value' style='color:#fff;'>$" + str(wallet) + "</div><div class='stat-sub'>Started $1,000</div></div>"
            "<div class='stat-card'><div class='stat-label'>Total P&L</div><div class='stat-value' style='color:" + pnl_color + ";'>" + pnl_prefix + "$" + str(total_pnl) + "</div><div class='stat-sub' style='color:" + pnl_color + ";'>" + pnl_prefix + str(pnl_pct) + "%</div></div>"
            "<div class='stat-card'><div class='stat-label'>Win Rate</div><div class='stat-value' style='color:" + wr_color + ";'>" + str(win_rate) + "%</div><div class='stat-sub'>" + str(wins) + "W / " + str(losses) + "L</div></div>"
            "<div class='stat-card'><div class='stat-label'>Today P&L</div><div class='stat-value' style='color:" + today_color + ";'>" + today_prefix + "$" + str(today_pnl) + "</div><div class='stat-sub'>" + str(today_trades) + " trades</div></div>"
            "<div class='stat-card'><div class='stat-label'>Best Trade</div><div class='stat-value' style='color:var(--green);'>+$" + str(best) + "</div></div>"
            "<div class='stat-card'><div class='stat-label'>Worst Trade</div><div class='stat-value' style='color:var(--red);'>$" + str(worst) + "</div></div>"
            "<div class='stat-card'><div class='stat-label'>Positions</div><div class='stat-value' style='color:var(--blue);'>" + str(open_pos) + "/" + str(MAX_POSITIONS) + "</div></div>"
            "<div class='stat-card'><div class='stat-label'>Leverage</div><div class='stat-value' style='color:var(--amber);'>" + str(LEVERAGE) + "x</div><div class='stat-sub'>4 pairs / 5-min</div></div>"
            "</div>"
            "<div class='sec-head'><div class='sec-title'>Daily P&L</div><div class='sec-line'></div></div>"
            "<div class='chart-card'><canvas id='pnlChart' style='max-height:130px;'></canvas></div>"
            "<div class='sec-head'><div class='sec-title'>Live Monitor</div><div class='sec-line'></div></div>"
            "<div class='monitor-cards'>" + monitor_rows + "</div>"
            "<div class='sec-head'><div class='sec-title'>Settlement Log</div><div class='sec-line'></div></div>"
            "<div class='trade-cards'>" + history_rows + "</div>"
            "</div>"
            "<footer>WhaleTrader Ultimate · 5-min auto-sync · Paper Trading · $1,000 Capital · 7 Strategies</footer>"
            "<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js'></script>"
            "<script>"
            "function updateClock(){var now=new Date();var ist=new Date(now.getTime()+19800000);var h=ist.getUTCHours(),m=ist.getUTCMinutes(),s=ist.getUTCSeconds();var ampm=h>=12?'PM':'AM';var hh=h%12||12;function pad(n){return n<10?'0'+n:''+n;}document.getElementById('clk').textContent=pad(hh)+':'+pad(m)+':'+pad(s)+' '+ampm+' IST';}setInterval(updateClock,1000);updateClock();"
            "var ctx=document.getElementById('pnlChart').getContext('2d');"
            "new Chart(ctx,{type:'bar',data:{labels:" + chart_labels + ",datasets:[{data:" + chart_values + ",backgroundColor:" + chart_colors + ",borderRadius:4}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(26,37,64,.5)'},ticks:{color:'#4a6080',font:{size:9}}},y:{grid:{color:'rgba(26,37,64,.5)'},ticks:{color:'#4a6080',font:{size:9},callback:function(v){return '$'+v;}}}}}});"
            "var symMap={btcusdt:'BTC',ethusdt:'ETH',solusdt:'SOL',bnbusdt:'BNB'};"
            "var prevPrices={};"
            "function updatePrice(sym,price,chg){"
            "  var pe=document.getElementById('p-'+sym);"
            "  var ce=document.getElementById('c-'+sym);"
            "  if(!pe)return;"
            "  var prev=prevPrices[sym]||price;"
            "  var fmt=price<1?price.toFixed(5):price<100?price.toFixed(4):price.toFixed(2);"
            "  pe.textContent='$'+fmt;"
            "  if(price>prev){pe.style.color='var(--green)';}"
            "  else if(price<prev){pe.style.color='var(--red)';}"
            "  else{pe.style.color='var(--text)';}"
            "  setTimeout(function(){pe.style.color='var(--text)';},800);"
            "  if(ce){ce.textContent=(chg>=0?'+':'')+chg.toFixed(2)+'%';ce.style.color=chg>=0?'var(--green)':'var(--red)';}"
            "  prevPrices[sym]=price;"
            "}"
            "function connectWS(){"
            "  var subs=Object.keys(symMap).map(function(s){return '5~CCCAGG~'+symMap[s]+'~USDT';});"
            "  var ws=new WebSocket('wss://streamer.cryptocompare.com/v2?api_key=9ebbc34e695a7e9dfec53ecd9f730def1f8e04e0c7c8b1de3c0bceeca2c12e2a');"
            "  ws.onopen=function(){ws.send(JSON.stringify({action:'SubAdd',subs:subs}));};"
            "  ws.onmessage=function(e){"
            "    var d=JSON.parse(e.data);"
            "    if(d.TYPE!=='5'||!d.PRICE)return;"
            "    var sym=d.FROMSYMBOL.toLowerCase()+'usdt';"
            "    var price=parseFloat(d.PRICE);"
            "    var chg=d.CHANGEPCT24HOUR||0;"
            "    updatePrice(sym,price,chg);"
            "  };"
            "  ws.onclose=function(){setTimeout(connectWS,3000);};"
            "  ws.onerror=function(){ws.close()};"
            "}"
            "connectWS();"
            "</script>"
            "</body></html>"
        )
        
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Dashboard saved — index.html")

if __name__ == "__main__":
    engine = WhaleEngine()
    engine.run()
