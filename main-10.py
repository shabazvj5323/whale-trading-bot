import time
import logging
import os
import json
import ccxt
import numpy as np
import requests
from datetime import datetime, timedelta

class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        from datetime import datetime, timedelta
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
MARGIN_PER_TRADE = 100.0   # Normal BB trade margin
MARGIN_BLAST     = 200.0   # BB Blast = 20% of capital
LEVERAGE         = 10
HISTORY_FILE     = "history.json"

# ── Scalping Config ─────────────────────────
MIN_SCORE   = 4        # More trades
VOL_MULT    = 1.3      # OKX volume spike
BB_PERIOD   = 20
BB_STD      = 2.0
BB_FAST_P   = 10
BB_FAST_S   = 1.5
SR_PERIOD   = 20
SR_THRESH   = 0.003
ATR_PERIOD  = 14

# TP/SL
TP_MULT     = 1.5
SL_MULT     = 1.0
TRAIL_STEP  = 0.15

# Risk management
MAX_POSITIONS   = 6
MAX_DAILY_LOSS  = 80.0
MARGIN_BLAST    = 200.0  # BB Blast margin


# ═══════════════════════════════════════════════
#  HELPERS
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
    """
    Detect market regime: BULL, BEAR, or SIDEWAYS
    Uses EMA50/EMA200 crossover + ADX for trend strength
    """
    closes = closes.astype(float)
    if len(closes) < 50:
        return "SIDEWAYS"

    ema50  = ema_calc(closes, 50)[-1]
    ema200 = ema_calc(closes, min(100, len(closes)-1))[-1]  # 100 reliable with 1-day data

    # ATR based volatility
    atr = atr_calc(highs, lows, closes, 14)
    price = closes[-1]
    atr_pct = (atr / price) * 100  # ATR as % of price

    # Trend direction
    if ema50 > ema200 * 1.002:
        regime = "BULL"
    elif ema50 < ema200 * 0.998:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS"

    return regime, atr_pct


# ═══════════════════════════════════════════════
#  DARVAS BOX STRATEGY — Nicolas Darvas (1960)
#  Core principles: Price Consolidation + Volume Confirmation
# ═══════════════════════════════════════════════
def darvas_box_strategy(opens, highs, lows, closes, vols, box_period=20, vol_period=20):
    """
    Darvas Box Theory implementation.
    Returns: (signal, box_top, box_bottom, trail_sl, reasons)
    signal: 'buy' | 'sell' | None
    """
    if len(closes) < box_period + vol_period:
        return None, None, None, None, []

    closes = closes.astype(float)
    highs  = highs.astype(float)
    lows   = lows.astype(float)
    vols   = vols.astype(float)

    # Step 1: Identify Box (Consolidation Zone) from last N candles
    box_highs = highs[-box_period:]
    box_lows  = lows[-box_period:]

    resistance = float(np.max(box_highs))   # Box Top
    support    = float(np.min(box_lows))    # Box Bottom
    box_range  = resistance - support

    current_close  = float(closes[-1])
    current_volume = float(vols[-1])

    # Step 2: Volume skipped — CoinGecko has no real candle volume
    vol_ma = float(np.mean(vols[-vol_period:]))

    # Step 3: Breakout detection — price action only
    breakout_up   = current_close > resistance
    breakout_down = current_close < support

    # Step 4: Trailing Stop Loss
    # Darvas: After BUY breakout, SL = Box Bottom (support)
    # After SELL breakout, SL = Box Top (resistance)
    trail_sl = None
    signal   = None
    reasons  = []

    if breakout_up:
        signal   = "buy"
        trail_sl = round(support, 6)       # SL at box bottom
        reasons  = [
            "DARVAS BOX",
            "Break: $" + str(round(resistance, 4)),
            "VOL " + str(round(current_volume / vol_ma, 1)) + "x",
        ]
    elif breakout_down:
        signal   = "sell"
        trail_sl = round(resistance, 6)    # SL at box top
        reasons  = [
            "DARVAS BOX",
            "Break: $" + str(round(support, 4)),
            "VOL " + str(round(current_volume / vol_ma, 1)) + "x",
        ]

    return signal, round(resistance, 6), round(support, 6), trail_sl, reasons



# ═══════════════════════════════════════════════
#  ENGINE
# ═══════════════════════════════════════════════
class WhaleEngine:
    def __init__(self):
        self.state     = self.load_history()
        self.dashboard      = []
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
            self.exchange.set_sandbox_mode(False)  # Real market data

    # ── History ──────────────────────────────
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
        """Force close positions older than 6 hours to prevent stuck trades."""
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

    # ── Risk check ───────────────────────────
    def can_trade(self, score=0):
        today = ist_now().strftime("%Y-%m-%d")
        daily_loss = abs(min(self.state["stats"]["daily_pnl"].get(today, 0), 0))
        open_pos   = len(self.state["active_positions"])
        # Daily loss limit always applies
        if daily_loss >= MAX_DAILY_LOSS:
            log.warning(f"Daily loss limit hit: ${daily_loss:.2f} — trading paused")
            return False
        # Score 7+ = high confidence = unlimited positions!
        if score >= 7:
            log.info(f"High confidence {score}/10 — no position limit!")
            return True
        # Score 5-6 = max 3 positions only
        if open_pos >= MAX_POSITIONS:
            log.info(f"Max positions: {open_pos}/{MAX_POSITIONS} (score {score}/10)")
            return False
        return True

    # ── Market data ──────────────────────────
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
        """Build CoinGecko params with API key if available."""
        p = extra.copy() if extra else {}
        if self.CG_API_KEY:
            p["x_cg_demo_api_key"] = self.CG_API_KEY
        return p

    def prefetch_all_volumes(self):
        """Fetch volume for ALL coins in ONE API call using /coins/markets."""
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

    def fetch_coingecko_ohlc(self, symbol, limit=120, vol_map=None):
        coin_id = self.COINGECKO_IDS.get(symbol)
        if not coin_id:
            return None
        for attempt in range(3):  # retry up to 3 times
            try:
                if attempt > 0:
                    time.sleep(10 * attempt)  # 10s, 20s backoff
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
                params = self._cg_params({"vs_currency": "usd", "days": "1"})
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 429:
                    wait = 15 * (attempt + 1)
                    log.warning(f"Rate limit hit {symbol}, waiting {wait}s ({attempt+1}/3)...")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                if not data or len(data) < 10:
                    return None
                arr = np.array(data, dtype=float)
                opens  = arr[:, 1]
                highs  = arr[:, 2]
                lows   = arr[:, 3]
                closes = arr[:, 4]
                base_vol = vol_map.get(coin_id, 1000) if vol_map else 1000
                vols = np.ones(len(closes)) * (base_vol / 1e6)
                closes15 = closes[::3]
                n = min(limit, len(closes))
                return opens[-n:], highs[-n:], lows[-n:], closes[-n:], vols[-n:], closes15
            except Exception as e:
                log.warning(f"CoinGecko OHLC failed {symbol} (attempt {attempt+1}): {e}")
        return None

    def fetch_okx_public(self, symbol, interval="5m", limit=120):
        """OKX public REST API — GitHub Actions accessible, real candle data!"""
        try:
            # OKX symbol format: BTC-USDT instead of BTC/USDT
            clean = symbol.replace("/", "-")
            url = "https://www.okx.com/api/v5/market/candles"
            params = {
                "instId": clean + "-SWAP",
                "bar": interval,
                "limit": str(limit)
            }
            headers = {"OK-ACCESS-KEY": ""}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "0" and data.get("data"):
                    raw = data["data"][::-1]  # newest first — reverse
                    arr = np.array([[float(c[1]), float(c[2]), float(c[3]),
                                     float(c[4]), float(c[5])] for c in raw])
                    return arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
            log.warning(f"OKX public {resp.status_code} for {symbol}")
            return None
        except Exception as e:
            log.warning(f"OKX public failed {symbol}: {e}")
            return None

    def fetch_data(self, symbol, limit=120, vol_map=None):
        # OKX public API — REAL data, GitHub Actions accessible!
        for attempt in range(3):
            result_5m  = self.fetch_okx_public(symbol, "5m",  limit)
            result_15m = self.fetch_okx_public(symbol, "15m", 60)
            if result_5m is not None and result_15m is not None:
                opens, highs, lows, closes, vols = result_5m
                closes15 = result_15m[3]
                log.info(f"OKX Real: {symbol} @ ${round(float(closes[-1]),4)}")
                return opens, highs, lows, closes, vols, closes15
            log.warning(f"OKX retry {attempt+1}/3 for {symbol}")
            time.sleep(2)
        log.error(f"OKX failed after 3 retries: {symbol} — skipping")
        return None

    # ══════════════════════════════════════════
    # PURE BB SCALPING — 4 Strategies from Research
    # ══════════════════════════════════════════
    def score_signal(self, opens, highs, lows, closes, vols, closes15):
        score, direction, reasons = 0, None, []
        is_blast = False
        price  = float(closes[-1])
        # Real RSI for logging
        try:
            from collections import deque
            gains, losses = [], []
            for i in range(1, 15):
                diff = float(closes[-i]) - float(closes[-i-1])
                if diff > 0: gains.append(diff)
                else: losses.append(abs(diff))
            avg_gain = sum(gains)/14 if gains else 0.01
            avg_loss = sum(losses)/14 if losses else 0.01
            rs = avg_gain / avg_loss
            r = round(100 - (100/(1+rs)), 1)
        except:
            r = 50.0
        # Regime for logging
        try:
            regime_result = market_regime(closes, highs, lows)
            regime = regime_result[0] if isinstance(regime_result, tuple) else regime_result
        except:
            regime = "RANGING"

        # Add regime as first reason for log display
        reasons.append(regime)

        # ── BB Standard (20,2) ─────────────────
        upper, mid, lower = bollinger_calc(closes, BB_PERIOD, BB_STD)
        # ── BB Fast (10,1.5) for confirmation ──
        u_fast, m_fast, l_fast = bollinger_calc(closes, BB_FAST_P, BB_FAST_S)

        # ── Volume data ─────────────────────────
        vol_avg = float(np.mean(vols[-15:-1])) if len(vols) > 15 else float(np.mean(vols))
        vol_now = float(vols[-1])
        vol_ratio = vol_now / (vol_avg + 1e-9)
        vol_spike = vol_ratio >= VOL_MULT

        # ── ATR for volatility check ─────────────
        atr = atr_calc(highs, lows, closes, ATR_PERIOD)

        # ════════════════════════════════════════
        # STRATEGY 1: BB BOUNCE (Mean Reversion)
        # Research: Price touches/crosses outer band → reverse to mid
        # Best in ranging/sideways market
        # ════════════════════════════════════════
        body_now  = float(closes[-1]) - float(opens[-1])
        body_prev = float(closes[-2]) - float(opens[-2]) if len(opens) >= 2 else 0
        avg_body  = float(np.mean([abs(float(closes[i])-float(opens[i])) for i in range(-6,-1)]))

        if price <= lower:
            score += 4
            direction = "buy"
            reasons.append("BB BOUNCE LOW")
            # Extra confirmation: reversal candle (green after touching lower)
            if body_now > 0:
                score += 2
                reasons.append("REVERSAL CANDLE")
        elif price >= upper:
            score += 4
            direction = "sell"
            reasons.append("BB BOUNCE HIGH")
            # Extra confirmation: reversal candle (red after touching upper)
            if body_now < 0:
                score += 2
                reasons.append("REVERSAL CANDLE")

        # ════════════════════════════════════════
        # STRATEGY 2: DOUBLE BB ZONE (Pro Strategy)
        # Research: Fast BB(10,1.5) + Slow BB(20,2)
        # Buy Zone: price between lower bands (strong downtrend reversal)
        # Sell Zone: price between upper bands (strong uptrend reversal)
        # ════════════════════════════════════════
        if direction is None:
            in_buy_zone  = l_fast <= price <= lower   # between fast lower & slow lower
            in_sell_zone = upper <= price <= u_fast   # between slow upper & fast upper
            if in_buy_zone:
                score += 4
                direction = "buy"
                reasons.append("DOUBLE BB BUY ZONE")
            elif in_sell_zone:
                score += 4
                direction = "sell"
                reasons.append("DOUBLE BB SELL ZONE")

        # ════════════════════════════════════════
        # STRATEGY 3: BB BLAST (Squeeze Breakout)
        # Research: Bands narrow (squeeze) → big candle breaks out → momentum trade
        # 65-70% accuracy per research — use 50% margin!
        # ════════════════════════════════════════
        # Detect squeeze: current band width < 80% of average band width last 20 candles
        band_widths = []
        for i in range(-20, -1):
            try:
                u_i, m_i, l_i = bollinger_calc(closes[:i], BB_PERIOD, BB_STD)
                band_widths.append(u_i - l_i)
            except:
                pass
        avg_band_width = float(np.mean(band_widths)) if band_widths else (upper - lower)
        band_width_now = upper - lower
        squeeze = band_width_now < avg_band_width * 0.75  # bands 25% narrower than avg

        # Big candle = blast signal
        big_candle = abs(body_now) > avg_body * 2.0

        if squeeze and big_candle:
            if body_now > 0 and price > mid:  # Bullish blast
                score += 6
                direction = "buy"
                is_blast = True
                reasons.append("BB BLAST UP 🚀")
            elif body_now < 0 and price < mid:  # Bearish blast
                score += 6
                direction = "sell"
                is_blast = True
                reasons.append("BB BLAST DN 🚀")

        # ════════════════════════════════════════
        # STRATEGY 4: S/R CONSOLIDATION BREAKOUT
        # Research: Find swing high/low (consolidation zone)
        # Breakout above resistance = BUY, below support = SELL
        # Use trend direction for confirmation
        # ════════════════════════════════════════
        if direction is None and len(highs) >= SR_PERIOD:
            # Find recent S/R levels
            resistance = float(np.max(highs[-SR_PERIOD:]))
            support    = float(np.min(lows[-SR_PERIOD:]))
            # Breakout confirmation: price closes beyond S/R
            broke_resistance = price > resistance * (1 + SR_THRESH)
            broke_support    = price < support    * (1 - SR_THRESH)
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
        # STRATEGY 6: VWAP SCALPING
        # Price vs VWAP — mean reversion + trend follow
        # ════════════════════════════════════════
        try:
            typical = (highs + lows + closes) / 3
            vwap = float(np.sum(typical * vols) / (np.sum(vols) + 1e-9))
            if price < vwap * 0.998:
                score += 3
                direction = direction or "buy"
                reasons.append("VWAP BUY")
            elif price > vwap * 1.002:
                score += 3
                direction = direction or "sell"
                reasons.append("VWAP SELL")
        except:
            pass

        # ════════════════════════════════════════
        # STRATEGY 7: EMA RIBBON
        # EMA 9/21/50 alignment = strong trend
        # ════════════════════════════════════════
        try:
            if len(closes) >= 50:
                e9  = ema_calc(closes, 9)[-1]
                e21 = ema_calc(closes, 21)[-1]
                e50 = ema_calc(closes, 50)[-1]
                if e9 > e21 > e50:
                    score += 2
                    direction = direction or "buy"
                    reasons.append("EMA RIBBON BULL")
                elif e9 < e21 < e50:
                    score += 2
                    direction = direction or "sell"
                    reasons.append("EMA RIBBON BEAR")
        except:
            pass

        # ════════════════════════════════════════
        # STRATEGY 8: ORDER FLOW IMBALANCE
        # Buy/Sell pressure from last 10 candles
        # ════════════════════════════════════════
        try:
            buy_pressure  = 0.0
            sell_pressure = 0.0
            for i in range(-10, 0):
                body = float(closes[i]) - float(opens[i])
                vol  = float(vols[i])
                if body > 0:
                    buy_pressure  += abs(body) * vol
                else:
                    sell_pressure += abs(body) * vol
            total_pressure = buy_pressure + sell_pressure + 1e-9
            buy_ratio = buy_pressure / total_pressure
            if buy_ratio > 0.65:
                score += 2
                direction = direction or "buy"
                reasons.append("ORDER FLOW BUY")
            elif buy_ratio < 0.35:
                score += 2
                direction = direction or "sell"
                reasons.append("ORDER FLOW SELL")
        except:
            pass

        # ── Volume bonus for any strategy ────────
        if direction and vol_spike and "VOL" not in str(reasons):
            score += 2
            reasons.append("VOL " + str(round(vol_ratio, 1)) + "x")

        # Store blast flag for margin calculation
        self._last_is_blast = is_blast
        adaptive_min = MIN_SCORE

        return score, direction, r, upper, lower, reasons, adaptive_min

    # ── Position management ──────────────────
    def check_positions(self, symbol, current_price):
        """
        Real trading behavior:
        - Sirf CURRENT price check karo (last candle ka close)
        - Puri history scan nahi — bilkul real exchange jaisa!
        - Har 15 min mein ek price check hoga
        """
        if symbol not in self.state["active_positions"]:
            return

        pos    = self.state["active_positions"][symbol]
        side   = pos["side"]
        entry  = float(pos["entry"])
        tp     = float(pos["tp"])
        sl     = float(pos["sl"])
        margin = float(pos.get("margin", MARGIN_PER_TRADE))
        qty    = (margin * LEVERAGE) / entry
        price  = float(current_price)  # sirf current price!

        sl_dist    = abs(entry - sl)
        trail_dist = sl_dist * TRAIL_STEP
        # BB mid as natural exit target
        try:
            _, bb_mid, _ = bollinger_calc(closes_ref if hasattr(self, "closes_ref") else np.array([entry]*25), BB_PERIOD, BB_STD)
        except:
            bb_mid = entry

        profit = (price - entry) if side == "buy" else (entry - price)

        # Trailing SL — BB middle band follow karo (sirf profit zone mein)
        # Recalculate BB middle (SMA20) for current symbol
        last_prices_arr = list(self.state.get("last_prices_history", {}).get(symbol, [entry] * 20))
        last_prices_arr.append(price)
        last_prices_arr = last_prices_arr[-20:]
        bb_mid = round(sum(last_prices_arr) / len(last_prices_arr), 6)

        if side == "buy":
            if price > tp:
                tp = round(price + trail_dist, 6)
                self.state["active_positions"][symbol]["tp"] = tp
            # Trail SL = BB middle (only in profit)
            if profit > sl_dist:
                new_sl = round(bb_mid - trail_dist, 6)
                if new_sl > sl:
                    sl = new_sl
                    self.state["active_positions"][symbol]["sl"] = sl
        else:
            if price < tp:
                tp = round(price - trail_dist, 6)
                self.state["active_positions"][symbol]["tp"] = tp
            # Trail SL = BB middle (only in profit)
            if profit > sl_dist:
                new_sl = round(bb_mid + trail_dist, 6)
                if new_sl < sl:
                    sl = new_sl
                    self.state["active_positions"][symbol]["sl"] = sl

        # SL/TP check on current price only
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
                pnl = (exit_p - entry) * qty
            else:
                pnl = (entry - exit_p) * qty

            pnl = round(pnl, 4)  # Real PnL — no artificial cap

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

            # Exit time = actual current IST time (real trade jaisa!)
            exit_time_str = ist_short()

            self.state["trades"].append({
                "time":   exit_time_str,
                "symbol": symbol,
                "side":   side.upper(),
                "entry":  round(entry, 6),
                "exit":   round(exit_p, 6),
                "pnl":    pnl,
                "result": reason,
                "score":  pos.get("score", 0),
                "margin": margin,
            })
            del self.state["active_positions"][symbol]
            icon = "PROFIT" if pnl > 0 else "LOSS"
            log.info(f"{icon} {symbol} {side.upper()} | Entry:${entry} Exit:${exit_p} PnL:${pnl} | {reason}")
            self.save_history()

    # ── Main run ─────────────────────────────
    def run(self):
        log.info("WhaleTrader Pro — Cycle Start")
        self.cleanup_stale_positions()

        # Prefetch all volumes in ONE call to save API quota
        vol_map = self.prefetch_all_volumes()
        log.info(f"Volume prefetch: {len(vol_map)} coins loaded")

        # Rotate symbol order so all pairs get equal priority over time
        import random
        symbols_this_run = SYMBOLS.copy()
        random.shuffle(symbols_this_run)

        for symbol in symbols_this_run:
            try:
                time.sleep(8)  # Avoid CoinGecko rate limit
                result = self.fetch_data(symbol, vol_map=vol_map)
                if result is None:
                    last_price = self.state["last_prices"].get(symbol, 0)
                    self.dashboard.append({
                        "symbol": symbol, "price": last_price,
                        "signal": "SCANNING", "score": 0,
                        "entry": None, "tp": None, "sl": None,
                        "reasons": ["Data unavailable"],
                    })
                    continue
                opens, highs, lows, closes, vols, closes15 = result

                # Check existing position first
                # Real trading: sirf current price check karo!
                current_price = float(closes[-1])
                self.check_positions(symbol, current_price)

                price = round(float(closes[-1]), 6)
                self.state["last_prices"][symbol] = price
                # Save price history for BB middle calculation in trailing
                if "last_prices_history" not in self.state:
                    self.state["last_prices_history"] = {}
                hist = self.state["last_prices_history"].get(symbol, [])
                hist.append(price)
                self.state["last_prices_history"][symbol] = hist[-20:]

                # If still holding after check — skip signal scoring, go to dashboard
                if symbol in self.state["active_positions"]:
                    pos = self.state["active_positions"][symbol]
                    self.dashboard.append({
                        "symbol": symbol, "price": price,
                        "signal": "HOLDING " + pos["side"].upper(),
                        "score":  pos.get("score", 0),
                        "entry":  pos["entry"],
                        "tp":     pos["tp"],
                        "sl":     pos["sl"],
                        "reasons": pos.get("reasons", []),
                    })
                    continue  # skip new signal scoring while in trade

                # Score signal for new entry
                score, direction, r, upper, lower, reasons, adaptive_min = self.score_signal(
                    opens, highs, lows, closes, vols, closes15)

                log.info(f"{symbol} | ${price} | RSI:{round(r,1)} | Score:{score}/10 | Regime:{reasons[0] if reasons else '?'} | Min:{adaptive_min}")

                # ── Darvas Box Strategy (additional layer) ──
                darvas_sig, box_top, box_bot, darvas_sl, darvas_reasons = darvas_box_strategy(
                    opens, highs, lows, closes, vols)

                # Boost score if Darvas confirms same direction
                if darvas_sig and darvas_sig == direction:
                    score = min(score + 2, 10)
                    reasons = reasons + darvas_reasons
                    log.info(f"DARVAS CONFIRM {symbol} {darvas_sig.upper()} | Box: {box_bot}-{box_top}")
                # Darvas standalone signal if main score too low
                elif darvas_sig and not direction and darvas_sl:
                    direction = darvas_sig
                    score     = max(score, 5)
                    reasons   = darvas_reasons
                    log.info(f"DARVAS SIGNAL {symbol} {darvas_sig.upper()} | Box: {box_bot}-{box_top}")

                # Enter trade if conditions met
                if direction and score >= adaptive_min and self.can_trade(score):
                    # BB Blast = 50% margin ($500), normal = $100
                    is_blast = getattr(self, "_last_is_blast", False)
                    margin = MARGIN_BLAST if is_blast else MARGIN_PER_TRADE
                    av = atr_calc(highs, lows, closes, ATR_PERIOD)

                    if direction == "buy":
                        tp = round(price + av * TP_MULT, 6)
                        # SL = Swing Low with proper ATR buffer
                        swing_low = float(np.min(lows[-10:]))
                        sl = round(min(swing_low - av * 0.5, price - av * SL_MULT), 6)
                    else:
                        tp = round(price - av * TP_MULT, 6)
                        # SL = Swing High with proper ATR buffer
                        swing_high = float(np.max(highs[-10:]))
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

                # Dashboard entry — show new/existing position or scanning
                if symbol in self.state["active_positions"]:
                    pos = self.state["active_positions"][symbol]
                    self.dashboard.append({
                        "symbol": symbol, "price": price,
                        "signal": "HOLDING " + pos["side"].upper(),
                        "score":  pos.get("score", score),
                        "entry":  pos["entry"],
                        "tp":     pos["tp"],
                        "sl":     pos["sl"],
                        "margin": pos.get("margin", MARGIN_PER_TRADE),
                        "reasons": pos.get("reasons", reasons),
                    })
                else:
                    self.dashboard.append({
                        "symbol":  symbol,
                        "price":   price,
                        "signal":  direction.upper() + " " + str(score) + "/10" if direction else "SCANNING",
                        "score":   score,
                        "entry":   None,
                        "tp":      None,
                        "sl":      None,
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

    # ── Dashboard ────────────────────────────
    def build_dashboard(self):
        state  = self.state
        stats  = state.get("stats", {})
        trades = state.get("trades", [])

        total_pnl  = round(state.get("total_pnl", 0.0), 2)
        wallet     = round(INITIAL_CAPITAL + total_pnl, 2)
        total_t    = stats.get("total_trades", 0)
        wins       = stats.get("wins", 0)
        losses     = stats.get("losses", 0)
        win_rate   = round(wins / total_t * 100, 1) if total_t > 0 else 0.0
        best       = round(stats.get("best_trade", 0.0), 2)
        worst      = round(stats.get("worst_trade", 0.0), 2)
        pnl_pct    = round((total_pnl / INITIAL_CAPITAL) * 100, 2)
        pnl_color  = "#00e676" if total_pnl >= 0 else "#ff1744"
        pnl_prefix = "+" if total_pnl >= 0 else ""
        wr_color   = "#00e676" if win_rate >= 55 else "#ffab00" if win_rate >= 45 else "#ff1744"
        now_str    = ist_str()

        today       = ist_now().strftime("%Y-%m-%d")
        today_pnl   = round(stats.get("daily_pnl", {}).get(today, 0.0), 2)
        today_trades= stats.get("daily_trades", {}).get(today, 0)
        today_color = "#00e676" if today_pnl >= 0 else "#ff1744"
        today_prefix= "+" if today_pnl >= 0 else ""

        open_pos    = len(state.get("active_positions", {}))
        daily_loss  = abs(min(stats.get("daily_pnl", {}).get(today, 0), 0))
        trading_ok  = daily_loss < MAX_DAILY_LOSS and open_pos < MAX_POSITIONS

        daily_pnl_data = stats.get("daily_pnl", {})
        sorted_days    = sorted(daily_pnl_data.items())[-14:]
        chart_labels   = json.dumps([d[0][5:] for d in sorted_days])
        chart_values   = json.dumps([round(d[1], 2) for d in sorted_days])
        chart_colors   = json.dumps([
            "rgba(0,230,118,0.85)" if d[1] >= 0 else "rgba(255,23,68,0.85)"
            for d in sorted_days
        ])

        # Monitor rows
        monitor_rows = ""
        for d in self.dashboard:
            sym   = d.get("symbol", "")
            clean = sym.replace("/", "").lower()
            sig   = d.get("signal", "SCANNING")
            score = int(d.get("score", 0))
            reasons = ", ".join(d.get("reasons", [])) or "Waiting..."
            tp_val    = "$" + str(d["tp"]) if d.get("tp") else "-"
            sl_val    = "$" + str(d["sl"]) if d.get("sl") else "-"
            entry_val = "$" + str(d["entry"]) if d.get("entry") and "HOLD" in sig else "-"

            if "HOLD" in sig:   pill = "<span class='pill-hold'>" + sig + "</span>"
            elif "BUY" in sig:  pill = "<span class='pill-buy'>" + sig + "</span>"
            elif "SELL" in sig: pill = "<span class='pill-sell'>" + sig + "</span>"
            else:               pill = "<span class='pill-scan'>SCANNING</span>"

            dots = ""
            for i in range(10):
                dots += "<span class='dotf'></span>" if i < score else "<span class='dot'></span>"

            okx_sym = sym.replace("/", "-").lower() + "-swap"
            okx_url = "https://www.okx.com/trade-swap/" + okx_sym
            monitor_rows += (
                "<div class='monitor-card'>"
                "<div class='monitor-top'>"
                "<div class='monitor-sym'>" + sym + "</div>" "<div style='font-size:9px;color:var(--muted);margin-top:1px;'>OKX ↗</div>"
                "<span id='p-" + clean + "' class='monitor-price'>$" + str(d.get('price','--')) + "</span>"
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

        # Trade history
        history_rows = ""
        for t in reversed(trades[-50:]):
            pv   = float(t.get("pnl", 0))
            pc   = "pos" if pv >= 0 else "neg"
            icon = "TP" if t.get("result", "") == "TP" else "SL"
            badge= "pill-buy" if t.get("side","") == "BUY" else "pill-sell"
            prefix = "+" if pv >= 0 else ""
            history_rows += (
                "<div class='trade-card'>"
                "<div class='trade-top'>"
                "<div style='display:flex;align-items:center;gap:8px;'>"
                "<div class='trade-pair'>" + str(t.get("symbol","")) + "</div>"
                "<span class='" + badge + "'>" + str(t.get("side","")) + "</span>"
                "</div>"
                "<div class='trade-pnl " + pc + "'>" + prefix + "$" + str(round(pv,2)) + "</div>"
                "</div>"
                "<div class='trade-row'>"
                "<div class='trade-item'><div class='trade-item-label'>Time</div><div class='trade-item-val'>" + str(t.get("time","")) + "</div></div>"
                "<div class='trade-item'><div class='trade-item-label'>Entry</div><div class='trade-item-val'>$" + str(t.get("entry","")) + "</div></div>"
                "<div class='trade-item'><div class='trade-item-label'>Exit</div><div class='trade-item-val'>$" + str(t.get("exit","")) + "</div></div>"
                "</div>"
                "<div class='trade-meta'>"
                "<span class='trade-result'>" + ("🎯 TP" if icon == "TP" else "🛑 SL") + "</span>"
                "<span class='trade-score'>Score: " + str(t.get("score","-")) + "/10</span>"
                "<span class='trade-margin' style='font-size:10px;color:var(--amber);font-family:var(--mono);margin-left:auto;font-weight:700;background:linear-gradient(135deg,rgba(255,165,2,0.15),rgba(255,109,0,0.15));padding:2px 8px;border-radius:6px;border:1px solid rgba(255,165,2,0.3);'>Margin: $" + str(t.get("margin",100)) + "</span>"
                "</div>"
                "</div>"
            )

        if not history_rows:
            history_rows = "<div class='trade-card' style='text-align:center;color:var(--muted);'>No trades yet — bot is scanning...</div>"

        # ── Professional Binance-style Dashboard ──
        risk_badge_text = "TRADING ACTIVE" if trading_ok else "LIMIT REACHED"
        risk_badge_cls  = "badge-ok" if trading_ok else "badge-limit"

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

/* ── NAV ── */
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

/* ── TABS ── */
.tabs{display:flex;background:var(--bg1);border-bottom:1px solid var(--line);overflow-x:auto;-webkit-overflow-scrolling:touch;position:sticky;top:52px;z-index:99;}
.tabs::-webkit-scrollbar{display:none;}
.tab{flex:none;padding:12px 16px;font-size:12px;font-weight:600;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:all .2s;}
.tab.active{color:var(--yellow);border-bottom-color:var(--yellow);}

/* ── CONTENT PANELS ── */
.panel{display:none;padding:12px;}
.panel.active{display:block;}

/* ── OVERVIEW STATS ── */
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

/* ── CHART ── */
.chart-box{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:8px;}
.chart-box-title{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px;}

/* ── MARKET / MONITOR ── */
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

/* ── OPEN POSITIONS ── */
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

/* ── SETTLEMENT LOG ── */
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
"""

        # ── Monitor cards ──────────────────────
        monitor_html = ""
        for d in self.dashboard:
            sym   = d.get("symbol","")
            clean = sym.replace("/","").lower()
            sig   = d.get("signal","SCANNING")
            score = int(d.get("score",0))
            reasons_str = ", ".join(d.get("reasons",[])) or "Waiting..."
            tp_val    = "$"+str(d["tp"])   if d.get("tp")    else "-"
            sl_val    = "$"+str(d["sl"])   if d.get("sl")    else "-"
            entry_val = "$"+str(d["entry"]) if d.get("entry") and "HOLD" in sig else "-"
            margin_val= str(d.get("margin", MARGIN_PER_TRADE))

            if "HOLD" in sig:
                if "BUY" in sig:  pill = "<span class='signal-pill sp-hold'>HOLDING BUY</span>"
                else:             pill = "<span class='signal-pill sp-hold'>HOLDING SELL</span>"
            elif "BUY" in sig:    pill = "<span class='signal-pill sp-buy'>" + sig + "</span>"
            elif "SELL" in sig:   pill = "<span class='signal-pill sp-sell'>" + sig + "</span>"
            else:                 pill = "<span class='signal-pill sp-scan'>SCANNING</span>"

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
                "<div class='market-card' style='cursor:pointer;' data-sym='" + sym.replace("/","") + "' onclick='openChart(this.dataset.sym)'>"

                "<div class='market-top'>"
                "<span class='market-pair'>" + sym + "</span>"
                "<span class='market-price' id='p-" + clean + "'>$" + str(d.get("price","--")) + "</span>"
                "</div>"
                "<div class='market-row'>"
                "<span class='market-chg up' id='c-" + clean + "'>--</span>"
                + pill +
                "<div class='score-dots'>" + dots + "</div>"
                "</div>"
                + pos_section +
                "</div>"
            )

        # ── Open Positions panel ─────────────
        open_positions_html = ""
        for d in self.dashboard:
            sig = d.get("signal","")
            if "HOLD" not in sig:
                continue
            sym   = d.get("symbol","")
            clean = sym.replace("/","").lower()
            side  = "BUY" if "BUY" in sig else "SELL"
            side_cls = "buy" if side == "BUY" else "sell"
            side_pill = "<span class='pos-side side-buy'>BUY</span>" if side=="BUY" else "<span class='pos-side side-sell'>SELL</span>"
            entry_val = "$"+str(d.get("entry","--"))
            tp_val    = "$"+str(d["tp"]) if d.get("tp") else "-"
            sl_val    = "$"+str(d["sl"]) if d.get("sl") else "-"
            margin_val= str(d.get("margin", MARGIN_PER_TRADE))
            score     = str(d.get("score",0))
            reasons_str = ", ".join(d.get("reasons",[])) or ""
            open_positions_html += (
                "<div class='pos-card " + side_cls + "'>"
                "<div class='pos-card-top'>"
                "<div style='display:flex;align-items:center;gap:8px;'>"
                "<span class='pos-pair'>" + sym + "</span>"
                + side_pill +
                "</div>"
                "<span class='pos-live-price' id='pp-" + clean + "'>$" + str(d.get("price","--")) + "</span>"
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

        # ── Settlement log ───────────────────
        log_html = ""
        for t in reversed(trades[-50:]):
            pv     = float(t.get("pnl",0))
            is_win = pv >= 0
            result = t.get("result","SL")
            side   = str(t.get("side","")).upper()
            side_pill = "<span class='log-side side-buy'>BUY</span>" if side=="BUY" else "<span class='log-side side-sell'>SELL</span>"
            result_span = "<span class='log-result win'>🎯 TP</span>" if result=="TP" else "<span class='log-result loss'>🛑 SL</span>"
            log_html += (
                "<div class='log-card " + ("win" if is_win else "loss") + "'>"
                "<div class='log-top'>"
                "<div class='log-pair-row'>"
                "<span class='log-pair'>" + str(t.get("symbol","")) + "</span>"
                + side_pill +
                "</div>"
                "<span class='log-pnl " + ("win" if is_win else "loss") + "'>" + ("+" if is_win else "") + "$" + str(round(pv,2)) + "</span>"
                "</div>"
                "<div class='log-grid'>"
                "<div class='log-cell'><div class='log-cell-label'>Entry</div><div class='log-cell-val'>$" + str(t.get("entry","")) + "</div></div>"
                "<div class='log-cell'><div class='log-cell-label'>Exit</div><div class='log-cell-val'>$" + str(t.get("exit","")) + "</div></div>"
                "<div class='log-cell'><div class='log-cell-label'>Time</div><div class='log-cell-val'>" + str(t.get("time","")) + "</div></div>"
                "</div>"
                "<div class='log-footer'>"
                + result_span +
                "<span class='log-score'>Score: " + str(t.get("score","-")) + "/10</span>"
                "<span class='log-margin'>💰 $" + str(t.get("margin",100)) + "</span>"
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
            # Nav
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
            # Tabs
            "<div class='tabs'>"
            "<div class='tab active' onclick='showTab(0)'>📊 Overview</div>"
            "<div class='tab' onclick='showTab(1)'>📈 Markets</div>"
            "<div class='tab' onclick='showTab(2)'>💼 Positions (" + str(open_pos) + ")</div>"
            "<div class='tab' onclick='showTab(3)'>📋 History</div>"
            "</div>"

            # Tab 0: Overview
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
            "<div class='stat-box'><div class='stat-box-label'>Win Rate</div><div class='stat-box-val " + ("green" if win_rate>=55 else "yellow" if win_rate>=45 else "red") + "'>" + str(win_rate) + "%</div><div class='stat-box-sub'>" + str(wins) + "W / " + str(losses) + "L / " + str(total_t) + " total</div></div>"
            "<div class='stat-box'><div class='stat-box-label'>Today P&amp;L</div><div class='stat-box-val " + ("green" if today_pnl>=0 else "red") + "'>" + ("+" if today_pnl>=0 else "") + "$" + str(today_pnl) + "</div><div class='stat-box-sub'>" + str(today_trades) + " trades today</div></div>"
            "</div>"
            "<div class='stats-row'>"
            "<div class='stat-box'><div class='stat-box-label'>Best Trade</div><div class='stat-box-val green'>+$" + str(best) + "</div><div class='stat-box-sub'>All time high</div></div>"
            "<div class='stat-box'><div class='stat-box-label'>Worst Trade</div><div class='stat-box-val red'>$" + str(worst) + "</div><div class='stat-box-sub'>Max drawdown</div></div>"
            "</div>"
            "<div class='stats-row'>"
            "<div class='stat-box'><div class='stat-box-label'>Positions</div><div class='stat-box-val yellow'>" + str(open_pos) + "/" + str(MAX_POSITIONS) + "</div><div class='stat-box-sub'>Max allowed</div></div>"
            "<div class='stat-box'><div class='stat-box-label'>Strategies</div><div class='stat-box-val yellow'>8</div><div class='stat-box-sub'>Active / 10x Leverage</div></div>"
            "</div>"
            "<div class='chart-box'>"
            "<div class='chart-box-title'>Daily P&amp;L — Last 14 Days</div>"
            "<canvas id='pnlChart' style='max-height:140px;'></canvas>"
            "</div>"
            "</div>"

            # Tab 1: Markets
            "<div class='panel' id='tab1'>"
            + monitor_html +
            "</div>"

            # Tab 2: Open Positions
            "<div class='panel' id='tab2'>"
            + open_positions_html +
            "</div>"

            # Tab 3: Settlement Log
            "<div class='panel' id='tab3'>"
            + log_html +
            "</div>"

            "<footer>WhaleTrader Ultimate &middot; 15 min auto-sync &middot; Paper Trading &middot; $1,000 Capital &middot; 8 Strategies</footer>"

            "<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js'></script>"
            "<script>"
            # Clock
            "function updateClock(){var n=new Date();var ist=new Date(n.getTime()+19800000);var h=ist.getUTCHours(),m=ist.getUTCMinutes(),s=ist.getUTCSeconds();var ap=h>=12?'PM':'AM';var hh=h%12||12;function p(x){return x<10?'0'+x:''+x;}document.getElementById('clk').textContent=p(hh)+':'+p(m)+':'+p(s)+' '+ap+' IST';}setInterval(updateClock,1000);updateClock();"
            # Tabs
            "function showTab(i){document.querySelectorAll('.panel').forEach(function(p,j){p.classList.toggle('active',i===j);});document.querySelectorAll('.tab').forEach(function(t,j){t.classList.toggle('active',i===j);});window._activeTab=i;}""function openChart(sym){var map={'BTCUSDT':'BTC','ETHUSDT':'ETH','SOLUSDT':'SOL','BNBUSDT':'BNB','XRPUSDT':'XRP','DOGEUSDT':'DOGE'};var base=map[sym]||sym.replace('USDT','');window.open('https://www.tradingview.com/chart/?symbol=OKX:'+base+'USDT&interval=5','_blank');}""(function(){var sx=0,sy=0;document.addEventListener('touchstart',function(e){sx=e.touches[0].clientX;sy=e.touches[0].clientY;},{passive:true});document.addEventListener('touchend',function(e){var dx=e.changedTouches[0].clientX-sx;var dy=e.changedTouches[0].clientY-sy;if(Math.abs(dx)>Math.abs(dy)&&Math.abs(dx)>50){var cur=window._activeTab||0;if(dx<0&&cur<3)showTab(cur+1);else if(dx>0&&cur>0)showTab(cur-1);}},{passive:true});})();"
            # Chart
            "var ctx=document.getElementById('pnlChart').getContext('2d');"
            "new Chart(ctx,{type:'bar',data:{labels:" + chart_labels + ",datasets:[{data:" + chart_values + ",backgroundColor:" + chart_colors + ",borderRadius:4,borderSkipped:false}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#848e9c',font:{size:9}}},y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#848e9c',font:{size:9},callback:function(v){return'$'+v;}}}}}});"
            # WebSocket live prices
            "var syms=['btcusdt','ethusdt','solusdt','bnbusdt','xrpusdt','dogeusdt'];"
            "function connectWS(){var ws=new WebSocket('wss://stream.binance.com:9443/ws/'+syms.map(function(s){return s+'@ticker';}).join('/'));"
            "ws.onmessage=function(e){var d=JSON.parse(e.data);var s=d.s.toLowerCase();var pe=document.getElementById('p-'+s);var pp=document.getElementById('pp-'+s);var ce=document.getElementById('c-'+s);if(!pe)return;var price=parseFloat(d.c);var chg=parseFloat(d.P);var priceStr=price<1?'$'+price.toFixed(5):'$'+price.toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2});pe.textContent=priceStr;if(pp)pp.textContent=priceStr;pe.style.color=chg>=0?'#0ecb81':'#f6465d';setTimeout(function(){pe.style.color='';},500);if(ce){ce.textContent=(chg>=0?'+':'')+chg.toFixed(2)+'%';ce.className='market-chg '+(chg>=0?'up':'dn');}};"
            "ws.onclose=function(){setTimeout(connectWS,3000);};}connectWS();"
            "</script>"
            "</body></html>"
        )

        # Add unique timestamp comment so git always detects change
        import time as _time
        html = html + "<!-- sync:" + str(int(_time.time())) + " -->"
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Dashboard saved — index.html")

if __name__ == "__main__":
    engine = WhaleEngine()
    engine.run()
