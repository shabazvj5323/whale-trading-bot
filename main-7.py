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

            monitor_rows += (
                "<div class='monitor-card'>"
                "<div class='monitor-top'>"
                "<div class='monitor-sym'>" + sym + "</div>"
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

        # ── Mobile-first HTML using mobile_preview.html design ──
        risk_badge = "<span class='badge-status status-ok'>TRADING ACTIVE</span>" if trading_ok else "<span class='badge-status status-limit'>LIMIT REACHED</span>"

        # Mobile CSS from mobile_preview.html design
        mobile_css = """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
:root{--bg0:#0a0e27;--bg1:#151b3d;--bg2:#1e2558;--bg3:#2a3268;--border:#3d4585;--border2:#4d56a0;--text:#ffffff;--muted:#8b92d9;--muted2:#6b72b9;--green:#00ff88;--green2:#00e676;--red:#ff4757;--red2:#ff1744;--blue:#3742fa;--amber:#ffa502;--purple:#a55eea;--cyan:#00e5ff;--font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;}
*{box-sizing:border-box;margin:0;padding:0;}
html{scroll-behavior:smooth;}
body{background:linear-gradient(135deg,#0a0e27 0%,#1a1f4d 100%);color:var(--text);font-family:var(--font);font-size:13px;min-height:100vh;}
::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-track{background:var(--bg0);}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;}
nav{background:rgba(3,5,10,0.98);backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);border-bottom:1px solid var(--border);padding:0 16px;height:54px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;}
.logo{display:flex;align-items:center;gap:9px;font-size:15px;font-weight:900;color:#fff;letter-spacing:-.3px;}
.logo-icon{width:30px;height:30px;border-radius:8px;background:linear-gradient(135deg,#1a56ff,#00e676);display:flex;align-items:center;justify-content:center;font-size:15px;box-shadow:0 0 12px rgba(0,230,118,.2);}
.badge-live{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.3);font-size:10px;font-weight:800;padding:3px 9px;border-radius:20px;display:flex;align-items:center;gap:5px;letter-spacing:.05em;}
.pulse-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;box-shadow:0 0 6px var(--green);}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(0,230,118,.6);}70%{box-shadow:0 0 0 6px rgba(0,230,118,0);}100%{box-shadow:0 0 0 0 rgba(0,230,118,0);}}
.nav-right{display:flex;align-items:center;gap:6px;flex-wrap:wrap;}
.nav-box{font-family:var(--mono);font-size:9px;color:var(--muted);background:var(--bg2);border:1px solid var(--border);padding:4px 8px;border-radius:6px;white-space:nowrap;}
.badge-status{font-size:9px;font-weight:800;padding:4px 9px;border-radius:20px;white-space:nowrap;letter-spacing:.05em;}
.status-ok{background:rgba(0,230,118,.12);color:var(--green);border:1px solid rgba(0,230,118,.3);}
.status-limit{background:rgba(255,23,68,.12);color:var(--red);border:1px solid rgba(255,23,68,.3);}
.main{padding:14px 12px;max-width:640px;margin:0 auto;}
.sync-row{text-align:right;font-size:9px;color:var(--muted);font-family:var(--mono);margin-bottom:12px;letter-spacing:.03em;}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:18px;}
.stat-card{background:linear-gradient(135deg,var(--bg1),var(--bg2));border:2px solid var(--border);border-radius:12px;padding:13px;position:relative;overflow:hidden;transition:border-color .2s;box-shadow:0 4px 15px rgba(0,0,0,.3);}
.stat-card:active{border-color:var(--border2);}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.c1::before{background:linear-gradient(90deg,#1a56ff,var(--purple));}
.c2::before{background:linear-gradient(90deg,var(--green),var(--green2));}
.c3::before{background:linear-gradient(90deg,var(--amber),#ff6d00);}
.c4::before{background:linear-gradient(90deg,var(--cyan),#1a56ff);}
.c5::before{background:linear-gradient(90deg,var(--red),#ff6d00);}
.c6::before{background:linear-gradient(90deg,var(--purple),#1a56ff);}
.c7::before{background:linear-gradient(90deg,var(--cyan),var(--green));}
.c8::before{background:linear-gradient(90deg,#ff6d00,var(--amber));}
.stat-label{font-size:9px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:700;margin-bottom:6px;}
.stat-value{font-size:20px;font-weight:900;font-family:var(--mono);line-height:1;letter-spacing:-.5px;}
.stat-sub{font-size:10px;color:var(--muted);margin-top:5px;font-weight:500;}
.mini-bar{background:var(--bg3);border-radius:3px;height:2px;margin-top:7px;}
.mini-fill{height:2px;border-radius:3px;}
.sec-head{display:flex;align-items:center;gap:8px;margin:18px 0 10px;}
.sec-title{font-size:9px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:800;white-space:nowrap;}
.sec-line{flex:1;height:1px;background:var(--border);}
.sec-tag{font-size:9px;color:var(--muted);background:var(--bg2);border:1px solid var(--border);padding:2px 7px;border-radius:8px;font-weight:600;}
.chart-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:14px;margin-bottom:4px;}
.chart-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}
.chart-title{font-size:13px;font-weight:800;color:#fff;}
.chart-sub{font-size:10px;color:var(--muted);}
.monitor-cards{display:flex;flex-direction:column;gap:9px;}
.monitor-card{background:linear-gradient(135deg,var(--bg1),var(--bg2));border:2px solid var(--border);border-radius:12px;padding:13px;transition:border-color .2s;box-shadow:0 4px 15px rgba(0,0,0,.3);}
.monitor-card:active{border-color:var(--border2);}
.monitor-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.monitor-sym{font-weight:900;color:#fff;font-size:16px;letter-spacing:-.3px;}
.monitor-price{font-family:var(--mono);font-size:17px;font-weight:800;color:#fff;transition:color .3s;}
.monitor-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.monitor-chg{font-family:var(--mono);font-size:12px;font-weight:700;}
.monitor-score-val{font-family:var(--mono);font-size:13px;font-weight:800;color:var(--green);}
.monitor-bottom{display:flex;align-items:flex-end;justify-content:space-between;padding-top:10px;border-top:1px solid var(--border);}
.monitor-label{font-size:9px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:.06em;font-weight:700;}
.monitor-val{font-family:var(--mono);font-size:13px;font-weight:700;}
.monitor-val.amber{color:var(--amber);}
.monitor-val.green{color:var(--green);}
.monitor-val.red{color:var(--red);}
.monitor-ind{font-size:10px;color:var(--muted);margin-top:8px;padding-top:8px;border-top:1px solid var(--border);font-weight:500;line-height:1.4;}
.dots{display:flex;gap:3px;align-items:center;}
.dot{width:6px;height:6px;border-radius:50%;background:var(--border2);}
.dotf{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 5px rgba(0,230,118,.6);}
.pill-buy,.pill-sell,.pill-hold,.pill-scan{font-size:10px;font-weight:800;padding:4px 10px;border-radius:6px;text-transform:uppercase;white-space:nowrap;display:inline-block;letter-spacing:.04em;}
.pill-buy{background:rgba(0,230,118,.12);color:var(--green);border:1px solid rgba(0,230,118,.3);}
.pill-sell{background:rgba(255,23,68,.12);color:var(--red);border:1px solid rgba(255,23,68,.3);}
.pill-hold{background:rgba(41,121,255,.12);color:var(--blue);border:1px solid rgba(41,121,255,.3);}
.pill-scan{background:rgba(255,171,0,.1);color:var(--amber);border:1px solid rgba(255,171,0,.25);}
.trade-cards{display:flex;flex-direction:column;gap:8px;}
.trade-card{background:linear-gradient(135deg,var(--bg1),var(--bg2));border:2px solid var(--border);border-radius:12px;padding:13px;box-shadow:0 4px 15px rgba(0,0,0,.3);}
.trade-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;}
.trade-pair{font-weight:900;color:#fff;font-size:15px;letter-spacing:-.2px;}
.trade-pnl{font-family:var(--mono);font-size:18px;font-weight:900;letter-spacing:-.5px;}
.trade-pnl.pos{color:var(--green);}
.trade-pnl.neg{color:var(--red);}
.trade-row{display:flex;gap:0;margin-bottom:9px;background:var(--bg2);border-radius:8px;overflow:hidden;}
.trade-item{flex:1;padding:7px 8px;border-right:1px solid var(--border);}
.trade-item:last-child{border-right:none;}
.trade-item-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:700;margin-bottom:3px;}
.trade-item-val{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--text);}
.trade-meta{display:flex;align-items:center;gap:8px;padding-top:9px;border-top:1px solid var(--border);}
.trade-result{font-size:12px;font-weight:800;color:var(--text);}
.trade-score{font-size:10px;color:var(--muted);font-family:var(--mono);font-weight:600;background:var(--bg2);padding:2px 6px;border-radius:4px;}
.trade-margin{font-size:10px;color:var(--amber);font-family:var(--mono);margin-left:auto;font-weight:700;}
.empty-card{text-align:center;color:var(--muted);padding:28px;font-size:12px;background:var(--bg1);border:1px solid var(--border);border-radius:12px;}
.chg-up{color:var(--green);}
.chg-dn{color:var(--red);}
footer{text-align:center;color:var(--muted2);font-size:10px;padding:20px;letter-spacing:.05em;}"""

        html = (
            "<!DOCTYPE html><html lang='en'><head>"
            "<meta charset='UTF-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=1.0'>"
            "<title>WhaleTrader Ultimate</title>"
            "<style>" + mobile_css + "</style>"
            "</head><body>"
            "<nav>"
            "<div class='logo'><div class='logo-icon'>&#x1F40B;</div>WhaleTrader Ultimate</div>"
            "<div class='nav-right'>"
            + risk_badge +
            "<div class='badge-live'><div class='pulse-dot'></div>LIVE</div>"
            "</div>"
            "</nav>"
            "<div class='main'>"
            "<div style='text-align:right;font-size:9px;color:var(--muted);font-family:var(--mono);margin-bottom:10px;'>"
            "Sync: " + now_str + " &nbsp;|&nbsp; <span id='clk'>-- IST</span>"
            "</div>"
            "<div class='stat-grid'>"
            "<div class='stat-card c1'><div class='stat-label'>Equity</div><div class='stat-value' style='color:#fff;'>$" + str(wallet) + "</div><div class='stat-sub'>Started $1,000</div><div class='mini-bar'><div class='mini-fill' style='width:100%;background:var(--blue);'></div></div></div>"
            "<div class='stat-card c2'><div class='stat-label'>Total P&amp;L</div><div class='stat-value' style='color:" + pnl_color + ";'>" + pnl_prefix + "$" + str(total_pnl) + "</div><div class='stat-sub' style='color:" + pnl_color + ";'>" + pnl_prefix + str(pnl_pct) + "% return</div><div class='mini-bar'><div class='mini-fill' style='width:" + str(min(abs(pnl_pct),100)) + "%;background:" + pnl_color + ";'></div></div></div>"
            "<div class='stat-card c3'><div class='stat-label'>Win Rate</div><div class='stat-value' style='color:" + wr_color + ";'>" + str(win_rate) + "%</div><div class='stat-sub'>" + str(wins) + "W / " + str(losses) + "L / " + str(total_t) + " total</div><div class='mini-bar'><div class='mini-fill' style='width:" + str(win_rate) + "%;background:" + wr_color + ";'></div></div></div>"
            "<div class='stat-card c4'><div class='stat-label'>Today P&amp;L</div><div class='stat-value' style='color:" + today_color + ";'>" + today_prefix + "$" + str(today_pnl) + "</div><div class='stat-sub'>" + str(today_trades) + " trades today</div></div>"
            "<div class='stat-card c5'><div class='stat-label'>Best Trade</div><div class='stat-value' style='color:var(--green);'>+$" + str(best) + "</div><div class='stat-sub'>All time high</div></div>"
            "<div class='stat-card c6'><div class='stat-label'>Worst Trade</div><div class='stat-value' style='color:var(--red);'>$" + str(worst) + "</div><div class='stat-sub'>Max drawdown</div></div>"
            "<div class='stat-card c7'><div class='stat-label'>Positions</div><div class='stat-value' style='color:var(--blue);'>" + str(open_pos) + "/" + str(MAX_POSITIONS) + "</div><div class='stat-sub'>Max " + str(MAX_POSITIONS) + " allowed</div></div>"
            "<div class='stat-card c8'><div class='stat-label'>Strategy</div><div class='stat-value' style='color:var(--amber);'>10x</div><div class='stat-sub'>8 Strategies Active</div></div>"
            "</div>"
            "<div class='sec-head'><div class='sec-title'>Daily P&amp;L</div><div class='sec-line'></div><div class='sec-tag'>14 days</div></div>"
            "<div class='chart-card'><div class='chart-head'><div class='chart-title'>Returns Per Day</div><div class='chart-sub'>Paper trading</div></div><canvas id='pnlChart' style='max-height:130px;'></canvas></div>"
            "<div class='sec-head'><div class='sec-title'>Live Monitor</div><div class='sec-line'></div><div class='sec-tag'>6 pairs</div></div>"
            "<div class='monitor-cards'>" + monitor_rows + "</div>"
            "<div class='sec-head'><div class='sec-title'>Settlement Log</div><div class='sec-line'></div><div class='sec-tag'>Last 50</div></div>"
            "<div class='trade-cards'>" + history_rows + "</div>"
            "</div>"
            "<footer>WhaleTrader Ultimate &middot; 15 min auto-sync &middot; Paper Trading &middot; $1,000 Capital &middot; 8 Strategies</footer>"
            "<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js'></script>"
            "<script>"
            "function updateClock(){var now=new Date();var ist=new Date(now.getTime()+19800000);var h=ist.getUTCHours(),m=ist.getUTCMinutes(),s=ist.getUTCSeconds();var ampm=h>=12?'PM':'AM';var hh=h%12||12;function pad(n){return n<10?'0'+n:''+n;}document.getElementById('clk').textContent=pad(hh)+':'+pad(m)+':'+pad(s)+' '+ampm+' IST';}setInterval(updateClock,1000);updateClock();"
            "var ctx=document.getElementById('pnlChart').getContext('2d');"
            "new Chart(ctx,{type:'bar',data:{labels:" + chart_labels + ",datasets:[{data:" + chart_values + ",backgroundColor:" + chart_colors + ",borderRadius:4,borderSkipped:false}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(26,37,64,.5)'},ticks:{color:'#4a6080',font:{size:9}}},y:{grid:{color:'rgba(26,37,64,.5)'},ticks:{color:'#4a6080',font:{size:9},callback:function(v){return '$'+v;}}}}}}});"
            "var symMap={btcusdt:'BTC',ethusdt:'ETH',solusdt:'SOL',bnbusdt:'BNB',xrpusdt:'XRP',dogeusdt:'DOGE'};"
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
"  ws.onerror=function(){ws.close();};"
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
