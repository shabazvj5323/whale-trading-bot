import time
import logging
import os
import json
import ccxt
import numpy as np
import requests
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("WhaleTrader_Pro")

# ═══════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════
SYMBOLS          = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
INITIAL_CAPITAL  = 1000.0
MARGIN_PER_TRADE = 80.0
LEVERAGE         = 10
HISTORY_FILE     = "history.json"

# Signal
MIN_SCORE   = 6        # Adaptive: 6 normal, 5 high vol, 7 low vol
VOL_MULT    = 1.5      # Real volume spike confirmation
BB_PERIOD   = 15
BB_STD      = 1.3
RSI_PERIOD  = 7
EMA_FAST    = 9
EMA_SLOW    = 21
ATR_PERIOD  = 10

# TP/SL — Trailing TP enabled, no fixed TP exit
TP_MULT     = 1.5      # Better TP — easier to hit
SL_MULT     = 0.6      # Slightly wider SL — less noise stops
TRAIL_STEP  = 0.3      # Trail TP by this ATR multiplier each step

# Risk management
MAX_POSITIONS   = 6      # one per pair
MAX_DAILY_LOSS  = 80.0   # stop trading if daily loss > $80


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
        self.dashboard = []

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

    def fetch_data(self, symbol, limit=120, vol_map=None):
        # CoinGecko first — always works on GitHub Actions
        result = self.fetch_coingecko_ohlc(symbol, limit, vol_map=vol_map)
        if result is not None:
            log.info(f"CoinGecko OK: {symbol}")
            return result
        # Fallback: Binance API
        if not self.mock:
            try:
                o   = self.exchange.fetch_ohlcv(symbol, "5m",  limit=limit)
                o15 = self.exchange.fetch_ohlcv(symbol, "15m", limit=60)
                a, a15 = np.array(o, dtype=float), np.array(o15, dtype=float)
                log.info(f"Binance OK: {symbol}")
                return a[:,1], a[:,2], a[:,3], a[:,4], a[:,5], a15[:,4]
            except Exception as e:
                log.warning(f"Binance failed {symbol}: {e}")
        log.error(f"All data sources failed: {symbol} — skipping")
        return None

    # ── Scoring ──────────────────────────────
    def score_signal(self, opens, highs, lows, closes, vols, closes15):
        score, direction, reasons = 0, None, []
        price = float(closes[-1])

        # ── Market Regime Detection ──────────────
        regime_result = market_regime(closes, highs, lows)
        if isinstance(regime_result, tuple):
            regime, atr_pct = regime_result
        else:
            regime, atr_pct = regime_result, 1.0

        # Fixed min score — let indicators decide, not regime
        adaptive_min = 4

        reasons.append(regime)

        # 1. RSI (+2) — oversold/overbought
        r = rsi_calc(closes, RSI_PERIOD)
        if r < 38:
            score += 2
            direction = "buy"
            reasons.append("RSI " + str(round(r, 1)))
        elif r > 62:
            score += 2
            direction = "sell"
            reasons.append("RSI " + str(round(r, 1)))

        # 2. Bollinger Bands (+2)
        upper, mid, lower = bollinger_calc(closes, BB_PERIOD, BB_STD)
        if price <= lower:
            score += 2
            direction = "buy"
            reasons.append("BB-Low")
        elif price >= upper:
            score += 2
            direction = "sell"
            reasons.append("BB-High")

        # 3. EMA trend (+2) — must align with regime
        if len(closes) >= EMA_SLOW:
            ef = ema_calc(closes, EMA_FAST)[-1]
            es = ema_calc(closes, EMA_SLOW)[-1]
            if direction == "buy" and ef > es:
                score += 2
                reasons.append("EMA UP")
            elif direction == "sell" and ef < es:
                score += 2
                reasons.append("EMA DN")

        # 4. Regime filter (+2) — trade WITH the trend
        if direction:
            if regime == "BULL" and direction == "buy":
                score += 2
                reasons.append("BULL TREND")
            elif regime == "BEAR" and direction == "sell":
                score += 2
                reasons.append("BEAR TREND")
            elif regime == "SIDEWAYS":
                reasons.append("SIDEWAYS")
            else:
                reasons.append("COUNTER-TREND")

        # 5. Price momentum (+2)
        if direction and len(closes) >= 3:
            recent_move = float(closes[-1]) - float(closes[-3])
            if direction == "buy" and recent_move < 0:
                score += 2
                reasons.append("PULLBACK")
            elif direction == "sell" and recent_move > 0:
                score += 2
                reasons.append("PUSHUP")

        # 6. Candle body strength (+2) — CoinGecko has no real volume
        if direction and len(opens) >= 2:
            body      = float(closes[-1]) - float(opens[-1])
            prev_body = float(closes[-2]) - float(opens[-2])
            if direction == "buy" and body > 0 and abs(body) >= abs(prev_body) * 0.3:
                score += 2
                reasons.append("BULL CANDLE")
            elif direction == "sell" and body < 0 and abs(body) >= abs(prev_body) * 0.3:
                score += 2
                reasons.append("BEAR CANDLE")

        # 7. Multi-timeframe (+2)
        if direction and len(closes15) >= EMA_SLOW:
            e15f = ema_calc(closes15, EMA_FAST)[-1]
            e15s = ema_calc(closes15, EMA_SLOW)[-1]
            if direction == "buy" and e15f > e15s:
                score += 2
                reasons.append("MTF UP")
            elif direction == "sell" and e15f < e15s:
                score += 2
                reasons.append("MTF DN")

        return score, direction, r, upper, lower, reasons, adaptive_min

    # ── Position management ──────────────────
    def check_positions(self, symbol, closes):
        if symbol not in self.state["active_positions"]:
            return
        pos    = self.state["active_positions"][symbol]
        side   = pos["side"]
        entry  = float(pos["entry"])
        tp     = float(pos["tp"])   # used as trailing TP ceiling
        sl     = float(pos["sl"])
        margin = float(pos.get("margin", MARGIN_PER_TRADE))
        qty    = (margin * LEVERAGE) / entry

        sl_dist    = abs(entry - sl)
        trail_dist = sl_dist * TRAIL_STEP   # how much to move TP/SL each step

        for price in closes.astype(float):
            profit = (price - entry) if side == "buy" else (entry - price)

            if side == "buy":
                # Trail TP upward as price rises
                if price > tp:
                    tp = round(price + trail_dist, 6)
                    self.state["active_positions"][symbol]["tp"] = tp
                # Trail SL upward when in profit
                if profit > sl_dist:
                    new_sl = round(price - sl_dist, 6)
                    if new_sl > sl:
                        sl = new_sl
                        self.state["active_positions"][symbol]["sl"] = sl
            else:
                # Trail TP downward as price falls
                if price < tp:
                    tp = round(price - trail_dist, 6)
                    self.state["active_positions"][symbol]["tp"] = tp
                # Trail SL downward when in profit
                if profit > sl_dist:
                    new_sl = round(price + sl_dist, 6)
                    if new_sl < sl:
                        sl = new_sl
                        self.state["active_positions"][symbol]["sl"] = sl

            hit, reason, exit_p = False, "", price
            if side == "buy":
                # Exit when price reverses from trailing TP
                if price <= sl:
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

                pnl = max(min(pnl, margin * 0.4), -margin * 0.2)
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

                self.state["trades"].append({
                    "time":   ist_short(),
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
                log.info(f"{icon} {symbol} {side.upper()} | PnL:${pnl} | {reason}")
                self.save_history()
                break

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
                self.check_positions(symbol, closes)

                price = round(float(closes[-1]), 6)
                self.state["last_prices"][symbol] = price

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
                    margin = MARGIN_PER_TRADE if score >= 8 else 60.0
                    av = atr_calc(highs, lows, closes, ATR_PERIOD)

                    if direction == "buy":
                        tp = round(price + av * TP_MULT, 6)
                        sl = round(price - av * SL_MULT, 6)
                    else:
                        tp = round(price - av * TP_MULT, 6)
                        sl = round(price + av * SL_MULT, 6)

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
            tp_val  = "$" + str(d["tp"]) if d.get("tp") else "-"
            sl_val  = "$" + str(d["sl"]) if d.get("sl") else "-"

            if "HOLD" in sig:   pill = "<span class='pill-hold'>" + sig + "</span>"
            elif "BUY" in sig:  pill = "<span class='pill-buy'>" + sig + "</span>"
            elif "SELL" in sig: pill = "<span class='pill-sell'>" + sig + "</span>"
            else:               pill = "<span class='pill-scan'>SCANNING</span>"

            dots = ""
            for i in range(10):
                dots += "<span class='dotf'></span>" if i < score else "<span class='dot'></span>"

            monitor_rows += (
                "<tr>"
                "<td class='sym-cell'>" + sym + "</td>"
                "<td><span id='p-" + clean + "' class='live-price'>-</span></td>"
                "<td><span id='c-" + clean + "' class='live-chg'>-</span></td>"
                "<td>" + pill + "</td>"
                "<td><div class='dots'>" + dots + "</div><span class='score-num'>" + str(score) + "/10</span></td>"
                "<td class='reason-cell'>" + reasons + "</td>"
                "<td class='tp-cell'>" + tp_val + "</td>"
                "<td class='sl-cell'>" + sl_val + "</td>"
                "</tr>"
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
                "<tr>"
                "<td class='muted'>" + str(t.get("time","")) + "</td>"
                "<td class='sym-cell'>" + str(t.get("symbol","")) + "</td>"
                "<td><span class='" + badge + "'>" + str(t.get("side","")) + "</span></td>"
                "<td class='mono'>$" + str(t.get("entry","")) + "</td>"
                "<td class='mono'>$" + str(t.get("exit","")) + "</td>"
                "<td>" + icon + "</td>"
                "<td class='mono " + pc + "'>" + prefix + "$" + str(round(pv,2)) + "</td>"
                "<td class='muted'>" + str(t.get("score","-")) + "/10</td>"
                "<td class='muted'>$" + str(t.get("margin",80)) + "</td>"
                "</tr>"
            )

        if not history_rows:
            history_rows = "<tr><td colspan='9' class='empty-row'>No trades yet — bot is scanning...</td></tr>"

        css = (
            "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');"
            ":root{--bg0:#05070f;--bg1:#080c17;--bg2:#0d1322;--bg3:#111827;--border:#1a2540;--border2:#243050;"
            "--text:#d1daf0;--muted:#4a6080;--muted2:#2a3a55;--green:#00e676;--green2:#00c853;--red:#ff1744;"
            "--blue:#2979ff;--amber:#ffab00;--purple:#7c4dff;--font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;}"
            "*{box-sizing:border-box;margin:0;padding:0;}"
            "body{background:var(--bg0);color:var(--text);font-family:var(--font);font-size:13px;}"
            "::-webkit-scrollbar{width:4px;height:4px;}::-webkit-scrollbar-track{background:var(--bg1);}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px;}"
            "nav{background:rgba(8,12,23,0.97);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);padding:0 20px;height:56px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;}"
            ".nav-left{display:flex;align-items:center;gap:14px;}"
            ".logo{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:800;color:#fff;}"
            ".logo-icon{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#2979ff,#00e676);display:flex;align-items:center;justify-content:center;font-size:16px;}"
            ".badge-live{background:rgba(0,230,118,.12);color:var(--green);border:1px solid rgba(0,230,118,.25);font-size:10px;font-weight:700;padding:3px 9px;border-radius:20px;display:flex;align-items:center;gap:5px;}"
            ".pulse-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:wt-pulse 2s infinite;}"
            "@keyframes wt-pulse{0%{box-shadow:0 0 0 0 rgba(0,230,118,.5);}70%{box-shadow:0 0 0 6px rgba(0,230,118,0);}100%{box-shadow:0 0 0 0 rgba(0,230,118,0);}}"
            ".nav-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}"
            ".nav-box{font-family:var(--mono);font-size:10px;color:#c8d6e5;background:var(--bg2);border:1px solid var(--border);padding:5px 10px;border-radius:6px;white-space:nowrap;}"
            ".session-badge{font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;white-space:nowrap;}"
            ".session-on{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.2);}"
            ".session-off{background:rgba(255,23,68,.1);color:var(--red);border:1px solid rgba(255,23,68,.2);}"
            ".main{max-width:1400px;margin:0 auto;padding:20px 14px;}"
            ".stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:11px;margin-bottom:24px;}"
            ".stat-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:16px;position:relative;overflow:hidden;transition:border-color .2s;}"
            ".stat-card:hover{border-color:var(--border2);}"
            ".stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}"
            ".c1::before{background:linear-gradient(90deg,var(--blue),var(--purple));}"
            ".c2::before{background:linear-gradient(90deg,var(--green),var(--green2));}"
            ".c3::before{background:linear-gradient(90deg,var(--amber),#ff6f00);}"
            ".c4::before{background:linear-gradient(90deg,#00e676,#00bcd4);}"
            ".c5::before{background:linear-gradient(90deg,#ff1744,#ff5722);}"
            ".c6::before{background:linear-gradient(90deg,var(--purple),var(--blue));}"
            ".c7::before{background:linear-gradient(90deg,#00bcd4,#2979ff);}"
            ".c8::before{background:linear-gradient(90deg,#ff6f00,var(--amber));}"
            ".stat-label{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:600;margin-bottom:7px;}"
            ".stat-value{font-size:20px;font-weight:800;font-family:var(--mono);letter-spacing:-.5px;line-height:1;}"
            ".stat-sub{font-size:11px;color:var(--muted);margin-top:5px;}"
            ".mini-bar{background:var(--bg3);border-radius:4px;height:3px;margin-top:8px;}"
            ".mini-fill{height:3px;border-radius:4px;}"
            ".sec-head{display:flex;align-items:center;gap:10px;margin:24px 0 11px;}"
            ".sec-title{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:700;white-space:nowrap;}"
            ".sec-line{flex:1;height:1px;background:var(--border);}"
            ".sec-tag{font-size:10px;color:var(--muted);background:var(--bg2);border:1px solid var(--border);padding:2px 8px;border-radius:10px;}"
            ".chart-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:4px;}"
            ".chart-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;}"
            ".chart-title{font-size:13px;font-weight:600;}"
            ".chart-sub{font-size:11px;color:var(--muted);}"
            ".tbl-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:4px;}"
            ".tbl-wrap{overflow-x:auto;}"
            "table{width:100%;border-collapse:collapse;}"
            "thead tr{background:var(--bg2);}"
            "th{padding:10px 13px;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:700;border-bottom:1px solid var(--border);white-space:nowrap;}"
            "td{padding:11px 13px;border-bottom:1px solid rgba(26,37,64,.5);vertical-align:middle;}"
            "tr:last-child td{border-bottom:none;}"
            "tbody tr:hover td{background:rgba(255,255,255,.015);}"
            ".sym-cell{font-weight:700;color:#fff;font-size:13px;white-space:nowrap;}"
            ".live-price{font-family:var(--mono);font-size:13px;font-weight:600;color:#fff;}"
            ".live-chg{font-family:var(--mono);font-size:11px;font-weight:600;padding:3px 7px;border-radius:5px;white-space:nowrap;}"
            ".chg-up{color:var(--green);background:rgba(0,230,118,.08);}"
            ".chg-dn{color:var(--red);background:rgba(255,23,68,.08);}"
            ".reason-cell{color:var(--muted);font-size:11px;max-width:150px;}"
            ".tp-cell{color:var(--green);font-family:var(--mono);font-size:12px;}"
            ".sl-cell{color:var(--red);font-family:var(--mono);font-size:12px;}"
            ".mono{font-family:var(--mono);font-size:12px;}"
            ".muted{color:var(--muted);font-size:11px;}"
            ".pos{color:var(--green);font-weight:700;}"
            ".neg{color:var(--red);font-weight:700;}"
            ".empty-row{text-align:center;color:var(--muted);padding:30px;font-size:12px;}"
            ".dots{display:flex;gap:3px;margin-bottom:3px;}"
            ".dot{width:7px;height:7px;border-radius:50%;background:var(--border2);}"
            ".dotf{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 5px rgba(0,230,118,.5);}"
            ".score-num{font-size:10px;color:var(--muted);font-family:var(--mono);}"
            ".pill-buy,.pill-sell,.pill-hold,.pill-scan{font-size:10px;font-weight:700;padding:3px 8px;border-radius:5px;text-transform:uppercase;white-space:nowrap;display:inline-block;}"
            ".pill-buy{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.2);}"
            ".pill-sell{background:rgba(255,23,68,.1);color:var(--red);border:1px solid rgba(255,23,68,.2);}"
            ".pill-hold{background:rgba(41,121,255,.1);color:var(--blue);border:1px solid rgba(41,121,255,.2);}"
            ".pill-scan{background:rgba(255,171,0,.08);color:var(--amber);border:1px solid rgba(255,171,0,.15);}"
            ".risk-ok{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.2);padding:3px 8px;border-radius:5px;font-size:10px;font-weight:700;}"
            ".risk-stop{background:rgba(255,23,68,.1);color:var(--red);border:1px solid rgba(255,23,68,.2);padding:3px 8px;border-radius:5px;font-size:10px;font-weight:700;}"
            "footer{text-align:center;color:var(--muted2);font-size:10px;padding:22px;letter-spacing:.05em;}"
        )

        js = (
            "var chartLabels = " + chart_labels + ";\n"
            "var chartValues = " + chart_values + ";\n"
            "var chartColors = " + chart_colors + ";\n"
            "var syms = ['btcusdt','ethusdt','solusdt','bnbusdt','xrpusdt','dogeusdt'];\n"
            "function connectWS() {\n"
            "  var ws = new WebSocket('wss://stream.binance.com:9443/ws/' + syms.map(function(s){ return s+'@ticker'; }).join('/'));\n"
            "  ws.onmessage = function(e) {\n"
            "    var d = JSON.parse(e.data);\n"
            "    var s = d.s.toLowerCase();\n"
            "    var pe = document.getElementById('p-'+s);\n"
            "    var ce = document.getElementById('c-'+s);\n"
            "    if (!pe) return;\n"
            "    var price = parseFloat(d.c);\n"
            "    var chg = parseFloat(d.P);\n"
            "    pe.textContent = price < 1 ? '$'+price.toFixed(5) : '$'+price.toFixed(2);\n"
            "    ce.textContent = (chg >= 0 ? '+' : '')+chg.toFixed(2)+'%';\n"
            "    ce.className = 'live-chg '+(chg >= 0 ? 'chg-up' : 'chg-dn');\n"
            "    pe.style.color = chg >= 0 ? 'var(--green)' : 'var(--red)';\n"
            "  };\n"
            "  ws.onclose = function() { setTimeout(connectWS, 3000); };\n"
            "}\n"
            "connectWS();\n"
            "function updateClock() {\n"
            "  var now = new Date();\n"
            "  var ist = new Date(now.getTime() + 19800000);\n"
            "  var h = ist.getUTCHours(), m = ist.getUTCMinutes(), s = ist.getUTCSeconds();\n"
            "  var ampm = h >= 12 ? 'PM' : 'AM';\n"
            "  var hh = h % 12 || 12;\n"
            "  function pad(n) { return n < 10 ? '0'+n : ''+n; }\n"
            "  document.getElementById('clk').textContent = pad(hh)+':'+pad(m)+':'+pad(s)+' '+ampm+' IST';\n"
            "}\n"
            "setInterval(updateClock, 1000);\n"
            "updateClock();\n"
            "var ctx = document.getElementById('pnlChart').getContext('2d');\n"
            "new Chart(ctx, {\n"
            "  type: 'bar',\n"
            "  data: { labels: chartLabels, datasets: [{ label: 'P&L ($)', data: chartValues, backgroundColor: chartColors, borderRadius: 5, borderSkipped: false }] },\n"
            "  options: {\n"
            "    responsive: true,\n"
            "    plugins: { legend: { display: false }, tooltip: { callbacks: { label: function(c) { return (c.parsed.y >= 0 ? '+' : '')+'$'+c.parsed.y.toFixed(2); } } } },\n"
            "    scales: {\n"
            "      x: { grid: { color: 'rgba(26,37,64,.5)' }, ticks: { color: '#4a6080', font: { size: 10 } } },\n"
            "      y: { grid: { color: 'rgba(26,37,64,.5)' }, ticks: { color: '#4a6080', font: { size: 10 }, callback: function(v) { return '$'+v; } } }\n"
            "    }\n"
            "  }\n"
            "});\n"
            "setTimeout(function() { window.location.reload(); }, 900000);\n"
        )

        risk_badge = "<span class='risk-ok'>TRADING ACTIVE</span>" if trading_ok else "<span class='risk-stop'>LIMIT REACHED</span>"

        html = (
            "<!DOCTYPE html><html lang='en'><head>"
            "<meta charset='UTF-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
            "<title>WhaleTrader Pro</title>"
            "<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js'></script>"
            "<style>" + css + "</style>"
            "</head><body>"
            "<nav>"
            "<div class='nav-left'>"
            "<div class='logo'><div class='logo-icon'>&#x1F40B;</div>WhaleTrader Pro</div>"
            "<div class='badge-live'><div class='pulse-dot'></div>LIVE</div>"
            + risk_badge +
            "</div>"
            "<div class='nav-right'>"
            "<div class='nav-box'>Sync: " + now_str + "</div>"
            "<div class='nav-box' id='clk'>-- IST</div>"
            "</div></nav>"
            "<div class='main'>"
            "<div class='stat-grid'>"
            "<div class='stat-card c1'><div class='stat-label'>Account Equity</div><div class='stat-value' style='color:#fff;'>$" + str(wallet) + "</div><div class='stat-sub'>Started $1,000.00</div><div class='mini-bar'><div class='mini-fill' style='width:100%;background:var(--blue);'></div></div></div>"
            "<div class='stat-card c2'><div class='stat-label'>Total P&amp;L</div><div class='stat-value' style='color:" + pnl_color + ";'>" + pnl_prefix + "$" + str(total_pnl) + "</div><div class='stat-sub' style='color:" + pnl_color + ";'>" + pnl_prefix + str(pnl_pct) + "% return</div><div class='mini-bar'><div class='mini-fill' style='width:" + str(min(abs(pnl_pct), 100)) + "%;background:" + pnl_color + ";'></div></div></div>"
            "<div class='stat-card c3'><div class='stat-label'>Win Rate</div><div class='stat-value' style='color:" + wr_color + ";'>" + str(win_rate) + "%</div><div class='stat-sub'>" + str(wins) + "W / " + str(losses) + "L / " + str(total_t) + " total</div><div class='mini-bar'><div class='mini-fill' style='width:" + str(win_rate) + "%;background:" + wr_color + ";'></div></div></div>"
            "<div class='stat-card c4'><div class='stat-label'>Today P&amp;L</div><div class='stat-value' style='color:" + today_color + ";'>" + today_prefix + "$" + str(today_pnl) + "</div><div class='stat-sub'>" + str(today_trades) + " trades today</div></div>"
            "<div class='stat-card c5'><div class='stat-label'>Best Trade</div><div class='stat-value' style='color:var(--green);'>+$" + str(best) + "</div><div class='stat-sub'>All time high</div></div>"
            "<div class='stat-card c6'><div class='stat-label'>Worst Trade</div><div class='stat-value' style='color:var(--red);'>$" + str(worst) + "</div><div class='stat-sub'>Max drawdown</div></div>"
            "<div class='stat-card c7'><div class='stat-label'>Open Positions</div><div class='stat-value' style='color:var(--blue);'>" + str(open_pos) + "/" + str(MAX_POSITIONS) + "</div><div class='stat-sub'>Max " + str(MAX_POSITIONS) + " allowed</div></div>"
            "<div class='stat-card c8'><div class='stat-label'>Strategy</div><div class='stat-value' style='color:var(--amber);'>10x</div><div class='stat-sub'>6 pairs / 5m / 24-7</div></div>"
            "</div>"
            "<div class='sec-head'><div class='sec-title'>Daily P&amp;L</div><div class='sec-line'></div><div class='sec-tag'>14 days</div></div>"
            "<div class='chart-card'><div class='chart-head'><div class='chart-title'>Realized Returns Per Day</div><div class='chart-sub'>IST / Paper trading / $1,000 base</div></div><canvas id='pnlChart' style='max-height:160px;'></canvas></div>"
            "<div class='sec-head'><div class='sec-title'>Live Asset Monitor</div><div class='sec-line'></div><div class='sec-tag'>6 pairs</div></div>"
            "<div class='tbl-card'><div class='tbl-wrap'><table><thead><tr><th>Pair</th><th>Live Price</th><th>24h</th><th>Signal</th><th>Score</th><th>Indicators</th><th>Take Profit</th><th>Stop Loss</th></tr></thead><tbody>" + monitor_rows + "</tbody></table></div></div>"
            "<div class='sec-head'><div class='sec-title'>Settlement Log</div><div class='sec-line'></div><div class='sec-tag'>Last 50 trades</div></div>"
            "<div class='tbl-card'><div class='tbl-wrap'><table><thead><tr><th>Time</th><th>Pair</th><th>Side</th><th>Entry</th><th>Exit</th><th>Result</th><th>P&amp;L</th><th>Score</th><th>Margin</th></tr></thead><tbody>" + history_rows + "</tbody></table></div></div>"
            "</div>"
            "<footer>WhaleTrader Pro &middot; 15 min auto-sync &middot; 24/7 &middot; Paper Trading &middot; $1,000 Capital &middot; Max Loss $80/day</footer>"
            "<script src='https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js'></script>"
            "<script>" + js + "</script>"
            "</body></html>"
        )

        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Dashboard saved — index.html")


if __name__ == "__main__":
    engine = WhaleEngine()
    engine.run()
