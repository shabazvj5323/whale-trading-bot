import time
import logging
import os
import json
import ccxt
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("WhaleTrader_Pro")

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
INITIAL_CAPITAL   = 1000.0
MARGIN_PER_TRADE  = 100.0
LEVERAGE          = 10
HISTORY_FILE      = "history.json"

# Signal thresholds
MIN_SCORE         = 7          # 7/10 minimum to enter trade
FULL_MARGIN_SCORE = 9          # 9-10 → full $100 margin
HALF_MARGIN_SCORE = 7          # 7-8  → $60 margin

# Indicators
RSI_PERIOD   = 7
BB_PERIOD    = 15
BB_STD       = 1.5
EMA_FAST     = 9
EMA_SLOW     = 21
ATR_PERIOD   = 10
VOL_MULT     = 1.2             # volume spike threshold

# Dynamic TP/SL multipliers (ATR based)
TP_MULT = 1.2
SL_MULT = 0.6                  # 2:1 reward:risk always

# IST active trading sessions (UTC hours)
ACTIVE_SESSIONS = [
    (6, 10),    # London open  11:30–15:30 IST
    (13, 18),   # NY open      18:30–23:30 IST
]


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def ist_str():
    return ist_now().strftime("%Y-%m-%d %I:%M:%S %p")

def ist_short():
    return ist_now().strftime("%m-%d %H:%M")

def is_active_session():
    utc_hour = datetime.utcnow().hour
    for start, end in ACTIVE_SESSIONS:
        if start <= utc_hour < end:
            return True
    return False

def ema(arr, period):
    k = 2 / (period + 1)
    result = [arr[0]]
    for price in arr[1:]:
        result.append(price * k + result[-1] * (1 - k))
    return np.array(result)

def rsi(closes, period):
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = np.mean(gains[:period])
    al = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    return 100 - (100 / (1 + ag / (al + 1e-10)))

def atr(highs, lows, closes, period):
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(closes[:-1] - lows[1:])))
    return np.mean(tr[-period:])

def bollinger(closes, period, std_mult):
    recent = closes[-period:]
    sma    = np.mean(recent)
    std    = np.std(recent)
    return sma + std_mult * std, sma, sma - std_mult * std


# ─────────────────────────────────────────────
#  MAIN ENGINE
# ─────────────────────────────────────────────
class WhaleEngine:
    def __init__(self):
        self.state        = self.load_history()
        self.dashboard    = []

        api_key    = os.getenv("BINANCE_API_KEY")
        secret_key = os.getenv("BINANCE_SECRET_KEY")

        if not api_key or not secret_key:
            self.mock = True
            log.warning("⚠️  Mock mode — no API keys found")
        else:
            self.mock = False
            self.exchange = ccxt.binance({
                "apiKey": api_key,
                "secret": secret_key,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            self.exchange.set_sandbox_mode(True)

    # ── Persistence ──────────────────────────
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
            }
        }
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE) as f:
                    data = json.load(f)
                # ensure all keys present
                for k, v in default.items():
                    if k not in data:
                        data[k] = v
                for k, v in default["stats"].items():
                    if k not in data.get("stats", {}):
                        data.setdefault("stats", {})[k] = v
                return data
            except Exception:
                return default
        return default

    def save_history(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    # ── Market data ──────────────────────────
    def fetch_data(self, symbol, tf="5m", limit=120):
        if self.mock:
            return self._synthetic(symbol, limit)
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
            arr   = np.array(ohlcv)
            # also fetch 15m for trend
            ohlcv15 = self.exchange.fetch_ohlcv(symbol, "15m", limit=60)
            arr15   = np.array(ohlcv15)
            return (arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5],
                    arr15[:, 4])   # 15m closes for trend
        except Exception as e:
            log.warning(f"Data fetch failed {symbol}: {e}")
            return self._synthetic(symbol, limit)

    def _synthetic(self, symbol, limit):
        np.random.seed(int(time.time() / 60) + sum(ord(c) for c in symbol))
        bases = {"BTC": 64000, "ETH": 1750, "SOL": 145, "BNB": 580, "XRP": 0.52, "DOGE": 0.12}
        base  = next((v for k, v in bases.items() if k in symbol), 100)
        closes = base + np.cumsum(np.random.normal(0, base * 0.0004, limit))
        closes = np.maximum(closes, base * 0.5)
        highs  = closes + np.abs(np.random.normal(0, base * 0.001, limit))
        lows   = closes - np.abs(np.random.normal(0, base * 0.001, limit))
        opens  = closes - np.random.normal(0, base * 0.0003, limit)
        vols   = np.random.uniform(200, 1500, limit)
        # inject occasional volume spike + price move for signal testing
        if np.random.random() < 0.4:
            idx = np.random.randint(limit - 5, limit)
            vols[idx]   *= 2.2
            closes[idx] -= base * 0.006  # price drop → BUY signal
        return opens, highs, lows, closes, vols, closes  # last = 15m proxy

    # ── Signal scoring ────────────────────────
    def score_signal(self, opens, highs, lows, closes, vols, closes15, symbol):
        score     = 0
        direction = None
        reasons   = []

        price   = closes[-1]
        vol_avg = np.mean(vols[-15:-1])
        vol_now = vols[-1]

        # 1. Volume spike (+2)
        vol_ratio = vol_now / (vol_avg + 1e-9)
        if vol_ratio > VOL_MULT:
            score += 2
            reasons.append(f"VOL {vol_ratio:.1f}x")

        # 2. RSI (+2)
        r = rsi(closes, RSI_PERIOD)
        if r < 35:
            score += 2
            direction = "buy"
            reasons.append(f"RSI {r:.1f}")
        elif r > 65:
            score += 2
            direction = "sell"
            reasons.append(f"RSI {r:.1f}")

        # 3. Bollinger Bands (+2)
        upper, mid, lower = bollinger(closes, BB_PERIOD, BB_STD)
        if price <= lower:
            score += 2
            direction = "buy"
            reasons.append("BB-Lower")
        elif price >= upper:
            score += 2
            direction = "sell"
            reasons.append("BB-Upper")

        # 4. EMA trend filter (+2) — must align with direction
        ema_f = ema(closes, EMA_FAST)[-1]
        ema_s = ema(closes, EMA_SLOW)[-1]
        trend_up   = ema_f > ema_s
        trend_down = ema_f < ema_s
        if direction == "buy"  and trend_up:
            score += 2
            reasons.append("EMA↑")
        elif direction == "sell" and trend_down:
            score += 2
            reasons.append("EMA↓")

        # 5. 15m trend confirmation (+2)
        if len(closes15) >= EMA_SLOW:
            ema15_f = ema(closes15, EMA_FAST)[-1]
            ema15_s = ema(closes15, EMA_SLOW)[-1]
            if direction == "buy"  and ema15_f > ema15_s:
                score += 2
                reasons.append("MTF↑")
            elif direction == "sell" and ema15_f < ema15_s:
                score += 2
                reasons.append("MTF↓")

        return score, direction, r, upper, lower, reasons

    # ── Position management ───────────────────
    def check_positions(self, symbol, closes):
        if symbol not in self.state["active_positions"]:
            return
        pos   = self.state["active_positions"][symbol]
        side  = pos["side"]
        entry = pos["entry"]
        tp    = pos["tp"]
        sl    = pos["sl"]
        margin = pos.get("margin", MARGIN_PER_TRADE)
        qty   = (margin * LEVERAGE) / entry

        for price in closes:
            hit, reason, exit_price = False, "", price
            if side == "buy":
                if price >= tp:
                    hit, reason, exit_price = True, "TP 🎯", tp
                elif price <= sl:
                    hit, reason, exit_price = True, "SL 🛑", sl
            else:
                if price <= tp:
                    hit, reason, exit_price = True, "TP 🎯", tp
                elif price >= sl:
                    hit, reason, exit_price = True, "SL 🛑", sl

            if hit:
                pnl = ((exit_price - entry) * qty) if side == "buy" else ((entry - exit_price) * qty)
                pnl = max(min(pnl, margin * 0.5), -margin * 0.25)

                self.state["total_pnl"] += pnl
                stats = self.state["stats"]
                stats["total_trades"] += 1
                if pnl > 0:
                    stats["wins"] += 1
                    stats["best_trade"] = max(stats["best_trade"], pnl)
                else:
                    stats["losses"] += 1
                    stats["worst_trade"] = min(stats["worst_trade"], pnl)

                today = ist_now().strftime("%Y-%m-%d")
                stats["daily_pnl"][today] = round(
                    stats["daily_pnl"].get(today, 0) + pnl, 2)

                record = {
                    "time": ist_short(), "symbol": symbol,
                    "side": side.upper(), "entry": round(entry, 4),
                    "exit": round(exit_price, 4), "pnl": round(pnl, 2),
                    "result": reason, "score": pos.get("score", 0),
                    "margin": margin,
                }
                self.state["trades"].append(record)
                del self.state["active_positions"][symbol]
                log.info(f"{'✅' if pnl > 0 else '❌'} {symbol} {side.upper()} closed | PnL: ${pnl:.2f} | {reason}")
                self.save_history()
                break

    # ── Main pipeline ─────────────────────────
    def run(self):
        log.info("🐋 WhaleTrader Pro — Pipeline Start")
        session_active = is_active_session()
        log.info(f"📍 Session active: {session_active}")

        for symbol in SYMBOLS:
            try:
                result = self.fetch_data(symbol)
                if result is None:
                    continue
                opens, highs, lows, closes, vols, closes15 = result

                # Check existing positions first
                self.check_positions(symbol, closes)

                price = round(closes[-1], 6)
                self.state["last_prices"][symbol] = price

                # Skip if already in position
                if symbol in self.state["active_positions"]:
                    pos = self.state["active_positions"][symbol]
                    self.dashboard.append({
                        "symbol": symbol, "price": price,
                        "signal": f"HOLDING {pos['side'].upper()}",
                        "score": pos.get("score", 0),
                        "entry": pos["entry"], "tp": pos["tp"], "sl": pos["sl"],
                        "reasons": pos.get("reasons", []),
                    })
                    continue

                # Score the signal
                score, direction, r, upper, lower, reasons = self.score_signal(
                    opens, highs, lows, closes, vols, closes15, symbol)

                log.info(f"📊 {symbol} | Price: {price} | RSI: {r:.1f} | Score: {score}/10 | {reasons}")

                # Only trade if session active AND score meets threshold
                if session_active and direction and score >= MIN_SCORE:
                    # Dynamic margin based on score
                    margin = MARGIN_PER_TRADE if score >= FULL_MARGIN_SCORE else 60.0

                    # Dynamic TP/SL using ATR
                    atr_val = atr(highs, lows, closes, ATR_PERIOD)
                    if direction == "buy":
                        tp = round(price + atr_val * TP_MULT, 6)
                        sl = round(price - atr_val * SL_MULT, 6)
                    else:
                        tp = round(price - atr_val * TP_MULT, 6)
                        sl = round(price + atr_val * SL_MULT, 6)

                    self.state["active_positions"][symbol] = {
                        "side": direction, "entry": price,
                        "tp": tp, "sl": sl, "margin": margin,
                        "score": score, "reasons": reasons,
                        "time": ist_short(),
                    }
                    self.save_history()
                    log.info(f"🚀 NEW TRADE: {symbol} {direction.upper()} | Score:{score} | Margin:${margin} | TP:{tp} SL:{sl}")

                self.dashboard.append({
                    "symbol": symbol, "price": price,
                    "signal": f"{direction.upper()} {score}/10" if direction else "SCANNING",
                    "score": score, "entry": price,
                    "tp": None, "sl": None, "reasons": reasons,
                })

            except Exception as e:
                log.error(f"Error processing {symbol}: {e}")

        self.save_history()
        html = self.generate_dashboard()
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
        log.info("✅ Dashboard updated → index.html")

    # ── Dashboard ─────────────────────────────
    def generate_dashboard(self):
        state  = self.state
        stats  = state.get("stats", {})
        trades = state.get("trades", [])

        total_pnl    = round(state.get("total_pnl", 0.0), 2)
        wallet       = round(INITIAL_CAPITAL + total_pnl, 2)
        pnl_color    = "#00e676" if total_pnl >= 0 else "#ff1744"
        pnl_prefix   = "+" if total_pnl >= 0 else ""
        total_t      = stats.get("total_trades", 0)
        wins         = stats.get("wins", 0)
        losses       = stats.get("losses", 0)
        win_rate     = round((wins / total_t * 100), 1) if total_t > 0 else 0
        best         = round(stats.get("best_trade", 0), 2)
        worst        = round(stats.get("worst_trade", 0), 2)

        # daily PnL chart data
        daily_pnl  = stats.get("daily_pnl", {})
        sorted_days = sorted(daily_pnl.items())[-14:]  # last 14 days
        chart_labels = [d[0][5:] for d in sorted_days]  # MM-DD
        chart_values = [d[1] for d in sorted_days]
        chart_colors = ["rgba(0,230,118,0.8)" if v >= 0 else "rgba(255,23,68,0.8)" for v in chart_values]

        # Monitor rows
        monitor_rows = ""
        for d in self.dashboard:
            sym       = d["symbol"]
            clean     = sym.replace("/", "").lower()
            sig       = d.get("signal", "SCANNING")
            score     = d.get("score", 0)
            reasons   = ", ".join(d.get("reasons", [])) or "—"
            tp_val    = f"${d['tp']}" if d.get("tp") else "—"
            sl_val    = f"${d['sl']}" if d.get("sl") else "—"

            if "HOLD" in sig:
                pill = f"<span class='pill hold'>{sig}</span>"
            elif "BUY" in sig:
                pill = f"<span class='pill buy'>{sig}</span>"
            elif "SELL" in sig:
                pill = f"<span class='pill sell'>{sig}</span>"
            else:
                pill = f"<span class='pill scan'>⚡ SCANNING</span>"

            score_bar = ""
            for i in range(10):
                filled = "filled" if i < score else ""
                score_bar += f"<span class='dot {filled}'></span>"

            monitor_rows += f"""
            <tr>
                <td class='sym'>{sym}</td>
                <td><span id='p-{clean}' class='price'>—</span></td>
                <td><span id='c-{clean}' class='chg'>—</span></td>
                <td>{pill}</td>
                <td><div class='score-bar'>{score_bar}</div></td>
                <td class='hint'>{reasons}</td>
                <td class='tp'>{tp_val}</td>
                <td class='sl'>{sl_val}</td>
            </tr>"""

        # Trade history rows
        history_rows = ""
        for t in reversed(trades[-30:]):
            pnl_v  = float(t["pnl"])
            pc     = "#00e676" if pnl_v >= 0 else "#ff1744"
            badge  = "buy" if t["side"] == "BUY" else "sell"
            icon   = "🎯" if "TP" in t.get("result","") else "🛑"
            sc     = t.get("score", "—")
            mg     = t.get("margin", 100)
            history_rows += f"""
            <tr>
                <td class='hint'>{t['time']}</td>
                <td class='sym'>{t['symbol']}</td>
                <td><span class='pill {badge}'>{t['side']}</span></td>
                <td>${t['entry']}</td>
                <td>${t['exit']}</td>
                <td>{icon} {t.get('result','—')}</td>
                <td style='color:{pc}; font-weight:700; font-family:monospace;'>{'+' if pnl_v>=0 else ''}${pnl_v}</td>
                <td class='hint'>{sc}/10</td>
                <td class='hint'>${mg}</td>
            </tr>"""

        if not history_rows:
            history_rows = "<tr><td colspan='9' class='empty'>No trades yet — scanning markets...</td></tr>"

        chart_labels_js = json.dumps(chart_labels)
        chart_values_js = json.dumps(chart_values)
        chart_colors_js = json.dumps(chart_colors)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WhaleTrader Pro</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  :root {{
    --bg:      #070a12;
    --bg2:     #0d1117;
    --bg3:     #131a27;
    --border:  #1e2d45;
    --text:    #c9d4e8;
    --muted:   #4a607a;
    --green:   #00e676;
    --red:     #ff1744;
    --blue:    #2979ff;
    --amber:   #ffab00;
    --font:    'Space Grotesk', sans-serif;
    --mono:    'Space Mono', monospace;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--font); font-size: 13px; }}

  /* HEADER */
  header {{
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
  }}
  .logo {{
    display: flex; align-items: center; gap: 10px;
    font-size: 15px; font-weight: 700; color: #fff;
  }}
  .pulse {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 0 rgba(0,230,118,0.4);
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(0,230,118,0.4); }}
    70%  {{ box-shadow: 0 0 0 8px rgba(0,230,118,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(0,230,118,0); }}
  }}
  .clock {{ font-family: var(--mono); font-size: 11px; color: var(--muted); }}

  /* LAYOUT */
  .wrap {{ max-width: 1300px; margin: 0 auto; padding: 20px 16px; }}

  /* STAT CARDS */
  .cards {{
    display: grid;
    grid-template-colu
