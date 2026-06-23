import time
import logging
import os
import json
import ccxt
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("WhaleTrader_Pro")

SYMBOLS          = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"]
INITIAL_CAPITAL  = 1000.0
MARGIN_PER_TRADE = 100.0
LEVERAGE         = 10
HISTORY_FILE     = "history.json"
TEMPLATE_FILE    = "template.html"
MIN_SCORE        = 7
RSI_PERIOD       = 7
BB_PERIOD        = 15
BB_STD           = 1.5
EMA_FAST         = 9
EMA_SLOW         = 21
ATR_PERIOD       = 10
VOL_MULT         = 1.2
TP_MULT          = 1.2
SL_MULT          = 0.6
ACTIVE_SESSIONS  = [(6, 10), (13, 18)]


def ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

def ist_str():
    return ist_now().strftime("%Y-%m-%d %I:%M:%S %p")

def ist_short():
    return ist_now().strftime("%m-%d %H:%M")

def is_active_session():
    h = datetime.utcnow().hour
    return any(s <= h < e for s, e in ACTIVE_SESSIONS)

def ema(arr, period):
    k = 2 / (period + 1)
    r = [arr[0]]
    for p in arr[1:]:
        r.append(p * k + r[-1] * (1 - k))
    return np.array(r)

def rsi(closes, period):
    d = np.diff(closes)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag, al = np.mean(g[:period]), np.mean(l[:period])
    for i in range(period, len(d)):
        ag = (ag * (period - 1) + g[i]) / period
        al = (al * (period - 1) + l[i]) / period
    return 100 - (100 / (1 + ag / (al + 1e-10)))

def atr_val(highs, lows, closes, period):
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(closes[:-1] - lows[1:])))
    return np.mean(tr[-period:])

def bollinger(closes, period, std_mult):
    r = closes[-period:]
    s = np.mean(r)
    d = np.std(r)
    return s + std_mult * d, s, s - std_mult * d


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
                "apiKey": api_key, "secret": secret_key,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            self.exchange.set_sandbox_mode(True)

    def load_history(self):
        default = {
            "total_pnl": 0.0, "active_positions": {},
            "trades": [], "last_prices": {},
            "stats": {
                "total_trades": 0, "wins": 0, "losses": 0,
                "best_trade": 0.0, "worst_trade": 0.0, "daily_pnl": {}
            }
        }
        if os.path.exists(HISTORY_FILE):
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
        return default

    def save_history(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.state, f, indent=2)

    def fetch_data(self, symbol, limit=120):
        if self.mock:
            return self._synthetic(symbol, limit)
        try:
            o = self.exchange.fetch_ohlcv(symbol, "5m", limit=limit)
            o15 = self.exchange.fetch_ohlcv(symbol, "15m", limit=60)
            a, a15 = np.array(o), np.array(o15)
            return a[:,1], a[:,2], a[:,3], a[:,4], a[:,5], a15[:,4]
        except Exception as e:
            log.warning(f"Fetch failed {symbol}: {e}")
            return self._synthetic(symbol, limit)

    def _synthetic(self, symbol, limit):
        np.random.seed(int(time.time() / 60) + sum(ord(c) for c in symbol))
        bases = {"BTC":64000,"ETH":1750,"SOL":145,"BNB":580,"XRP":0.52,"DOGE":0.12}
        base  = next((v for k,v in bases.items() if k in symbol), 100)
        closes = base + np.cumsum(np.random.normal(0, base*0.0004, limit))
        closes = np.maximum(closes, base*0.5)
        highs  = closes + np.abs(np.random.normal(0, base*0.001, limit))
        lows   = closes - np.abs(np.random.normal(0, base*0.001, limit))
        opens  = closes - np.random.normal(0, base*0.0003, limit)
        vols   = np.random.uniform(200, 1500, limit)
        if np.random.random() < 0.45:
            idx = np.random.randint(limit-5, limit)
            vols[idx]   *= 2.3
            closes[idx] -= base * 0.006
        return opens, highs, lows, closes, vols, closes

    def score_signal(self, opens, highs, lows, closes, vols, closes15):
        score, direction, reasons = 0, None, []
        price   = closes[-1]
        vol_avg = np.mean(vols[-15:-1])
        vol_now = vols[-1]
        vol_ratio = vol_now / (vol_avg + 1e-9)
        if vol_ratio > VOL_MULT:
            score += 2
            reasons.append(f"VOL {vol_ratio:.1f}x")
        r = rsi(closes, RSI_PERIOD)
        if r < 35:
            score += 2; direction = "buy";  reasons.append(f"RSI {r:.1f}")
        elif r > 65:
            score += 2; direction = "sell"; reasons.append(f"RSI {r:.1f}")
        upper, mid, lower = bollinger(closes, BB_PERIOD, BB_STD)
        if price <= lower:
            score += 2; direction = "buy";  reasons.append("BB-Low")
        elif price >= upper:
            score += 2; direction = "sell"; reasons.append("BB-High")
        ef = ema(closes, EMA_FAST)[-1]
        es = ema(closes, EMA_SLOW)[-1]
        if direction == "buy"  and ef > es: score += 2; reasons.append("EMA UP")
        elif direction == "sell" and ef < es: score += 2; reasons.append("EMA DN")
        if len(closes15) >= EMA_SLOW:
            e15f = ema(closes15, EMA_FAST)[-1]
            e15s = ema(closes15, EMA_SLOW)[-1]
            if direction == "buy"  and e15f > e15s: score += 2; reasons.append("MTF UP")
            elif direction == "sell" and e15f < e15s: score += 2; reasons.append("MTF DN")
        return score, direction, r, upper, lower, reasons

    def check_positions(self, symbol, closes):
        if symbol not in self.state["active_positions"]:
            return
        pos    = self.state["active_positions"][symbol]
        side   = pos["side"]
        entry  = pos["entry"]
        tp     = pos["tp"]
        sl     = pos["sl"]
        margin = pos.get("margin", MARGIN_PER_TRADE)
        qty    = (margin * LEVERAGE) / entry
        for price in closes:
            hit, reason, exit_p = False, "", price
            if side == "buy":
                if price >= tp: hit, reason, exit_p = True, "TP", tp
                elif price <= sl: hit, reason, exit_p = True, "SL", sl
            else:
                if price <= tp: hit, reason, exit_p = True, "TP", tp
                elif price >= sl: hit, reason, exit_p = True, "SL", sl
            if hit:
                pnl = ((exit_p-entry)*qty) if side=="buy" else ((entry-exit_p)*qty)
                pnl = max(min(pnl, margin*0.5), -margin*0.25)
                self.state["total_pnl"] += pnl
                st  = self.state["stats"]
                st["total_trades"] += 1
                if pnl > 0:
                    st["wins"] += 1
                    st["best_trade"] = max(st["best_trade"], pnl)
                else:
                    st["losses"] += 1
                    st["worst_trade"] = min(st["worst_trade"], pnl)
                today = ist_now().strftime("%Y-%m-%d")
                st["daily_pnl"][today] = round(st["daily_pnl"].get(today, 0)+pnl, 2)
                self.state["trades"].append({
                    "time": ist_short(), "symbol": symbol,
                    "side": side.upper(), "entry": round(entry, 6),
                    "exit": round(exit_p, 6), "pnl": round(pnl, 2),
                    "result": reason, "score": pos.get("score", 0),
                    "margin": margin,
                })
                del self.state["active_positions"][symbol]
                log.info(f"{'OK' if pnl>0 else 'LOSS'} {symbol} {side.upper()} | PnL:${pnl:.2f} | {reason}")
                self.save_history()
                break

    def run(self):
        log.info("WhaleTrader Pro — Start")
        session = is_active_session()
        for symbol in SYMBOLS:
            try:
                result = self.fetch_data(symbol)
                if result is None: continue
                opens, highs, lows, closes, vols, closes15 = result
                self.check_positions(symbol, closes)
                price = round(closes[-1], 6)
                self.state["last_prices"][symbol] = price
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
                score, direction, r, upper, lower, reasons = self.score_signal(
                    opens, highs, lows, closes, vols, closes15)
                log.info(f"{symbol} | {price} | RSI:{r:.1f} | Score:{score}/10")
                if session and direction and score >= MIN_SCORE:
                    margin = MARGIN_PER_TRADE if score >= 9 else 60.0
                    av     = atr_val(highs, lows, closes, ATR_PERIOD)
                    if direction == "buy":
                        tp = round(price + av * TP_MULT, 6)
                        sl = round(price - av * SL_MULT, 6)
                    else:
                        tp = round(price - av * TP_MULT, 6)
                        sl = round(price + av * SL_MULT, 6)
                    self.state["active_positions"][symbol] = {
                        "side": direction, "entry": price,
                        "tp": tp, "sl": sl, "margin": margin,
                        "score": score, "reasons": reasons, "time": ist_short(),
                    }
                    self.save_history()
                    log.info(f"NEW TRADE: {symbol} {direction.upper()} Score:{score} Margin:${margin}")
                self.dashboard.append({
                    "symbol": symbol, "price": price,
                    "signal": f"{direction.upper()} {score}/10" if direction else "SCANNING",
                    "score": score, "entry": price,
                    "tp": None, "sl": None, "reasons": reasons,
                })
            except Exception as e:
                log.error(f"Error {symbol}: {e}")
        self.save_history()
        self.build_dashboard()

    def build_dashboard(self):
        state  = self.state
        stats  = state.get("stats", {})
        trades = state.get("trades", [])
        total_pnl  = round(state.get("total_pnl", 0.0), 2)
        wallet     = round(INITIAL_CAPITAL + total_pnl, 2)
        total_t    = stats.get("total_trades", 0)
        wins       = stats.get("wins", 0)
        losses     = stats.get("losses", 0)
        win_rate   = round(wins/total_t*100, 1) if total_t > 0 else 0
        best       = round(stats.get("best_trade", 0), 2)
        worst      = round(stats.get("worst_trade", 0), 2)
        pnl_pct    = round((total_pnl / INITIAL_CAPITAL) * 100, 2)
        daily_pnl  = stats.get("daily_pnl", {})
        sorted_days = sorted(daily_pnl.items())[-14:]
        chart_labels = json.dumps([d[0][5:] for d in sorted_days])
        chart_values = json.dumps([d[1] for d in sorted_days])
        chart_colors = json.dumps(["rgba(0,230,118,0.85)" if d[1]>=0 else "rgba(255,23,68,0.85)" for d in sorted_days])

        monitor_rows = ""
        for d in self.dashboard:
            sym   = d["symbol"]
            clean = sym.replace("/","").lower()
            sig   = d.get("signal","SCANNING")
            score = d.get("score", 0)
            reasons = ", ".join(d.get("reasons",[])) or "Waiting..."
            tp_val  = f'${d["tp"]}' if d.get("tp") else "—"
            sl_val  = f'${d["sl"]}' if d.get("sl") else "—"
            if "HOLD" in sig:   pill_cls = "pill-hold"
            elif "BUY" in sig:  pill_cls = "pill-buy"
            elif "SELL" in sig: pill_cls = "pill-sell"
            else:               pill_cls = "pill-scan"
            dots = "".join(f'<span class="dot{"f" if i<score else ""}"></span>' for i in range(10))
            monitor_rows += f"""<tr>
<td class="sym-cell">{sym}</td>
<td><span id="p-{clean}" class="live-price">—</span></td>
<td><span id="c-{clean}" class="live-chg">—</span></td>
<td><span class="{pill_cls}">{sig}</span></td>
<td><div class="dots">{dots}</div><span class="score-num">{score}/10</span></td>
<td class="reason-cell">{reasons}</td>
<td class="tp-cell">{tp_val}</td>
<td class="sl-cell">{sl_val}</td>
</tr>"""

        history_rows = ""
        for t in reversed(trades[-50:]):
            pnl_v = float(t["pnl"])
            pc    = "pos" if pnl_v >= 0 else "neg"
            icon  = "🎯" if t.get("result","")=="TP" else "🛑"
            history_rows += f"""<tr>
<td class="muted">{t["time"]}</td>
<td class="sym-cell">{t["symbol"]}</td>
<td><span class="{"pill-buy" if t["side"]=="BUY" else "pill-sell"}">{t["side"]}</span></td>
<td class="mono">${t["entry"]}</td>
<td class="mono">${t["exit"]}</td>
<td>{icon} {t.get("result","—")}</td>
<td class="mono {pc}">{("+" if pnl_v>=0 else "")}${pnl_v}</td>
<td class="muted">{t.get("score","—")}/10</td>
<td class="muted">${t.get("margin",100)}</td>
</tr>"""

        if not history_rows:
            history_rows = '<tr><td colspan="9" class="empty-row">Scanning markets — no trades yet</td></tr>'

        pnl_color  = "#00e676" if total_pnl >= 0 else "#ff1744"
        pnl_prefix = "+" if total_pnl >= 0 else ""
        wr_color   = "#00e676" if win_rate >= 55 else "#ffab00" if win_rate >= 45 else "#ff1744"
        now_str    = ist_str()

        try:
            with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
                html = f.read()
        except FileNotFoundError:
            log.error("template.html not found!")
            return

        html = html.replace("{{WALLET}}", str(wallet))
        html = html.replace("{{TOTAL_PNL}}", f"{pnl_prefix}${total_pnl}")
        html = html.replace("{{PNL_COLOR}}", pnl_color)
        html = html.replace("{{PNL_PCT}}", f"{pnl_prefix}{pnl_pct}%")
        html = html.replace("{{WIN_RATE}}", f"{win_rate}%")
        html = html.replace("{{WR_COLOR}}", wr_color)
        html = html.replace("{{WINS}}", str(wins))
        html = html.replace("{{LOSSES}}", str(losses))
        html = html.replace("{{TOTAL_TRADES}}", str(total_t))
        html = html.replace("{{BEST_TRADE}}", f"+${best}")
        html = html.replace("{{WORST_TRADE}}", f"${worst}")
        html = html.replace("{{NOW_STR}}", now_str)
        html = html.replace("{{MONITOR_ROWS}}", monitor_rows)
        html = html.replace("{{HISTORY_ROWS}}", history_rows)
        html = html.replace("{{CHART_LABELS}}", chart_labels)
        html = html.replace("{{CHART_VALUES}}", chart_values)
        html = html.replace("{{CHART_COLORS}}", chart_colors)

        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Dashboard saved to index.html")


if __name__ == "__main__":
    engine = WhaleEngine()
    engine.run()
            
