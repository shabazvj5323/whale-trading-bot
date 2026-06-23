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


TEMPLATE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WhaleTrader Pro</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
:root{
  --bg0:#05070f;--bg1:#080c17;--bg2:#0d1322;--bg3:#111827;
  --border:#1a2540;--border2:#243050;
  --text:#d1daf0;--muted:#4a6080;--muted2:#2a3a55;
  --green:#00e676;--green2:#00c853;--red:#ff1744;
  --blue:#2979ff;--amber:#ffab00;--purple:#7c4dff;
  --font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg0);color:var(--text);font-family:var(--font);font-size:13px;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg1);}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px;}
nav{background:rgba(8,12,23,0.97);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);padding:0 20px;height:56px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;}
.nav-left{display:flex;align-items:center;gap:14px;}
.logo{display:flex;align-items:center;gap:10px;font-size:15px;font-weight:800;color:#fff;}
.logo-icon{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#2979ff,#00e676);display:flex;align-items:center;justify-content:center;font-size:16px;}
.badge-live{background:rgba(0,230,118,.12);color:var(--green);border:1px solid rgba(0,230,118,.25);font-size:10px;font-weight:700;padding:3px 9px;border-radius:20px;display:flex;align-items:center;gap:5px;}
.pulse-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;}
@keyframes pulse{ 0%{ box-shadow:0 0 0 0 rgba(0,230,118,.5); } 70%{ box-shadow:0 0 0 6px rgba(0,230,118,0); } 100%{ box-shadow:0 0 0 0 rgba(0,230,118,0); } }
.nav-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.nav-box{font-family:var(--mono);font-size:10px;color:var(--muted);background:var(--bg2);border:1px solid var(--border);padding:5px 10px;border-radius:6px;white-space:nowrap;}
.session-badge{font-size:10px;font-weight:700;padding:4px 10px;border-radius:20px;white-space:nowrap;}
.session-on{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.2);}
.session-off{background:rgba(255,171,0,.1);color:var(--amber);border:1px solid rgba(255,171,0,.2);}
.main{max-width:1400px;margin:0 auto;padding:20px 14px;}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:11px;margin-bottom:24px;}
.stat-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:16px;position:relative;overflow:hidden;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.c1::before{background:linear-gradient(90deg,var(--blue),var(--purple));}
.c2::before{background:linear-gradient(90deg,var(--green),var(--green2));}
.c3::before{background:linear-gradient(90deg,var(--amber),#ff6f00);}
.c4::before{background:linear-gradient(90deg,#00e676,#00bcd4);}
.c5::before{background:linear-gradient(90deg,#ff1744,#ff5722);}
.c6::before{background:linear-gradient(90deg,var(--purple),var(--blue));}
.stat-label{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:600;margin-bottom:7px;}
.stat-value{font-size:22px;font-weight:800;font-family:var(--mono);letter-spacing:-.5px;line-height:1;}
.stat-sub{font-size:11px;color:var(--muted);margin-top:5px;}
.mini-bar{background:var(--bg3);border-radius:4px;height:3px;margin-top:8px;}
.mini-fill{height:3px;border-radius:4px;}
.sec-head{display:flex;align-items:center;gap:10px;margin:24px 0 11px;}
.sec-title{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:700;white-space:nowrap;}
.sec-line{flex:1;height:1px;background:var(--border);}
.sec-tag{font-size:10px;color:var(--muted);background:var(--bg2);border:1px solid var(--border);padding:2px 8px;border-radius:10px;}
.chart-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:4px;}
.chart-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;}
.chart-title{font-size:13px;font-weight:600;}
.chart-sub{font-size:11px;color:var(--muted);}
.tbl-card{background:var(--bg1);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:4px;}
.tbl-wrap{overflow-x:auto;}
table{width:100%;border-collapse:collapse;}
thead tr{background:var(--bg2);}
th{padding:10px 13px;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:700;border-bottom:1px solid var(--border);white-space:nowrap;}
td{padding:11px 13px;border-bottom:1px solid rgba(26,37,64,.5);vertical-align:middle;}
tr:last-child td{border-bottom:none;}
tbody tr:hover td{background:rgba(255,255,255,.015);}
.sym-cell{font-weight:700;color:#fff;font-size:13px;white-space:nowrap;}
.live-price{font-family:var(--mono);font-size:13px;font-weight:600;color:#fff;}
.live-chg{font-family:var(--mono);font-size:11px;font-weight:600;padding:3px 7px;border-radius:5px;white-space:nowrap;}
.chg-up{color:var(--green);background:rgba(0,230,118,.08);}
.chg-dn{color:var(--red);background:rgba(255,23,68,.08);}
.reason-cell{color:var(--muted);font-size:11px;}
.tp-cell{color:var(--green);font-family:var(--mono);font-size:12px;}
.sl-cell{color:var(--red);font-family:var(--mono);font-size:12px;}
.mono{font-family:var(--mono);font-size:12px;}
.muted{color:var(--muted);font-size:11px;}
.pos{color:var(--green);font-weight:700;}
.neg{color:var(--red);font-weight:700;}
.empty-row{text-align:center;color:var(--muted);padding:30px;font-size:12px;}
.dots{display:flex;gap:3px;margin-bottom:3px;}
.dot{width:7px;height:7px;border-radius:50%;background:var(--border2);}
.dotf{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 5px rgba(0,230,118,.5);}
.score-num{font-size:10px;color:var(--muted);font-family:var(--mono);}
.pill-buy,.pill-sell,.pill-hold,.pill-scan{font-size:10px;font-weight:700;padding:3px 8px;border-radius:5px;text-transform:uppercase;white-space:nowrap;display:inline-block;}
.pill-buy{background:rgba(0,230,118,.1);color:var(--green);border:1px solid rgba(0,230,118,.2);}
.pill-sell{background:rgba(255,23,68,.1);color:var(--red);border:1px solid rgba(255,23,68,.2);}
.pill-hold{background:rgba(41,121,255,.1);color:var(--blue);border:1px solid rgba(41,121,255,.2);}
.pill-scan{background:rgba(255,171,0,.08);color:var(--amber);border:1px solid rgba(255,171,0,.15);}
footer{text-align:center;color:var(--muted2);font-size:10px;padding:22px;letter-spacing:.05em;}
</style>
</head>
<body>

<nav>
  <div class="nav-left">
    <div class="logo">
      <div class="logo-icon">&#x1F40B;</div>
      WhaleTrader Pro
    </div>
    <div class="badge-live"><div class="pulse-dot"></div>LIVE</div>
  </div>
  <div class="nav-right">
    <div class="nav-box">Bot sync: BOT_SYNC_TIME</div>
    <div class="session-badge" id="sess">LOADING</div>
    <div class="nav-box" id="clk">-- IST</div>
  </div>
</nav>

<div class="main">

  <div class="stat-grid">
    <div class="stat-card c1">
      <div class="stat-label">Account Equity</div>
      <div class="stat-value" style="color:#fff;">$WALLET_VAL</div>
      <div class="stat-sub">Started $1,000.00</div>
      <div class="mini-bar"><div class="mini-fill" style="width:100%;background:var(--blue);"></div></div>
    </div>
    <div class="stat-card c2">
      <div class="stat-label">Total P&amp;L</div>
      <div class="stat-value" style="color:PNL_COLOR_VAL;">TOTAL_PNL_VAL</div>
      <div class="stat-sub" style="color:PNL_COLOR_VAL;">PNL_PCT_VAL return</div>
      <div class="mini-bar"><div class="mini-fill" style="width:55%;background:PNL_COLOR_VAL;"></div></div>
    </div>
    <div class="stat-card c3">
      <div class="stat-label">Win Rate</div>
      <div class="stat-value" style="color:WR_COLOR_VAL;">WIN_RATE_VAL</div>
      <div class="stat-sub">WINS_VAL W / LOSSES_VAL L / TOTAL_T_VAL total</div>
      <div class="mini-bar"><div class="mini-fill" style="width:WIN_RATE_VAL;background:WR_COLOR_VAL;"></div></div>
    </div>
    <div class="stat-card c4">
      <div class="stat-label">Best Trade</div>
      <div class="stat-value" style="color:var(--green);">BEST_VAL</div>
      <div class="stat-sub">All time high</div>
    </div>
    <div class="stat-card c5">
      <div class="stat-label">Worst Trade</div>
      <div class="stat-value" style="color:var(--red);">WORST_VAL</div>
      <div class="stat-sub">Max drawdown</div>
    </div>
    <div class="stat-card c6">
      <div class="stat-label">Strategy</div>
      <div class="stat-value" style="color:var(--amber);">10x</div>
      <div class="stat-sub">Isolated / 6 pairs / 5m</div>
    </div>
  </div>

  <div class="sec-head"><div class="sec-title">Daily P&amp;L</div><div class="sec-line"></div><div class="sec-tag">14 days</div></div>
  <div class="chart-card">
    <div class="chart-head">
      <div class="chart-title">Realized Returns Per Day</div>
      <div class="chart-sub">IST timezone · Paper trading</div>
    </div>
    <canvas id="pnlChart" style="max-height:160px;"></canvas>
  </div>

  <div class="sec-head"><div class="sec-title">Live Asset Monitor</div><div class="sec-line"></div><div class="sec-tag">6 pairs</div></div>
  <div class="tbl-card"><div class="tbl-wrap">
    <table>
      <thead><tr><th>Pair</th><th>Live Price</th><th>24h</th><th>Signal</th><th>Score</th><th>Indicators</th><th>Take Profit</th><th>Stop Loss</th></tr></thead>
      <tbody>MONITOR_ROWS_VAL</tbody>
    </table>
  </div></div>

  <div class="sec-head"><div class="sec-title">Settlement Log</div><div class="sec-line"></div><div class="sec-tag">Last 50 trades</div></div>
  <div class="tbl-card"><div class="tbl-wrap">
    <table>
      <thead><tr><th>Time</th><th>Pair</th><th>Side</th><th>Entry</th><th>Exit</th><th>Result</th><th>P&amp;L</th><th>Score</th><th>Margin</th></tr></thead>
      <tbody>HISTORY_ROWS_VAL</tbody>
    </table>
  </div></div>

</div>

<footer>WhaleTrader Pro &middot; Auto-updates every 15 min &middot; GitHub Actions &middot; Paper Trading &middot; $1,000 Capital</footer>

<script>
var chartLabels = CHART_LABELS_VAL || [];
var chartValues = CHART_VALUES_VAL || [];
var chartColors = CHART_COLORS_VAL || [];

var syms = ['btcusdt','ethusdt','solusdt','bnbusdt','xrpusdt','dogeusdt'];
function connectWS() {
  var ws = new WebSocket("wss://stream.binance.com:9443/ws/" + syms.map(function(s){ return s+"@ticker"; }).join("/"));
  ws.onmessage = function(e) {
    var d = JSON.parse(e.data);
    var s = d.s.toLowerCase();
    var pe = document.getElementById("p-"+s);
    var ce = document.getElementById("c-"+s);
    if (!pe) return;
    var price = parseFloat(d.c);
    var chg = parseFloat(d.P);
    if (price < 1) { pe.textContent = "$" + price.toFixed(5); }
    else { pe.textContent = "$" + price.toFixed(2); }
    ce.textContent = (chg >= 0 ? "+" : "") + chg.toFixed(2) + "%";
    ce.className = "live-chg " + (chg >= 0 ? "chg-up" : "chg-dn");
    pe.style.color = chg >= 0 ? "var(--green)" : "var(--red)";
  };
  ws.onclose = function() { setTimeout(connectWS, 3000); };
}
connectWS();

function updateClock() {
  var now = new Date();
  var ist = new Date(now.getTime() + 19800000);
  var h = ist.getUTCHours();
  var m = ist.getUTCMinutes();
  var s = ist.getUTCSeconds();
  var ampm = h >= 12 ? "PM" : "AM";
  var hh = h % 12 || 12;
  function pad(n) { return n < 10 ? "0"+n : ""+n; }
  document.getElementById("clk").textContent = pad(hh)+":"+pad(m)+":"+pad(s)+" "+ampm+" IST";
  var utcH = now.getUTCHours();
  var active = (utcH >= 6 && utcH < 10) || (utcH >= 13 && utcH < 18);
  var badge = document.getElementById("sess");
  if (active) {
    badge.textContent = "SESSION ACTIVE";
    badge.className = "session-badge session-on";
  } else {
    badge.textContent = "OFF SESSION";
    badge.className = "session-badge session-off";
  }
}
setInterval(updateClock, 1000);
updateClock();

var ctx = document.getElementById("pnlChart").getContext("2d");
new Chart(ctx, {
  type: "bar",
  data: {
    labels: chartLabels,
    datasets: [{
      label: "P&L ($)",
      data: chartValues,
      backgroundColor: chartColors,
      borderRadius: 5,
      borderSkipped: false
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: function(c) {
            return (c.parsed.y >= 0 ? "+" : "") + "$" + c.parsed.y.toFixed(2);
          }
        }
      }
    },
    scales: {
      x: { grid: { color: "rgba(26,37,64,.5)" }, ticks: { color: "#4a6080", font: { size: 10 } } },
      y: { grid: { color: "rgba(26,37,64,.5)" }, ticks: { color: "#4a6080", font: { size: 10 }, callback: function(v) { return "$"+v; } } }
    }
  }
});

setTimeout(function() { window.location.reload(); }, 900000);
</script>
</body>
</html>
"""

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
        session = True  # Crypto 24/7 - no session filter
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
                if direction and score >= MIN_SCORE:
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

        html = TEMPLATE_HTML

        html = html.replace("$WALLET_VAL", str(wallet))
        html = html.replace("TOTAL_PNL_VAL", f"{pnl_prefix}${total_pnl}")
        html = html.replace("PNL_COLOR_VAL", pnl_color)
        html = html.replace("PNL_PCT_VAL", f"{pnl_prefix}{pnl_pct}%")
        html = html.replace("WIN_RATE_VAL", f"{win_rate}%")
        html = html.replace("WR_COLOR_VAL", wr_color)
        html = html.replace("WINS_VAL", str(wins))
        html = html.replace("LOSSES_VAL", str(losses))
        html = html.replace("TOTAL_T_VAL", str(total_t))
        html = html.replace("BEST_VAL", f"+${best}")
        html = html.replace("WORST_VAL", f"${worst}")
        html = html.replace("BOT_SYNC_TIME", now_str)
        html = html.replace("MONITOR_ROWS_VAL", monitor_rows)
        html = html.replace("HISTORY_ROWS_VAL", history_rows)
        html = html.replace("CHART_LABELS_VAL", chart_labels)
        html = html.replace("CHART_VALUES_VAL", chart_values)
        html = html.replace("CHART_COLORS_VAL", chart_colors)

        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
        log.info("Dashboard saved to index.html")


if __name__ == "__main__":
    engine = WhaleEngine()
    engine.run()
