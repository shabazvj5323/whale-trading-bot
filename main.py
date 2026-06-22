import time
import logging
import os
import json
import ccxt
import numpy as np
from datetime import datetime, timedelta

# --- WAHI PURANI STRATEGY ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("WhaleTrader_Pro_Quant")

class WhaleQuantEngine:
    def __init__(self):
        self.symbols = ["BTC/USDT", "ETH/USDT", "PAXG/USDT"]
        self.leverage = 10 
        self.initial_capital = 1000.0  
        self.margin_per_trade = 100.0  
        self.volume_multiplier = 1.5
        self.rsi_period = 9
        self.bb_period = 20
        self.bb_std_dev = 1.8
        self.atr_period = 10
        self.history_file = "history.json"
        self.state = self.load_and_clean_history()
        self.dashboard_data = []
        api_key = os.getenv("BINANCE_API_KEY")
        secret_key = os.getenv("BINANCE_SECRET_KEY")
        self.mock_mode = not (api_key and secret_key)
        if not self.mock_mode:
            self.exchange = ccxt.binance({"apiKey": api_key, "secret": secret_key, "enableRateLimit": True, "options": {"defaultType": "future"}})
            self.exchange.set_sandbox_mode(True)

    def get_ist_time_str(self):
        return (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %I:%M:%S %p")

    def load_and_clean_history(self):
        default = {"total_pnl": 0.0, "active_positions": {}, "trades": [], "last_prices": {}}
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f: return json.load(f)
            except: return default
        return default

    def save_history(self):
        with open(self.history_file, "w") as f: json.dump(self.state, f, indent=4)

    def fetch_market_data(self, symbol, limit=100):
        try:
            if self.mock_mode:
                np.random.seed(int(time.time()) + sum(ord(c) for c in symbol))
                base = 64100.0 if "BTC" in symbol else (1750.0 if "ETH" in symbol else 4200.0)
                closes = base + np.cumsum(np.random.normal(0, base * 0.0003, limit))
                return closes-1, closes+1, closes-2, closes, np.random.uniform(500, 2000, limit)
            ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=limit)
            ohlcv_np = np.array(ohlcv)
            return ohlcv_np[:, 1], ohlcv_np[:, 2], ohlcv_np[:, 3], ohlcv_np[:, 4], ohlcv_np[:, 5]
        except: return None

    def calculate_indicators(self, closes, highs, lows):
        rsi = 50.0 
        sma = np.mean(closes[-20:])
        std = np.std(closes[-20:])
        return rsi, sma + (1.8 * std), sma, sma - (1.8 * std), 10.0

    def evaluate_signals(self, symbol, opens, highs, lows, closes, volumes):
        rsi, up, sma, low, atr = self.calculate_indicators(closes, highs, lows)
        curr = closes[-1]
        
        # WAHI SCALPING STRATEGY LOGIC
        if symbol not in self.state["active_positions"]:
            if curr <= low: 
                self.state["active_positions"][symbol] = {"side": "buy", "entry": curr, "tp": curr*1.015, "sl": curr*0.995}
                self.dashboard_data.append({"symbol": symbol, "signal": "SCALPING BUY", "rsi": round(rsi, 2), "entry": round(curr, 2)})
            elif curr >= up: 
                self.state["active_positions"][symbol] = {"side": "sell", "entry": curr, "tp": curr*0.985, "sl": curr*1.005}
                self.dashboard_data.append({"symbol": symbol, "signal": "SCALPING SELL", "rsi": round(rsi, 2), "entry": round(curr, 2)})
        else:
            self.dashboard_data.append({"symbol": symbol, "signal": f"WAITING {self.state['active_positions'][symbol]['side'].upper()}", "rsi": round(rsi, 2), "entry": self.state['active_positions'][symbol]['entry']})
            pos = self.state["active_positions"][symbol]
            if (pos["side"] == "buy" and curr >= pos["tp"]) or (pos["side"] == "sell" and curr <= pos["tp"]):
                self.state["trades"].append({"time": self.get_ist_time_str(), "symbol": symbol, "side": pos["side"].upper(), "entry": pos["entry"], "exit": curr, "pnl": 5.0, "result": "TP"})
                del self.state["active_positions"][symbol]
        self.save_history()

    def generate_html_dashboard(self):
        now_str = self.get_ist_time_str()
        rows = "".join([f"<tr><td>{d['symbol']}</td><td>{d['signal']}</td><td>{d['rsi']}</td><td>{d['entry']}</td></tr>" for d in self.dashboard_data])
        hist = "".join([f"<tr><td>{t['time']}</td><td>{t['symbol']}</td><td>{t['pnl']}</td></tr>" for t in self.state["trades"][-5:]])
        
        # DASHBOARD MEIN TIMINGS ADDED
        html = f"""<!DOCTYPE html>
<html>
<head>
<style>
    body {{ background: #08090c; color: white; font-family: monospace; padding: 20px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th, td {{ border: 1px solid #333; padding: 8px; text-align: left; }}
</style>
</head>
<body>
    <h1>WhaleTrader Pro</h1>
    <div style="background:#111; padding:10px; border-radius:5px;">
        <div>Live Clock: <span id="clock" style="color:#00ff00;"></span></div>
        <div>Last Sync Time: {now_str}</div>
        <div id="timer" style="color: yellow; font-size: 20px; font-weight:bold;">Next Sync In: 15:00</div>
    </div>
    <table><tr><th>Asset</th><th>Signal</th><th>RSI</th><th>Entry</th></tr>{rows}</table>
    <h3>Settlement Log</h3>
    <table><tr><th>Time</th><th>Asset</th><th>P&L</th></tr>{hist}</table>
    <script>
        setInterval(()=>{{ document.getElementById('clock').innerText = new Date().toLocaleTimeString(); }}, 1000);
        let t = 900;
        setInterval(()=>{{
            t--;
            let m = Math.floor(t/60), s = t%60;
            document.getElementById('timer').innerText = "Next Sync In: " + m + ":" + (s<10?'0':'') + s;
            if(t <= 0) location.reload();
        }}, 1000);
    </script>
</body>
</html>"""
        with open("index.html", "w") as f: f.write(html)

    def run_pipeline(self):
        for s in self.symbols:
            data = self.fetch_market_data(s)
            if data: self.evaluate_signals(s, *data)
        self.generate_html_dashboard()

if __name__ == "__main__":
    WhaleQuantEngine().run_pipeline()
    
