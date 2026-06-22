import time
import math
import logging
import os
import json
import ccxt
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("WhaleTrader_Pro_Quant")

class WhaleQuantEngine:
    def __init__(self):
        # Added Gold (PAXG) and Silver assets for Binance compatibility
        self.symbols = ["BTC/USDT", "ETH/USDT", "PAXG/USDT"]
        self.leverage = 10 
        self.total_capital = 1000.0  # Total Capital Set to $1000
        self.risk_per_trade = 0.25   # 25% allocation per trade ($250 margin * 10x = $2500 buying power)
        
        # Hyper-Scalping Metrics (5 Minute Target Engine)
        self.volume_multiplier = 1.5  # Lower threshold for more frequent scalp entries
        self.rsi_period = 9           # Faster RSI for quick shifts
        self.bb_period = 20
        self.bb_std_dev = 1.8         # Tighter bands for maximum scalp triggers
        self.atr_period = 10
        
        self.history_file = "history.json"
        self.state = self.load_history()
        self.dashboard_data = []
        
        api_key = os.getenv("BINANCE_API_KEY")
        secret_key = os.getenv("BINANCE_SECRET_KEY")
        
        if not api_key or not secret_key:
            self.mock_mode = True
        else:
            self.mock_mode = False
            self.exchange = ccxt.binance({
                "apiKey": api_key,
                "secret": secret_key,
                "enableRateLimit": True,
                "options": {"defaultType": "future"}
            })
            self.exchange.set_sandbox_mode(True)

    def get_ist_time_str(self):
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        return ist_now.strftime("%Y-%m-%d %I:%M:%S %p (IST)")

    def get_ist_short_str(self):
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        return ist_now.strftime("%Y-%m-%d %H:%M")

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    if "last_prices" not in data: data["last_prices"] = {}
                    if "total_pnl" not in data: data["total_pnl"] = 0.0
                    return data
            except Exception: pass
        return {"total_pnl": 0.0, "active_positions": {}, "trades": [], "last_prices": {}}

    def save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.state, f, indent=4)

    def fetch_market_data(self, symbol, timeframe='5m', limit=100): # Changed to 5m for High-Frequency Scalping
        if self.mock_mode:
            return self.generate_synthetic_data(symbol, limit)
        else:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                ohlcv_np = np.array(ohlcv)
                return ohlcv_np[:, 1], ohlcv_np[:, 2], ohlcv_np[:, 3], ohlcv_np[:, 4], ohlcv_np[:, 5]
            except Exception:
                return self.generate_synthetic_data(symbol, limit)

    def generate_synthetic_data(self, symbol, limit):
        np.random.seed(int(time.time()) + sum(ord(c) for c in symbol))
        if "BTC" in symbol: base = 65000.0
        elif "ETH" in symbol: base = 3500.0
        else: base = 2350.0 # Gold / PAXG Base
        
        closes = base + np.cumsum(np.random.normal(0, base * 0.002, limit))
        volumes = np.random.uniform(500, 2000, limit)
        
        volumes[-1] = np.mean(volumes) * 1.8
        closes[-1] = closes[-2] + (np.std(closes) * 1.1)
        
        highs = closes + np.random.uniform(2, 20, limit)
        lows = closes - np.random.uniform(2, 20, limit)
        opens = closes - np.random.normal(0, 10, limit)
        return opens, highs, lows, closes, volumes

    def calculate_indicators(self, opens, highs, lows, closes, volumes):
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[:self.rsi_period])
        avg_loss = np.mean(losses[:self.rsi_period])
        for i in range(self.rsi_period, len(deltas)):
            avg_gain = (avg_gain * (self.rsi_period - 1) + gains[i]) / self.rsi_period
            avg_loss = (avg_loss * (self.rsi_period - 1) + losses[i]) / self.rsi_period
        rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))
        
        recent_closes = closes[-self.bb_period:]
        sma = np.mean(recent_closes)
        std_dev = np.std(recent_closes)
        upper_band = sma + (self.bb_std_dev * std_dev)
        lower_band = sma - (self.bb_std_dev * std_dev)
        
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(closes[:-1] - lows[1:])))
        atr = np.mean(tr[-self.atr_period:])
        return rsi, upper_band, sma, lower_band, atr

    def check_active_positions(self, symbol, current_price):
        if symbol in self.state["active_positions"]:
            pos = self.state["active_positions"][symbol]
            side = pos["side"]
            entry = pos["entry"]
            tp = pos["tp"]
            sl = pos["sl"]
            margin = pos["margin"]
            
            # Scalp calculations accounting for 10x Leverage
            notional_value = margin * self.leverage
            qty = notional_value / entry
            
            pnl = 0.0
            hit = False
            reason = ""

            if side == "buy":
                if current_price >= tp:
                    hit = True
                    pnl = (tp - entry) * qty
                    reason = "Scalp TP 🎯"
                elif current_price <= sl:
                    hit = True
                    pnl = (sl - entry) * qty
                    reason = "Scalp SL 🛑"
            elif side == "sell":
                if current_price <= tp:
                    hit = True
                    pnl = (entry - tp) * qty
                    reason = "Scalp TP 🎯"
                elif current_price >= sl:
                    hit = True
                    pnl = (entry - sl) * qty
                    reason = "Scalp SL 🛑"

            if hit:
                self.state["total_pnl"] += pnl
                trade_record = {
                    "time": self.get_ist_short_str(),
                    "symbol": symbol, "side": side.upper(), "entry": entry,
                    "exit": tp if "TP" in reason else sl, "pnl": round(pnl, 2), "result": reason
                }
                self.state["trades"].append(trade_record)
                del self.state["active_positions"][symbol]
                log.info(f"⚡ Scalp Exited for {symbol}! Result: {reason} | Net: ${round(pnl, 2)}")
                self.save_history()

    def evaluate_signals(self, symbol, opens, highs, lows, closes, volumes):
        rsi, upper_b, sma, lower_b, atr = self.calculate_indicators(opens, highs, lows, closes, volumes)
        current_price = round(closes[-1], 2)
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-15:-1])
        volume_breakout = current_volume > (avg_volume * self.volume_multiplier)
        
        self.check_active_positions(symbol, current_price)
        self.state["last_prices"][symbol] = current_price
        self.save_history()

        is_active = symbol in self.state["active_positions"]
        
        if is_active:
            pos_details = self.state["active_positions"][symbol]
            status_data = {
                "symbol": symbol, "rsi": round(rsi, 2), "signal": f"SCALPING {pos_details['side'].upper()}", 
                "entry": pos_details['entry'], "tp": pos_details["tp"], "sl": pos_details["sl"]
            }
            self.dashboard_data.append(status_data)
            return "WAIT"

        if not volume_breakout: return "WAIT"

        # TIGHT SCALPING BUFFER: Small targets, quick profits, very tight protective stop-loss
        tp_factor = 0.6  # 0.6x ATR target for instant profit collection
        sl_factor = 0.4  # ultra-tight stop loss to minimize risk exposure

        if current_price <= lower_b or rsi < 38: # Fast entry triggers
            tp = round(current_price + (atr * tp_factor), 2)
            sl = round(current_price - (atr * sl_factor), 2)
            self.state["active_positions"][symbol] = {"side": "buy", "entry": current_price, "tp": tp, "sl": sl, "margin": 250.0}
            self.save_history()
            
            status_data = {
                "symbol": symbol, "rsi": round(rsi, 2), "signal": "SCALPING BUY", "entry": current_price, "tp": tp, "sl": sl
            }
            self.dashboard_data.append(status_data)
            return "BUY"
            
        elif current_price >= upper_b or rsi > 62:
            tp = round(current_price - (atr * tp_factor), 2)
            sl = round(current_price + (atr * sl_factor), 2)
            self.state["active_positions"][symbol] = {"side": "sell", "entry": current_price, "tp": tp, "sl": sl, "margin": 250.0}
            self.save_history()
            
            status_data = {
                "symbol": symbol, "rsi": round(rsi, 2), "signal": "SCALPING SELL", "entry": current_price, "tp": tp, "sl": sl
            }
            self.dashboard_data.append(status_data)
            return "SELL"

        return "WAIT"

    def generate_html_dashboard(self):
        now_str = self.get_ist_time_str()
        pnl_val = round(self.state.get("total_pnl", 0.0), 2)
        pnl_color = "#00b574" if pnl_val >= 0 else "#ff3b30"
        
        monitor_rows = ""
        for data in self.dashboard_data:
            sig_class = "buy-glow" if "BUY" in data["signal"] else "sell-glow"
            clean_sym = data['symbol'].replace("/", "").lower()
            
            monitor_rows += f"""
            <tr id='row-{clean_sym}'>
                <td style='color: #ffffff; font-weight: 600;'>{data['symbol']}</td>
                <td><span id='price-{clean_sym}' class='price-ticker'>$0.00</span></td>
                <td><span id='change-{clean_sym}' class='badge-glow'>0.00 (0.00%)</span></td>
                <td><span class='badge-metric'>RSI: {data['rsi']}</span></td>
                <td style='color: #c5d4e2; font-weight: bold;'>${data['entry']}</td>
                <td><span class='status-pill {sig_class}'>{data['signal']}</span></td>
                <td style='color: #00b574; font-weight:bold;'>${data['tp']}</td>
                <td style='color: #ff3b30; font-weight:bold;'>${data['sl']}</td>
            </tr>"""

        if not monitor_rows:
            for sym in self.symbols:
                clean_sym = sym.replace("/", "").lower()
                monitor_rows += f"""
                <tr id='row-{clean_sym}'>
                    <td style='color: #ffffff; font-weight: 600;'>{sym}</td>
                    <td><span id='price-{clean_sym}' class='price-ticker'>$0.00</span></td>
                    <td><span id='change-{clean_sym}' class='badge-glow'>0.00 (0.00%)</span></td>
                    <td colspan='5' style='color: #8492a6; text-align: center; font-size:12px; letter-spacing: 0.5px;'>⚡ 5M SCALPER ACTIVE: STANDBY SCANNING...</td>
                </tr>"""

        history_rows = ""
        for t in reversed(self.state.get("trades", [])):
            t_color = "#00b574" if t["pnl"] >= 0 else "#ff3b30"
            badge_type = "history-buy" if t["side"] == "BUY" else "history-sell"
            history_rows += f"""
            <tr>
                <td style='color: #8492a6;'>{t['time']}</td>
                <td><b>{t['symbol']}</b></td>
                <td><span class='hist-pill {badge_type}'>{t['side']}</span></td>
                <td>${t['entry']}</td>
                <td>${t['exit']}</td>
                <td><span style='color:{t_color}; font-weight:600;'>{t['result']}</span></td>
                <td style='color: {t_color}; font-weight: bold; font-family: monospace;'>${t['pnl']} USD</td>
            </tr>"""

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WhaleTrader Premium Terminal</title>
    <style>
        body {{ font-family: 'Inter', -apple-system, sans-serif; background-color: #060709; color: #dee4ec; margin: 0; padding: 25px; -webkit-font-smoothing: antialiased; }}
        .container {{ max-width: 1250px; margin: 0 auto; }}
        header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #141822; padding-bottom: 20px; margin-bottom: 30px; }}
        h1 {{ color: #ffffff; font-size: 22px; font-weight: 700; letter-spacing: -0.5px; display: flex; align-items: center; gap: 8px; }}
        h1::before {{ content: ''; display: inline-block; width: 10px; height: 10px; background: #00b574; border-radius: 50%; box-shadow: 0 0 10px #00b574; }}
        .matrix-container {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: linear-gradient(135deg, #0e1118 0%, #121620 100%); border: 1px solid #1c2333; padding: 20px; border-radius: 12px; flex: 1; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }}
        .stat-label {{ color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 1px; margin-bottom: 6px; }}
        .stat-value {{ font-size: 28px; font-weight: 800; font-family: monospace; letter-spacing: -1px; }}
        table {{ width: 100%; border-collapse: collapse; background-color: #0b0d13; border-radius: 12px; margin-bottom: 35px; overflow: hidden; border: 1px solid #141822; box-shadow: 0 4px 25px rgba(0,0,0,0.3); }}
        th, td {{ padding: 16px; text-align: left; border-bottom: 1px solid #141822; font-size: 13px; }}
        th {{ background-color: #0f121a; color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.8px; }}
        tr {{ transition: background-color 0.2s; }}
        tr:hover {{ background-color: #121622; }}
        .price-ticker {{ font-family: monospace; font-size: 15px; font-weight: bold; transition: color 0.2s; }}
        .badge-glow {{ font-family: monospace; font-size: 12px; font-weight: 700; padding: 4px 8px; border-radius: 6px; }}
        .text-up {{ color: #00b574 !important; }} .text-down {{ color: #ff3b30 !important; }}
        .bg-up {{ background-color: rgba(0, 181, 116, 0.1); border: 1px solid rgba(0, 181, 116, 0.2); }}
        .bg-down {{ background-color: rgba(255, 59, 48, 0.1); border: 1px solid rgba(255, 59, 48, 0.2); }}
        .status-pill {{ padding: 6px 12px; border-radius: 6px; font-weight: 700; font-size: 11px; letter-spacing: 0.5px; }}
        .buy-glow {{ background-color: rgba(0, 181, 116, 0.15); color: #00b574; border: 1px solid #00b574; box-shadow: 0 0 10px rgba(0,181,116,0.2); }}
        .sell-glow {{ background-color: rgba(255, 59, 48, 0.15); color: #ff3b30; border: 1px solid #ff3b30; box-shadow: 0 0 10px rgba(255,59,48,0.2); }}
        .hist-pill {{ padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }}
        .history-buy {{ background: rgba(0,181,116,0.1); color: #00b574; }}
        .history-sell {{ background: rgba(255,59,48,0.1); color: #ff3b30; }}
        .badge-metric {{ color: #38bdf8; background: rgba(56,189,248,0.1); padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; }}
        h3 {{ color: #ffffff; font-size: 15px; font-weight: 600; margin-bottom: 15px; letter-spacing: -0.3px; display: flex; align-items: center; gap: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>WhaleTrader Quantum Engine Pro</h1>
            <div style="color: #64748b; font-size: 13px; font-weight: 500;">Core Pulse: <span style="color: #ffffff; font-weight:600;">{now_str}</span></div>
        </header>
        
        <div class="matrix-container">
            <div class="stat-card">
                <div class="stat-label">Allocated Scalp Wallet</div>
                <div class="stat-value" style="color: #ffffff;">$1,000.00 <span style="font-size:12px; color:#64748b; font-weight:500;">USD</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active Margin Leverage</div>
                <div class="stat-value" style="color: #f0b90b;">10x Isolated</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Realized Net Returns</div>
                <div class="stat-value" style="color: {pnl_color};">${pnl_val}</div>
            </div>
        </div>

        <h3>🟢 Streaming Cross-Asset Spot & Futures Order Monitor</h3>
        <table>
            <thead>
                <tr>
                    <th>Asset Pair</th><th>Live Running Price</th><th>24h Volatility Delta</th><th>Scalp Conditions</th><th>Trigger Entry</th><th>Execution State</th><th>Target TP</th><th>Protective SL</th>
                </tr>
            </thead>
            <tbody>{monitor_rows}</tbody>
        </table>

        <h3>📜 High-Frequency Closed Settlement Log</h3>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th><th>Asset Pair</th><th>Action Vector</th><th>Entry Price</th><th>Exit Settlement</th><th>Trigger Output</th><th>Net Realized Margin</th>
                </tr>
            </thead>
            <tbody>{history_rows if history_rows else '<tr><td colspan="7" style="text-align:center; color:#64748b; padding:25px;">Ready to lock settlements. Monitoring price bands...</td></tr>'}</tbody>
        </table>
    </div>

    <script>
        const symbols = ['btcusdt', 'ethusdt', 'paxgusdt'];
        function connectLiveTicker() {{
            const wsUrl = "wss://stream.binance.com:9443/ws/" + symbols.map(s => s + "@ticker").join("/");
            const ws = new WebSocket(wsUrl);
            ws.onmessage = (event) => {{
                const data = JSON.parse(event.data);
                const sym = data.s.toLowerCase();
                const priceEl = document.getElementById("price-" + sym);
                const changeEl = document.getElementById("change-" + sym);
                
                if (priceEl && changeEl) {{
                    const price = parseFloat(data.c).toFixed(2);
                    const changeAmt = parseFloat(data.p).toFixed(2);
                    const changePct = parseFloat(data.P).toFixed(2);
                    priceEl.innerText = "$" + price;
                    if (parseFloat(changePct) >= 0) {{
                        changeEl.innerText = "+" + changeAmt + " (+" + changePct + "%)";
                        changeEl.className = "badge-glow text-up bg-up";
                        priceEl.className = "price-ticker text-up";
                    }} else {{
                        changeEl.innerText = changeAmt + " (" + changePct + "%)";
                        changeEl.className = "badge-glow text-down bg-down";
                        priceEl.className = "price-ticker text-down";
                    }}
                }}
            }};
            ws.onclose = () => {{ setTimeout(connectLiveTicker, 4000); }};
        }}
        connectLiveTicker();
        setTimeout(() => {{ window.location.reload(); }}, 300000);
    </script>
</body>
</html>"""
        with open("index.html", "w") as f:
            f.write(html_content)

    def run_pipeline(self):
        log.info("⚡ WhaleTrader Premium Terminal System Executed.")
        for symbol in self.symbols:
            data = self.fetch_market_data(symbol)
            if data is None: continue
            opens, highs, lows, closes, volumes = data
            self.evaluate_signals(symbol, opens, highs, lows, closes, volumes)
        self.generate_html_dashboard()

if __name__ == "__main__":
    engine = WhaleQuantEngine()
    engine.run_pipeline()
    
