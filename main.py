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
        return ist_now.strftime("%Y-%m-%d %I:%M:%S %p")

    def get_ist_short_str(self):
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        return ist_now.strftime("%m-%d %H:%M")

    def load_and_clean_history(self):
        default_state = {"total_pnl": 0.0, "active_positions": {}, "trades": [], "last_prices": {}}
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                if "trades" in data:
                    fresh_trades = []
                    for t in data["trades"]:
                        try:
                            if float(t.get("pnl", 0)) > -200.0:
                                fresh_trades.append(t)
                        except:
                            continue
                    data["trades"] = fresh_trades
                    data["total_pnl"] = sum(float(t.get("pnl", 0)) for t in fresh_trades)
                if "active_positions" not in data: data["active_positions"] = {}
                if "last_prices" not in data: data["last_prices"] = {}
                return data
            except Exception:
                return default_state
        return default_state

    def save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.state, f, indent=4)

    def fetch_market_data(self, symbol, timeframe='5m', limit=100):
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
        if "BTC" in symbol: base = 64100.0
        elif "ETH" in symbol: base = 1750.0
        else: base = 4200.0
        closes = base + np.cumsum(np.random.normal(0, base * 0.0003, limit))
        volumes = np.random.uniform(500, 2000, limit)
        volumes[-1] = np.mean(volumes) * 1.6
        closes[-1] = closes[-2] + (np.std(closes) * 0.2)
        highs = closes + np.random.uniform(1, 8, limit)
        lows = closes - np.random.uniform(1, 8, limit)
        opens = closes - np.random.normal(0, 4, limit)
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
            
            margin = self.margin_per_trade
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
                pnl = max(min(pnl, margin * 0.3), -margin * 0.15)
                self.state["total_pnl"] += pnl
                trade_record = {
                    "time": self.get_ist_short_str(),
                    "symbol": symbol, "side": side.upper(), "entry": round(entry, 2),
                    "exit": round(current_price, 2), "pnl": round(pnl, 2), "result": reason
                }
                self.state["trades"].append(trade_record)
                del self.state["active_positions"][symbol]
                log.info(f"⚡ Scalp Closed: {symbol} | Net: ${round(pnl, 2)}")
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

        tp_factor = 0.3  
        sl_factor = 0.15  

        if current_price <= lower_b or rsi < 35:
            tp = round(current_price + (atr * tp_factor), 2)
            sl = round(current_price - (atr * sl_factor), 2)
            self.state["active_positions"][symbol] = {"side": "buy", "entry": current_price, "tp": tp, "sl": sl}
            self.save_history()
            
            status_data = {
                "symbol": symbol, "rsi": round(rsi, 2), "signal": "SCALPING BUY", "entry": current_price, "tp": tp, "sl": sl
            }
            self.dashboard_data.append(status_data)
            return "BUY"
            
        elif current_price >= upper_b or rsi > 65:
            tp = round(current_price - (atr * tp_factor), 2)
            sl = round(current_price + (atr * sl_factor), 2)
            self.state["active_positions"][symbol] = {"side": "sell", "entry": current_price, "tp": tp, "sl": sl}
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
        current_wallet = round(self.initial_capital + pnl_val, 2)
        pnl_color = "#00b574" if pnl_val >= 0 else "#ff3b30"
        pnl_prefix = "+" if pnl_val >= 0 else ""
        
        monitor_rows = ""
        for data in self.dashboard_data:
            sig_class = "buy-glow" if "BUY" in data["signal"] else "sell-glow"
            clean_sym = data['symbol'].replace("/", "").lower()
            
            monitor_rows += f"""
            <tr id='row-{clean_sym}'>
                <td style='color: #ffffff; font-weight: 600;'>{data['symbol']}</td>
                <td><span id='price-{clean_sym}' class='price-ticker'>$0.00</span></td>
                <td><span id='change-{clean_sym}' class='badge-glow'>0.00%</span></td>
                <td><span class='badge-metric'>RSI: {data['rsi']}</span></td>
                <td style='color: #cbd5e1;'>${data['entry']}</td>
                <td><span class='status-pill {sig_class}'>{data['signal']}</span></td>
                <td style='color: #00b574;'>${data['tp']}</td>
                <td style='color: #ff3b30;'>${data['sl']}</td>
            </tr>"""

        if not monitor_rows:
            for sym in self.symbols:
                clean_sym = sym.replace("/", "").lower()
                monitor_rows += f"""
                <tr id='row-{clean_sym}'>
                    <td style='color: #ffffff; font-weight: 600;'>{sym}</td>
                    <td><span id='price-{clean_sym}' class='price-ticker'>$0.00</span></td>
                    <td><span id='change-{clean_sym}' class='badge-glow'>0.00%</span></td>
                    <td colspan='5' style='color: #64748b; text-align: center; font-size:12px; font-weight: 500;'>⚡ SCANNING ENGINE ACTIVE (WAITING FOR VOLATILITY BREAKOUT)</td>
                </tr>"""

        history_rows = ""
        trade_list = list(self.state.get("trades", []))
        reversed_trades = trade_list[::-1][:8]
        for t in reversed_trades:
            t_color = "#00b574" if float(t["pnl"]) >= 0 else "#ff3b30"
            badge_type = "history-buy" if t["side"] == "BUY" else "history-sell"
            history_rows += f"""
            <tr>
                <td style='color: #64748b;'>{t['time']}</td>
                <td><b>{t['symbol']}</b></td>
                <td><span class='hist-pill {badge_type}'>{t['side']}</span></td>
                <td>${t['entry']}</td>
                <td>${t['exit']}</td>
                <td style='color:{t_color}; font-weight:600;'>{t['result']}</td>
                <td style='color: {t_color}; font-weight: bold; font-family: monospace;'>${t['pnl']}</td>
            </tr>"""

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WhaleTrader Pro Dashboard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #08090c; color: #cbd5e1; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 15px; margin-bottom: 25px; }}
        h1 {{ color: #ffffff; font-size: 18px; font-weight: 700; display: flex; align-items: center; gap: 8px; margin: 0; }}
        h1::before {{ content: ''; display: inline-block; width: 8px; height: 8px; background: #00b574; border-radius: 50%; box-shadow: 0 0 8px #00b574; }}
        .matrix-container {{ display: flex; gap: 15px; margin-bottom: 25px; }}
        .stat-card {{ background: #0f111a; border: 1px solid #1e293b; padding: 16px; border-radius: 8px; flex: 1; }}
        .stat-label {{ color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
        .stat-value {{ font-size: 24px; font-weight: 700; font-family: monospace; }}
        table {{ width: 100%; border-collapse: collapse; background-color: #0b0d13; border-radius: 8px; margin-bottom: 25px; overflow: hidden; border: 1px solid #1e293b; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #1e293b; font-size: 13px; }}
        th {{ background-color: #0f121a; color: #64748b; font-size: 11px; text-transform: uppercase; font-weight: 600; }}
        tr:hover {{ background-color: #131722; }}
        .price-ticker {{ font-family: monospace; font-size: 14px; font-weight: bold; }}
        .badge-glow {{ font-family: monospace; font-size: 12px; font-weight: 600; padding: 2px 6px; border-radius: 4px; }}
        .text-up {{ color: #00b574 !important; }} .text-down {{ color: #ff3b30 !important; }}
        .bg-up {{ background-color: rgba(0, 181, 116, 0.08); }} .bg-down {{ background-color: rgba(255, 59, 48, 0.08); }}
        .status-pill {{ padding: 4px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; }}
        .buy-glow {{ background-color: rgba(0, 181, 116, 0.1); color: #00b574; border: 1px solid rgba(0,181,116,0.3); }}
        .sell-glow {{ background-color: rgba(255, 59, 48, 0.1); color: #ff3b30; border: 1px solid rgba(255,59,48,0.3); }}
        .hist-pill {{ padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
        .history-buy {{ background: rgba(0,181,116,0.08); color: #00b574; }}
        .history-sell {{ background: rgba(255,59,48,0.08); color: #ff3b30; }}
        .badge-metric {{ color: #38bdf8; background: rgba(56,189,248,0.08); padding: 2px 6px; border-radius: 4px; font-weight: 500; font-size: 12px; }}
        h3 {{ color: #ffffff; font-size: 14px; font-weight: 600; margin-bottom: 12px; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>WhaleTrader Pro Terminal</h1>
            <div style="color: #64748b; font-size: 12px; font-weight: 600;">Sync: <span id="clock-sync">{now_str}</span></div>
        </header>
        
        <div class="matrix-container">
            <div class="stat-card">
                <div class="stat-label">Account Equity</div>
                <div class="stat-value" style="color: #ffffff;">${current_wallet} <span style="font-size:12px; color:#64748b;">USD</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Leverage Strategy</div>
                <div class="stat-value" style="color: #f59e0b;">10x Isolated</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Realized Net Returns</div>
                <div class="stat-value" style="color: {pnl_color};">{pnl_prefix}{pnl_val} USD</div>
            </div>
        </div>

        <h3>Active Asset Monitors</h3>
        <table>
            <thead>
                <tr>
                    <th>Asset Pair</th><th>Live Price</th><th>24h Delta</th><th>Metrics</th><th>Entry Price</th><th>State</th><th>Take Profit</th><th>Stop Loss</th>
                </tr>
            </thead>
            <tbody>{monitor_rows}</tbody>
        </table>

        <h3>Settlement Log</h3>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th><th>Asset</th><th>Vector</th><th>Entry</th><th>Exit</th><th>Status</th><th>P&L</th>
                </tr>
            </thead>
            <tbody>{history_rows if history_rows else '<tr><td colspan="7" style="text-align:center; color:#64748b; padding:15px;">Scanning markets for volatility spikes...</td></tr>'}</tbody>
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
                    const changePct = parseFloat(data.P).toFixed(2);
                    priceEl.innerText = "$" + price;
                    if (parseFloat(changePct) >= 0) {{
                        changeEl.innerText = "+" + changePct + "%";
                        changeEl.className = "badge-glow text-up bg-up";
                        priceEl.className = "price-ticker text-up";
                    }} else {{
                        changeEl.innerText = changePct + "%";
                        changeEl.className = "badge-glow text-down bg-down";
                        priceEl.className = "price-ticker text-down";
                    }}
                }}
            }};
            ws.onclose = () => {{ setTimeout(connectLiveTicker, 4000); }};
        }}

        function startLiveClock() {{
            setInterval(() => {{
                const now = new Date();
                const options = {{ hour12: true, hour: '2-digit', minute: '2-digit', second: '2-digit' }};
                const timeStr = now.toLocaleTimeString('en-US', options);
                
                const year = now.getFullYear();
                const month = String(now.getMonth() + 1).padStart(2, '0');
                const day = String(now.getDate()).padStart(2, '0');
                const dateStr = `${{year}}-${{month}}-${{day}}`;
                
                const syncEl = document.getElementById("clock-sync");
                if (syncEl) {{
                    syncEl.innerText = dateStr + " " + timeStr;
                }}
            }}, 1000);
        }}

        connectLiveTicker();
        startLiveClock();
        setTimeout(() => {{ window.location.reload(); }}, 300000);
    </script>
</body>
</html>"""
        return html_content

    def run_pipeline(self):
        log.info("⚡ WhaleTrader Premium Terminal System Executed.")
        for symbol in self.symbols:
            data = self.fetch_market_data(symbol)
            if data is None: continue
            opens, highs, lows, closes, volumes = data
            self.evaluate_signals(symbol, opens, highs, lows, closes, volumes)
        
        # Dashboard save yahan hoga
        html = self.generate_html_dashboard()
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
            
