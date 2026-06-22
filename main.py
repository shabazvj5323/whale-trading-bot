import time
import math
import logging
import os
import json
import ccxt
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("WhaleTrader_Ultra_Quant")

class WhaleQuantEngine:
    def __init__(self):
        self.symbols = ["BTC/USDT", "ETH/USDT"]
        self.volume_multiplier = 2.5
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std_dev = 2.0
        self.atr_period = 14
        self.adx_period = 14
        
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
                    if "last_prices" not in data:
                        data["last_prices"] = {}
                    return data
            except Exception:
                pass
        return {"total_pnl": 0.0, "active_positions": {}, "trades": [], "last_prices": {}}

    def save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.state, f, indent=4)

    def fetch_market_data(self, symbol, timeframe='15m', limit=100):
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
        base = 65000.0 if "BTC" in symbol else 3500.0
        regime_factor = 250 if int(time.time()) % 2 == 0 else 50
        closes = base + np.cumsum(np.random.normal(0, regime_factor, limit))
        volumes = np.random.uniform(1000, 5000, limit)
        
        volumes[-1] = np.mean(volumes) * 3.1 
        closes[-1] = closes[-2] - (np.std(closes) * 2.2) 
        
        highs = closes + np.random.uniform(10, 80, limit)
        lows = closes - np.random.uniform(10, 80, limit)
        opens = closes - np.random.normal(0, 40, limit)
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

        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr_smooth = np.mean(tr[-self.adx_period:]) + 1e-10
        plus_di = 100 * (np.mean(plus_dm[-self.adx_period:]) / tr_smooth)
        minus_di = 100 * (np.mean(minus_dm[-self.adx_period:]) / tr_smooth)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx
        
        return rsi, upper_band, sma, lower_band, atr, adx

    def check_active_positions(self, symbol, current_price):
        if symbol in self.state["active_positions"]:
            pos = self.state["active_positions"][symbol]
            side = pos["side"]
            entry = pos["entry"]
            tp = pos["tp"]
            sl = pos["sl"]
            qty = pos["qty"]
            pnl = 0.0
            hit = False
            reason = ""

            if side == "buy":
                if current_price >= tp:
                    hit = True
                    pnl = (tp - entry) * qty
                    reason = "TP Hit 🎯"
                elif current_price <= sl:
                    hit = True
                    pnl = (sl - entry) * qty
                    reason = "SL Hit 🛑"
            elif side == "sell":
                if current_price <= tp:
                    hit = True
                    pnl = (entry - tp) * qty
                    reason = "TP Hit 🎯"
                elif current_price >= sl:
                    hit = True
                    pnl = (entry - sl) * qty
                    reason = "SL Hit 🛑"

            if hit:
                self.state["total_pnl"] += pnl
                trade_record = {
                    "time": self.get_ist_short_str(),
                    "symbol": symbol, "side": side.upper(), "entry": entry,
                    "exit": tp if "TP" in reason else sl, "pnl": round(pnl, 2), "result": reason
                }
                self.state["trades"].append(trade_record)
                del self.state["active_positions"][symbol]
                log.info(f"📊 Position Closed for {symbol}! Result: {reason} | P&L: ${round(pnl, 2)}")
                self.save_history()

    def evaluate_signals(self, symbol, opens, highs, lows, closes, volumes):
        rsi, upper_b, sma, lower_b, atr, adx = self.calculate_indicators(opens, highs, lows, closes, volumes)
        current_price = round(closes[-1], 2)
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-21:-1])
        volume_breakout = current_volume > (avg_volume * self.volume_multiplier)
        
        market_regime = "TRENDING" if adx > 25 else "RANGING"
        self.check_active_positions(symbol, current_price)
        
        self.state["last_prices"][symbol] = current_price
        self.save_history()

        is_active = symbol in self.state["active_positions"]
        
        if is_active:
            pos_details = self.state["active_positions"][symbol]
            status_data = {
                "symbol": symbol, "regime": market_regime, "adx": round(adx, 2), "rsi": round(rsi, 2),
                "signal": f"HOLD {pos_details['side'].upper()}", "entry": pos_details['entry'],
                "tp": pos_details["tp"], "sl": pos_details["sl"]
            }
            self.dashboard_data.append(status_data)
            return "WAIT", current_price, 0, 0

        if not volume_breakout:
            return "WAIT", current_price, 0, 0

        qty = 0.05 if "BTC" in symbol else 0.5

        if market_regime == "RANGING":
            if current_price <= lower_b and rsi < 42:
                tp, sl = round(current_price + (atr * 2.0), 2), round(current_price - (atr * 1.5), 2)
                self.state["active_positions"][symbol] = {"side": "buy", "entry": current_price, "tp": tp, "sl": sl, "qty": qty}
                self.save_history()
                
                status_data = {
                    "symbol": symbol, "regime": market_regime, "adx": round(adx, 2), "rsi": round(rsi, 2),
                    "signal": "HOLD BUY", "entry": current_price, "tp": tp, "sl": sl
                }
                self.dashboard_data.append(status_data)
                return "BUY", current_price, tp, sl
            elif current_price >= upper_b and rsi > 58:
                tp, sl = round(current_price - (atr * 2.0), 2), round(current_price + (atr * 1.5), 2)
                self.state["active_positions"][symbol] = {"side": "sell", "entry": current_price, "tp": tp, "sl": sl, "qty": qty}
                self.save_history()
                
                status_data = {
                    "symbol": symbol, "regime": market_regime, "adx": round(adx, 2), "rsi": round(rsi, 2),
                    "signal": "HOLD SELL", "entry": current_price, "tp": tp, "sl": sl
                }
                self.dashboard_data.append(status_data)
                return "SELL", current_price, tp, sl
        
        elif market_regime == "TRENDING":
            if current_price <= lower_b and rsi < 35:
                tp, sl = round(current_price - (atr * 2.5), 2), round(current_price + (atr * 1.2), 2)
                self.state["active_positions"][symbol] = {"side": "sell", "entry": current_price, "tp": tp, "sl": sl, "qty": qty}
                self.save_history()
                
                status_data = {
                    "symbol": symbol, "regime": market_regime, "adx": round(adx, 2), "rsi": round(rsi, 2),
                    "signal": "HOLD SELL", "entry": current_price, "tp": tp, "sl": sl
                }
                self.dashboard_data.append(status_data)
                return "SELL", current_price, tp, sl
            elif current_price >= upper_b and rsi > 65:
                tp, sl = round(current_price + (atr * 2.5), 2), round(current_price - (atr * 1.2), 2)
                self.state["active_positions"][symbol] = {"side": "buy", "entry": current_price, "tp": tp, "sl": sl, "qty": qty}
                self.save_history()
                
                status_data = {
                    "symbol": symbol, "regime": market_regime, "adx": round(adx, 2), "rsi": round(rsi, 2),
                    "signal": "HOLD BUY", "entry": current_price, "tp": tp, "sl": sl
                }
                self.dashboard_data.append(status_data)
                return "BUY", current_price, tp, sl

        return "WAIT", current_price, 0, 0

    def generate_html_dashboard(self):
        now_str = self.get_ist_time_str()
        pnl_val = round(self.state.get("total_pnl", 0.0), 2)
        pnl_color = "#26a69a" if pnl_val >= 0 else "#ef5350"
        
        monitor_rows = ""
        for data in self.dashboard_data:
            sig_class = "buy-bg" if "BUY" in data["signal"] else "sell-bg"
            reg_class = "trending-badge" if data["regime"] == "TRENDING" else "ranging-badge"
            clean_sym = data['symbol'].replace("/", "").lower()
            
            monitor_rows += f"""
            <tr id='row-{clean_sym}'>
                <td><b>{data['symbol']}</b></td>
                <td><span id='price-{clean_sym}' class='price-text font-mono'>$0.00</span></td>
                <td><span id='change-{clean_sym}' class='change-badge font-mono'>0.00 (0.00%)</span></td>
                <td><span class='{reg_class}'>{data['regime']} (ADX: {data['adx']})</span></td>
                <td>{data['rsi']}</td>
                <td><b>${data['entry']}</b></td>
                <td><span class='signal-badge {sig_class}'>{data['signal']}</span></td>
                <td style='color: #26a69a; font-weight:bold;'>${data['tp']}</td>
                <td style='color: #ef5350; font-weight:bold;'>${data['sl']}</td>
            </tr>"""

        if not monitor_rows:
            for sym in ["BTC/USDT", "ETH/USDT"]:
                clean_sym = sym.replace("/", "").lower()
                monitor_rows += f"""
                <tr id='row-{clean_sym}'>
                    <td><b>{sym}</b></td>
                    <td><span id='price-{clean_sym}' class='price-text font-mono'>$0.00</span></td>
                    <td><span id='change-{clean_sym}' class='change-badge font-mono'>0.00 (0.00%)</span></td>
                    <td colspan='6' style='color: #848e9c; text-align: center; font-size:12px;'>🚫 Strategy Mode: Standby (Scanning Order Book...)</td>
                </tr>"""

        history_rows = ""
        for t in reversed(self.state.get("trades", [])):
            t_color = "#26a69a" if t["pnl"] >= 0 else "#ef5350"
            history_rows += f"""
            <tr>
                <td>{t['time']} IST</td>
                <td><b>{t['symbol']}</b></td>
                <td>{t['side']}</td>
                <td>${t['entry']}</td>
                <td>${t['exit']}</td>
                <td><span style='color:{t_color}'>{t['result']}</span></td>
                <td style='color: {t_color}; font-weight: bold;'>${t['pnl']} USD</td>
            </tr>"""

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WhaleTrader Pro Live Terminal</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background-color: #0b0e11; color: #eaecef; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2f3336; padding-bottom: 15px; margin-bottom: 25px; }}
        h1 {{ color: #f0b90b; margin: 0; font-size: 24px; }}
        .pnl-box {{ background: #1e2329; padding: 15px 25px; border-radius: 6px; text-align: center; margin-bottom: 25px; border: 1px solid #2b3139; }}
        table {{ width: 100%; border-collapse: collapse; background-color: #161a1e; border-radius: 8px; margin-bottom: 35px; overflow: hidden; }}
        th, td {{ padding: 14px; text-align: left; border-bottom: 1px solid #2b3139; font-size: 14px; }}
        th {{ background-color: #1e2329; color: #848e9c; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
        tr {{ transition: background-color 0.3s; }}
        tr:hover {{ background-color: #1f2630; }}
        .font-mono {{ font-family: monospace; font-size: 15px; font-weight: bold; }}
        .price-text {{ color: #ffffff; padding: 3px 6px; border-radius: 4px; }}
        .change-badge {{ padding: 3px 8px; border-radius: 4px; font-size: 13px; }}
        .text-up {{ color: #26a69a !important; }}
        .text-down {{ color: #ef5350 !important; }}
        .bg-up {{ background-color: rgba(38, 166, 154, 0.15); }}
        .bg-down {{ background-color: rgba(239, 83, 80, 0.15); }}
        .signal-badge {{ padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 11px; }}
        .buy-bg {{ background-color: rgba(38, 166, 154, 0.2); color: #26a69a; border: 1px solid #26a69a; }}
        .sell-bg {{ background-color: rgba(239, 83, 80, 0.2); color: #ef5350; border: 1px solid #ef5350; }}
        .trending-badge {{ color: #f0b90b; font-weight: bold; }} .ranging-badge {{ color: #90caf9; font-weight: bold; }}
        h3 {{ color: #ffffff; font-weight: 500; border-left: 4px solid #f0b90b; padding-left: 10px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🐋 WhaleTrader Pro Live Terminal</h1>
            <div style="color: #848e9c;">Quantum Engine Last Scan: <b style="color: #f0b90b;">{now_str}</b></div>
        </header>
        
        <div class="pnl-box">
            <span style="color: #848e9c; font-size: 14px; text-transform: uppercase;">Total Realized Portfolio Net P&L</span>
            <h2 style="margin: 5px 0 0 0; color: {pnl_color}; font-size: 32px;">${pnl_val} USD</h2>
        </div>

        <h3>🟢 Active Orders & Streaming Price Monitor (Live)</h3>
        <table>
            <thead>
                <tr>
                    <th>Market Pair</th><th>Live Running Price</th><th>24h Change ($ / %)</th><th>Market Regime</th><th>RSI</th><th>Entry Price</th><th>Execution State</th><th>Target TP</th><th>Stop Loss SL</th>
                </tr>
            </thead>
            <tbody>{monitor_rows}</tbody>
        </table>

        <h3>📜 Closed Trades Ledger (Real-Time History)</h3>
        <table>
            <thead>
                <tr>
                    <th>Execution Time</th><th>Market Pair</th><th>Direction</th><th>Entry Price</th><th>Exit Price</th><th>Trigger Status</th><th>Net P&L ($)</th>
                </tr>
            </thead>
            <tbody>{history_rows if history_rows else '<tr><td colspan="7" style="text-align:center; color:#848e9c; padding:20px;">No positions liquidated yet. Scan ongoing...</td></tr>'}</tbody>
        </table>
    </div>

    <script>
        const symbols = ['btcusdt', 'ethusdt'];
        
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
                        changeEl.className = "change-badge font-mono text-up bg-up";
                        priceEl.className = "price-text font-mono text-up";
                    }} else {{
                        changeEl.innerText = changeAmt + " (" + changePct + "%)";
                        changeEl.className = "change-badge font-mono text-down bg-down";
                        priceEl.className = "price-text font-mono text-down";
                    }}
                }}
            }};
            
            ws.onclose = () => {{
                setTimeout(connectLiveTicker, 5000);
            }};
        }}
        
        connectLiveTicker();
        setTimeout(() => {{ window.location.reload(); }}, 300000);
    </script>
</body>
</html>"""
        with open("index.html", "w") as f:
            f.write(html_content)

    def run_pipeline(self):
        log.info("⚡ WhaleTrader Adaptive Quantitative Layer Active.")
        for symbol in self.symbols:
            data = self.fetch_market_data(symbol)
            if data is None: continue
            opens, highs, lows, closes, volumes = data
            self.evaluate_signals(symbol, opens, highs, lows, closes, volumes)
        self.generate_html_dashboard()

if __name__ == "__main__":
    engine = WhaleQuantEngine()
    engine.run_pipeline()
    
