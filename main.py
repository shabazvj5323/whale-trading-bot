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
        """Calculates exact Indian Standard Time (IST) from UTC"""
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        return ist_now.strftime("%Y-%m-%d %I:%M:%S %p (IST)")

    def get_ist_short_str(self):
        """Short time format for ledger logs"""
        utc_now = datetime.utcnow()
        ist_now = utc_now + timedelta(hours=5, minutes=30)
        return ist_now.strftime("%Y-%m-%d %H:%M")

    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"total_pnl": 0.0, "active_positions": {}, "trades": []}

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
        current_price = closes[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-21:-1])
        volume_breakout = current_volume > (avg_volume * self.volume_multiplier)
        
        market_regime = "TRENDING" if adx > 25 else "RANGING"
        self.check_active_positions(symbol, current_price)
        
        is_active = symbol in self.state["active_positions"]
        active_display = "IN POSITION" if is_active else "STANDBY"
        
        status_data = {
            "symbol": symbol, "price": round(current_price, 2), "adx": round(adx, 2),
            "regime": market_regime, "rsi": round(rsi, 2), "upper": round(upper_b, 2),
            "lower": round(lower_b, 2), "signal": active_display, "tp": 0, "sl": 0
        }
        
        if is_active:
            status_data.update({
                "signal": f"HOLD {self.state['active_positions'][symbol]['side'].upper()}",
                "tp": self.state["active_positions"][symbol]["tp"],
                "sl": self.state["active_positions"][symbol]["sl"]
            })
            self.dashboard_data.append(status_data)
            return "WAIT", current_price, 0, 0

        if not volume_breakout:
            self.dashboard_data.append(status_data)
            return "WAIT", current_price, 0, 0

        qty = 0.05 if "BTC" in symbol else 0.5

        if market_regime == "RANGING":
            if current_price <= lower_b and rsi < 42:
                tp, sl = round(current_price + (atr * 2.0), 2), round(current_price - (atr * 1.5), 2)
                status_data.update({"signal": "BUY Triggered", "tp": tp, "sl": sl})
                self.state["active_positions"][symbol] = {"side": "buy", "entry": round(current_price, 2), "tp": tp, "sl": sl, "qty": qty}
                self.save_history()
                self.dashboard_data.append(status_data)
                return "BUY", current_price, tp, sl
            elif current_price >= upper_b and rsi > 58:
                tp, sl = round(current_price - (atr * 2.0), 2), round(current_price + (atr * 1.5), 2)
                status_data.update({"signal": "SELL Triggered", "tp": tp, "sl": sl})
                self.state["active_positions"][symbol] = {"side": "sell", "entry": round(current_price, 2), "tp": tp, "sl": sl, "qty": qty}
                self.save_history()
                self.dashboard_data.append(status_data)
                return "SELL", current_price, tp, sl
        
        elif market_regime == "TRENDING":
            if current_price <= lower_b and rsi < 35:
                tp, sl = round(current_price - (atr * 2.5), 2), round(current_price + (atr * 1.2), 2)
                status_data.update({"signal": "SHORT Breakout", "tp": tp, "sl": sl})
                self.state["active_positions"][symbol] = {"side": "sell", "entry": round(current_price, 2), "tp": tp, "sl": sl, "qty": qty}
                self.save_history()
                self.dashboard_data.append(status_data)
                return "SELL", current_price, tp, sl
            elif current_price >= upper_b and rsi > 65:
                tp, sl = round(current_price + (atr * 2.5), 2), round(current_price - (atr * 1.2), 2)
                status_data.update({"signal": "LONG Breakout", "tp": tp, "sl": sl})
                self.state["active_positions"][symbol] = {"side": "buy", "entry": round(current_price, 2), "tp": tp, "sl": sl, "qty": qty}
                self.save_history()
                self.dashboard_data.append(status_data)
                return "BUY", current_price, tp, sl

        self.dashboard_data.append(status_data)
        return "WAIT", current_price, 0, 0

    def generate_html_dashboard(self):
        now_str = self.get_ist_time_str()
        pnl_val = round(self.state.get("total_pnl", 0.0), 2)
        pnl_color = "#26a69a" if pnl_val >= 0 else "#ef5350"
        
        rows_html = ""
        for data in self.dashboard_data:
            sig_class = "wait" if "STANDBY" in data["signal"] else ("buy-bg" if "BUY" in data["signal"] or "LONG" in data["signal"] else "sell-bg")
            reg_class = "trending-badge" if data["regime"] == "TRENDING" else "ranging-badge"
            rows_html += f"""
            <tr>
                <td><b>{data['symbol']}</b></td>
                <td><span class='price-text'>${data['price']}</span></td>
                <td><span class='{reg_class}'>{data['regime']} (ADX: {data['adx']})</span></td>
                <td>{data['rsi']}</td>
                <td><span class='signal-badge {sig_class}'>{data['signal']}</span></td>
                <td style='color: #26a69a;'>{data['tp'] if data['tp'] > 0 else '-'}</td>
                <td style='color: #ef5350;'>{data['sl'] if data['sl'] > 0 else '-'}</td>
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
                <td>{t['result']}</td>
                <td style='color: {t_color}; font-weight: bold;'>${t['pnl']}</td>
            </tr>"""

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WhaleTrader Quant Live Terminal</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; background-color: #0b0e11; color: #eaecef; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2f3336; padding-bottom: 15px; margin-bottom: 25px; }}
        h1 {{ color: #f0b90b; margin: 0; font-size: 24px; }}
        .pnl-box {{ background: #1e2329; padding: 15px 25px; border-radius: 6px; text-align: center; margin-bottom: 25px; border: 1px solid #2b3139; }}
        table {{ width: 100%; border-collapse: collapse; background-color: #161a1e; border-radius: 8px; margin-bottom: 30px; overflow: hidden; }}
        th, td {{ padding: 14px; text-align: left; border-bottom: 1px solid #2b3139; }}
        th {{ background-color: #1e2329; color: #848e9c; font-size: 12px; text-transform: uppercase; }}
        .price-text {{ font-family: monospace; font-size: 15px; font-weight: bold; }}
        .signal-badge {{ padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 11px; }}
        .wait {{ background-color: #2b3139; color: #848e9c; }}
        .buy-bg {{ background-color: rgba(38, 166, 154, 0.2); color: #26a69a; border: 1px solid #26a69a; }}
        .sell-bg {{ background-color: rgba(239, 83, 80, 0.2); color: #ef5350; border: 1px solid #ef5350; }}
        .trending-badge {{ color: #f0b90b; }} .ranging-badge {{ color: #90caf9; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🐋 WhaleTrader Quant Live Terminal</h1>
            <div style="color: #848e9c;">Last Scan: <b style="color: #f0b90b;">{now_str}</b></div>
        </header>
        
        <div class="pnl-box">
            <span style="color: #848e9c; font-size: 14px; text-transform: uppercase;">Total Realized Account P&L</span>
            <h2 style="margin: 5px 0 0 0; color: {pnl_color}; font-size: 32px;">${pnl_val} USD</h2>
        </div>

        <h3>📡 Live Market Scan & Active Monitor</h3>
        <table>
            <thead>
                <tr>
                    <th>Pair</th><th>Price</th><th>Regime Index</th><th>RSI</th><th>Status</th><th>Target TP</th><th>Stop SL</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>

        <h3>📜 Closed Trades Ledger (P&L History)</h3>
        <table>
            <thead>
                <tr>
                    <th>Close Time</th><th>Pair</th><th>Side</th><th>Entry Price</th><th>Exit Price</th><th>Trigger Reason</th><th>Profit/Loss</th>
                </tr>
            </thead>
            <tbody>{history_rows if history_rows else '<tr><td colspan="7" style="text-align:center; color:#848e9c;">No closed transactions yet. System monitoring live targets...</td></tr>'}</tbody>
        </table>
    </div>
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
    
