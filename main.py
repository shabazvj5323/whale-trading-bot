import time
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
        self.history_file = "history.json"
        # Purani history secure karne ke liye default list yahan daal di hai
        self.state = self.load_and_clean_history()
        self.dashboard_data = []

    def load_and_clean_history(self):
        # Yahan aapki wohi purani history hai jo screenshot mein dikh rahi thi
        default = {
            "total_pnl": 41.19,
            "trades": [
                {"time": "06-22 16:08", "asset": "ETH/USDT", "vector": "SELL", "entry": "1755.07", "exit": "1742.5", "status": "Scalp TP 🎯", "pnl": "+$1.44"},
                {"time": "06-22 14:56", "asset": "PAXG/USDT", "vector": "SELL", "entry": "2343.41", "exit": "2320.22", "status": "Scalp TP 🎯", "pnl": "+$6.09"},
                {"time": "06-22 14:56", "asset": "BTC/USDT", "vector": "SELL", "entry": "66132.65", "exit": "64297.58", "status": "Scalp TP 🎯", "pnl": "+$1.01"},
                {"time": "2026-06-22 14:45", "asset": "BTC/USDT", "vector": "SELL", "entry": "64175.85", "exit": "64479.76", "status": "Scalp SL 🔴", "pnl": "-$11.84"},
                {"time": "2026-06-22 14:04", "asset": "BTC/USDT", "vector": "SELL", "entry": "66014.52", "exit": "65124.76", "status": "TP Hit 🎯", "pnl": "+$44.49"}
            ]
        }
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f: return json.load(f)
            except: return default
        return default

    def generate_html_dashboard(self):
        # History table ki rows
        hist_rows = "".join([f"<tr><td>{t['time']}</td><td>{t['asset']}</td><td style='color:red;'>{t['vector']}</td><td>{t['entry']}</td><td>{t['exit']}</td><td style='color:green;'>{t['status']}</td><td style='color:green;'>{t['pnl']}</td></tr>" for t in self.state["trades"]])
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<style>
    body {{ background: #08090c; color: #fff; font-family: sans-serif; padding: 20px; }}
    .stats-container {{ display: flex; gap: 20px; margin-bottom: 20px; }}
    .stat-card {{ background: #111; padding: 20px; border-radius: 10px; border: 1px solid #333; width: 30%; }}
    table {{ width: 100%; border-collapse: collapse; background: #0b0d13; margin-top: 20px; }}
    th, td {{ padding: 15px; border: 1px solid #222; text-align: left; }}
</style>
</head>
<body>
    <h1>WhaleTrader Pro Terminal</h1>
    <div class="stats-container">
        <div class="stat-card">ACCOUNT EQUITY<br><h2>$1041.19</h2></div>
        <div class="stat-card">LEVERAGE STRATEGY<br><h2>10x Isolated</h2></div>
        <div class="stat-card">REALIZED NET RETURNS<br><h2>+$41.19 USD</h2></div>
    </div>
    <h3>Settlement Log</h3>
    <table>
        <tr><th>TIMESTAMP</th><th>ASSET</th><th>VECTOR</th><th>ENTRY</th><th>EXIT</th><th>STATUS</th><th>P&L</th></tr>
        {hist_rows}
    </table>
</body>
</html>"""
        with open("index.html", "w") as f: f.write(html_content)

    def run(self):
        self.generate_html_dashboard()

if __name__ == "__main__":
    WhaleQuantEngine().run()
    
