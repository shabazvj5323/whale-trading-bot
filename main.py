import os
import json
from datetime import datetime, timedelta

class WhaleQuantEngine:
    def __init__(self):
        self.history_file = "history.json"
        # Yahan main wahi keys use kar raha hoon jo aapke log mein dikh rahi hain
        self.state = self.load_history()

    def load_history(self):
        default = {"trades": [{"time":"06-22 16:08","symbol":"ETH/USDT","pnl":"1.44"}, {"time":"06-22 14:56","symbol":"PAXG/USDT","pnl":"6.09"}, {"time":"06-22 14:56","symbol":"BTC/USDT","pnl":"1.01"}, {"time":"06-22 14:45","symbol":"BTC/USDT","pnl":"-11.84"}, {"time":"06-22 14:04","symbol":"BTC/USDT","pnl":"44.49"}]}
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f: return json.load(f)
            except: return default
        return default

    def generate_html_dashboard(self):
        # Yahan sirf wahi keys use ki hain jo aapke JSON mein hain
        hist_rows = "".join([f"<tr><td>{t.get('time', 'N/A')}</td><td>{t.get('symbol', 'N/A')}</td><td>{t.get('pnl', '0')}</td></tr>" for t in self.state["trades"]])
        
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
        <div class="stat-card">LEVERAGE<br><h2>10x Isolated</h2></div>
        <div class="stat-card">REALIZED PnL<br><h2>+$41.19 USD</h2></div>
    </div>
    <h3>Settlement Log</h3>
    <table><tr><th>Time</th><th>Asset</th><th>P&L</th></tr>{hist_rows}</table>
    <script>
        let t = 900; 
        setInterval(()=>{{
            t--;
            if(t <= 0) location.reload();
        }}, 1000);
    </script>
</body>
</html>"""
        with open("index.html", "w") as f: f.write(html_content)

    def run(self):
        self.generate_html_dashboard()

if __name__ == "__main__":
    WhaleQuantEngine().run()
    
