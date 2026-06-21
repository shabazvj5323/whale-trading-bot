import os
import requests
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("WhaleTrader")

# --- CORE PARAMETERS (AAPKI ORIGINAL STRATEGY) ---
SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT"]
VOLUME_MULTIPLIER = 2.5  
RSI_PERIOD = 14

def get_market_data(symbol):
    try:
        # KuCoin Public Live Feed - Yeh GitHub enterprise ranges par 100% unblocked hai
        url = f"https://api.kucoin.com/api/v1/market/candles?symbol={symbol}&type=15min"
        
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            res_data = r.json().get("data", [])
            if not res_data: return None, None
            
            candles = []
            # KuCoin returns: [time, open, close, high, low, volume, turnover]
            # Hamein kam se kam 30 candles chahiye calculation ke liye
            for k in reversed(res_data[:60]): 
                candles.append({
                    "o": float(k[1]), "c": float(k[2]),
                    "h": float(k[3]), "l": float(k[4]), "v": float(k[5])
                })
            return candles, candles[-1]["c"]
    except Exception as e:
        log.debug(f"Stream interface filter: {str(e)}")
    return None, None

def calculate_rsi(prices, period=14):
    if len(prices) < period: return 50
    gains = [max(prices[i] - prices[i-1], 0) for i in range(1, len(prices))]
    losses = [max(prices[i-1] - prices[i], 0) for i in range(1, len(prices))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0: return 100
    for i in range(period, len(prices) - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def extract_institutional_signals(candles):
    closes = [c["c"] for c in candles]
    volumes = [c["v"] for c in candles]
    current_volume = volumes[-1]
    avg_volume = sum(volumes[-21:-1]) / 20
    rsi = calculate_rsi(closes, RSI_PERIOD)
    
    # Original Breakout Conditions
    volume_breakout = current_volume > (avg_volume * VOLUME_MULTIPLIER)
    
    if volume_breakout and closes[-1] > closes[-2] and rsi < 70: return "BUY", rsi
    elif volume_breakout and closes[-1] < closes[-2] and rsi > 30: return "SELL", rsi
    return "WAIT", rsi

if __name__ == "__main__":
    log.info("WhaleTrader Pro V4 Engine Booted. GitHub Network Integration Active.")
    for symbol in SYMBOLS:
        display_name = symbol.replace("-", "")
        log.info(f"--- Evaluating Matrix Array: {display_name} ---")
        candles, price = get_market_data(symbol)
        if candles:
            signal, rsi = extract_institutional_signals(candles)
            log.info(f"{display_name} Live Price: {price} | RSI: {round(rsi, 2)} | Signal: {signal}")
        else:
            log.error(f"GitHub cloud network strictly blocked stream for {display_name}")
            
