import os
import requests
import logging
import hmac
import hashlib
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("WhaleTrader")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XAUUSDT", "XAGUSDT"]
VOLUME_MULTIPLIER = 2.5  
RSI_PERIOD = 14

def get_market_data(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=15m&limit=60"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            candles = []
            for k in r.json():
                candles.append({
                    "o": float(k[1]), "h": float(k[2]),
                    "l": float(k[3]), "c": float(k[4]), "v": float(k[5])
                })
            return candles, candles[-1]["c"]
    except Exception as e:
        log.error(f"Binance Telemetry Failed for {symbol}: {str(e)}")
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
    volume_breakout = current_volume > (avg_volume * VOLUME_MULTIPLIER)
    
    if volume_breakout and closes[-1] > closes[-2] and rsi < 70: return "BUY", rsi
    elif volume_breakout and closes[-1] < closes[-2] and rsi > 30: return "SELL", rsi
    return "WAIT", rsi

if __name__ == "__main__":
    log.info("WhaleTrader Pro V4 Engine Booted. Portfolio Tracking Engaged.")
    for symbol in SYMBOLS:
        log.info(f"--- Evaluating Matrix Array: {symbol} ---")
        candles, price = get_market_data(symbol)
        if candles:
            signal, rsi = extract_institutional_signals(candles)
            log.info(f"{symbol} Price: {price} | RSI: {round(rsi, 2)} | Signal: {signal}")
        else:
            log.warning(f"Telemetry dropped for {symbol}")
            
