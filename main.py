import os
import logging
import time
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("WhaleTrader")

# --- AAPKI ORIGINAL STRATEGY PARAMETERS (NO CHANGE) ---
# Mapping tickers directly to Yahoo Finance Stable feeds
TICKERS = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "XAUUSDT": "GC=F",   # Gold Futures
    "XAGUSDT": "SI=F"    # Silver Futures
}
VOLUME_MULTIPLIER = 2.5  
RSI_PERIOD = 14

def get_market_data(ticker_symbol):
    try:
        # Fetching 15m intervals data via unblocked Yahoo Network
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="2d", interval="15m")
        
        if df.empty or len(df) < 30: 
            return None, None
            
        candles = []
        for index, row in df.iterrows():
            candles.append({
                "o": float(row['Open']), "h": float(row['High']),
                "l": float(row['Low']), "c": float(row['Close']), "v": float(row['Volume'])
            })
        return candles, candles[-1]["c"]
    except Exception as e:
        log.error(f"Yahoo Feed Fetch Error: {str(e)}")
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
    
    # Core Strategy Logic (Exactly Same)
    volume_breakout = current_volume > (avg_volume * VOLUME_MULTIPLIER)
    
    if volume_breakout and closes[-1] > closes[-2] and rsi < 70: return "BUY", rsi
    elif volume_breakout and closes[-1] < closes[-2] and rsi > 30: return "SELL", rsi
    return "WAIT", rsi

if __name__ == "__main__":
    log.info("WhaleTrader Pro V4 Engine Booted. Data Bridge active.")
    for display_name, yahoo_ticker in TICKERS.items():
        log.info(f"--- Evaluating Matrix Array: {display_name} ---")
        candles, price = get_market_data(yahoo_ticker)
        if candles:
            signal, rsi = extract_institutional_signals(candles)
            log.info(f"{display_name} Live Price: {round(price, 2)} | RSI: {round(rsi, 2)} | Signal: {signal}")
        else:
            log.error(f"Data stream unavailable for {display_name}")
            
