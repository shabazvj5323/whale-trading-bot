import os
import time
import requests
import json
import logging
import math
import httpx
import hmac
import hashlib
from datetime import datetime, timezone
from groq import Groq

# Whale Grade Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# GitHub Secrets Architecture
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
EMAIL_FROM       = os.environ.get("EMAIL_FROM", "")
EMAIL_TO         = os.environ.get("EMAIL_TO", "")
BINANCE_API_KEY  = os.environ.get("BINANCE_API_KEY", "")
BINANCE_SECRET   = os.environ.get("BINANCE_SECRET_KEY", "")

# 100% Free Public Aggregator (Bypasses Binance Cloudflare Block on GitHub/Railway)
DATA_URL  = "https://min-api.cryptocompare.com/data/v2/histominute"
DEMO_URL  = "https://testnet.binancefuture.com"  # Tight Order Execution Layer

# UPDATED: Added Gold (XAUUSDT) and Silver (XAGUSDT)
SYMBOLS   = ["BTCUSDT", "ETHUSDT", "XAUUSDT", "XAGUSDT"]
MIN_SCORE = 7
MODEL     = "llama-3.3-70b-versatile"
LEVERAGE  = 10
CAPITAL   = 100

groq_client = Groq(api_key=GROQ_API_KEY, http_client=httpx.Client()) if GROQ_API_KEY else None

def send_email(subject, body):
    if not SENDGRID_API_KEY: return False
    try:
        r = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"},
            json={
                "personalizations": [{"to": [{"email": EMAIL_TO}]}],
                "from": {"email": EMAIL_FROM, "name": "WhaleTrader Pro Engine"},
                "subject": subject,
                "content": [{"type": "text/html", "value": body}],
            },
            timeout=15,
        )
        return r.status_code == 202
    except Exception as e:
        log.error(f"Email routing failed: {str(e)}")
        return False

def binance_request(method, endpoint, params=None, signed=False):
    if params is None: params = {}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        query = "&".join(str(k) + "=" + str(v) for k, v in params.items())
        sig = hmac.new(BINANCE_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
    try:
        url = DEMO_URL + endpoint
        if method == "GET": r = requests.get(url, params=params, headers=headers, timeout=10)
        elif method == "POST": r = requests.post(url, params=params, headers=headers, timeout=10)
        return r.json()
    except Exception as e:
        log.error(f"Execution API Error: {str(e)}")
        return None

def get_position(symbol):
    try:
        data = binance_request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
        if data and isinstance(data, list):
            for p in data:
                if isinstance(p, dict) and p.get("symbol") == symbol:
                    if float(p.get("positionAmt", 0)) != 0: return p
        return None
    except Exception:
        return None

# Bypassing Cloud Blockers: Dynamic Mapping for CryptoCompare Engine
def get_market_data(symbol):
    if "BTC" in symbol: fsym = "BTC"
    elif "ETH" in symbol: fsym = "ETH"
    elif "XAU" in symbol: fsym = "XAU"
    elif "XAG" in symbol: fsym = "XAG"
    else: fsym = symbol.replace("USDT", "")

    try:
        r = requests.get(f"{DATA_URL}?fsym={fsym}&tsym=USDT&limit=60", timeout=10)
        if r.status_code == 200:
            raw_data = r.json().get("Data", {}).get("Data", [])
            if raw_data:
                candles = [{"o": float(c["open"]), "h": float(c["high"]), "l": float(c["low"]), "c": float(c["close"]), "v": float(c["volumeto"])} for c in raw_data]
                return candles, candles[-1]["c"]
    except Exception as e:
        log.error(f"Bypass Telemetry Failed for {symbol}: {str(e)}")
    return None, None

# UPDATED: Precision rules handling for Commodities and Crypto Assets
def format_precision(symbol, price, qty):
    if "BTC" in symbol: 
        return round(price, 1), round(qty, 3)
    elif "ETH" in symbol: 
        return round(price, 2), round(qty, 3)
    elif "XAU" in symbol:  # Gold Precision (e.g., $2350.55, Qty 0.01)
        return round(price, 2), round(qty, 2)
    elif "XAG" in symbol:  # Silver Precision (e.g., $29.456, Qty 0.1)
        return round(price, 3), round(qty, 1)
    return round(price, 2), round(qty, 2)

def calc_atr(candles, period=14):
    if len(candles) < period + 1: return 10.0
    tr_values = []
    for i in range(1, len(candles)):
        h = candles[i]["h"]
        l = candles[i]["l"]
        pc = candles[i-1]["c"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_values.append(tr)
    return sum(tr_values[-period:]) / period

def bollinger_bands(closes):
    if len(closes) < 20: return None
    mid = sum(closes[-20:]) / 20
    variance = sum((x - mid) ** 2 for x in closes[-20:]) / 20
    std = math.sqrt(variance)
    upper, lower = mid + 2.0 * std, mid - 2.0 * std
    width = (upper - lower) / mid * 100
    pct_b = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
    return {"upper": round(upper, 3), "middle": round(mid, 3), "lower": round(lower, 3), "width": round(width, 2), "pct_b": round(pct_b, 2), "squeeze": width < 2.2}

def calc_rsi(closes):
    if len(closes) < 15: return 50
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag, al = sum(gains[-14:]) / 14, sum(losses[-14:]) / 14
    return round(100 - (100 / (1 + ag / al)), 1) if al != 0 else 100

def calc_ema(closes, p):
    if len(closes) < p: return closes[-1]
    k = 2 / (p + 1)
    v = sum(closes[:p]) / p
    for c in closes[p:]: v = c * k + v * (1 - k)
    return round(v, 3)

def detect_patterns(candles, bb, vol_ratio):
    if len(candles) < 3 or not bb: return []
    c, p1, p2 = candles[-1], candles[-2], candles[-3]
    pts = []
    body_c = abs(c["c"] - c["o"])
    rng_c = c["h"] - c["l"]

    if p1["l"] <= bb["lower"] * 1.002 and c["c"] > bb["lower"] and c["c"] > c["o"]:
        pts.append("BB Lower Bounce BUY")
    if p1["h"] >= bb["upper"] * 0.998 and c["c"] < bb["upper"] and c["c"] < c["o"]:
        pts.append("BB Upper Reject SELL")
    if bb["squeeze"] and c["c"] > bb["upper"] and vol_ratio >= 1.5:
        pts.append("CONFIRMED Institution Breakout UP BUY")
    if bb["squeeze"] and c["c"] < bb["lower"] and vol_ratio >= 1.5:
        pts.append("CONFIRMED Institution Breakout DOWN SELL")
    if p1["c"] < p1["o"] and c["c"] > c["o"] and c["c"] > p1["o"] and vol_ratio >= 1.3:
        pts.append("Bullish Engulfing BUY")
    if p1["c"] > p1["o"] and c["c"] < c["o"] and c["c"] < p1["o"] and vol_ratio >= 1.3:
        pts.append("Bearish Engulfing SELL")
        
    lw = min(c["o"], c["c"]) - c["l"]
    if lw > body_c * 2 and rng_c > 0 and c["l"] <= bb["lower"] * 1.01: pts.append("Hammer BUY")
    uw = c["h"] - max(c["o"], c["c"])
    if uw > body_c * 2 and rng_c > 0 and c["h"] >= bb["upper"] * 0.99: pts.append("Shooting Star SELL")
    return pts

def get_ai_signal(symbol, price, bb, rsi_v, ema9, ema21, vol_ratio, patterns, atr):
    if not groq_client: return {"action": "WAIT", "score": 0, "reason": "System offline."}
    try:
        pat_str = ", ".join(patterns) if patterns else "None"
        prompt = (
            f"You are a Top 10 World-Class Crypto & Commodities Whale Trader running an institutional desk.\n"
            f"Asset Focus: {symbol} | Current Price: ${price} | Market ATR: {atr}\n"
            f"Bollinger Bands: Upper: ${bb['upper']} | Middle: ${bb['middle']} | Lower: ${bb['lower']}\n"
            f"Squeeze Status: {bb['squeeze']} | RSI(14): {rsi_v} | Vol Ratio: {vol_ratio}x of baseline\n"
            f"EMA9/21: {ema9} / {ema21} | Detected Structural Patterns: {pat_str}\n\n"
            f"Strict Institutional Directives:\n"
            f"1. Only act if technical structure perfectly matches high-volume directional intent.\n"
            f"2. Set Stop Loss (sl) dynamically based on market structure and ATR protection.\n"
            f"3. Ensure Take Profit (tp1) offers an authentic risk-to-reward matrix (Minimum 1:1.5, ideally 1:2).\n"
            f"Return ONLY valid minified JSON framework:\n"
            f'{{"action":"BUY","score":8,"entry":{price},"sl":{price - (1.5 * atr)},"tp1":{price + (3 * atr)},"reason":"1-line Hinglish risk mapping"}}\n'
            f"Rules: action must be BUY, SELL, or WAIT. score range 1-10."
        )
        r = groq_client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=150)
        raw = r.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
    except Exception as e:
        log.error(f"AI Matrix Inference Failure: {str(e)}")
        return {"action": "WAIT", "score": 0}

def execute_trade(symbol, action, price, sl, tp1):
    try:
        binance_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": LEVERAGE}, signed=True)
        qty = (CAPITAL * LEVERAGE) / price
        clean_price, clean_qty = format_precision(symbol, price, qty)
        clean_sl, _ = format_precision(symbol, sl, qty)
        clean_tp, _ = format_precision(symbol, tp1, qty)

        # Force structural safety minimum size logic
        if clean_qty <= 0:
            clean_qty = 0.01 if "XAU" in symbol else 0.1

        # Execution Routing to Binance Core Engines
        order = binance_request("POST", "/fapi/v1/order", {"symbol": symbol, "side": action, "type": "MARKET", "quantity": clean_qty, "positionSide": "BOTH"}, signed=True)
        if order and "orderId" in str(order):
            time.sleep(2)
            close_side = "SELL" if action == "BUY" else "BUY"
            binance_request("POST", "/fapi/v1/order", {"symbol": symbol, "side": close_side, "type": "STOP_MARKET", "stopPrice": clean_sl, "quantity": clean_qty, "positionSide": "BOTH", "reduceOnly": "true", "timeInForce": "GTE_GTC"}, signed=True)
            binance_request("POST", "/fapi/v1/order", {"symbol": symbol, "side": close_side, "type": "TAKE_PROFIT_MARKET", "stopPrice": clean_tp, "quantity": clean_qty, "positionSide": "BOTH", "reduceOnly": "true", "timeInForce": "GTE_GTC"}, signed=True)
            log.info(f"💎 Institutional Order Filled for {symbol}! Qty: {clean_qty}, SL: {clean_sl}, TP: {clean_tp}")
            return order
    except Exception as e:
        log.error(f"Execution Protocol Breached: {str(e)}")
    return None

def analyze(symbol):
    log.info(f"--- Evaluating Matrix Array: {symbol} ---")
    if get_position(symbol):
        log.info(f"Whale Position already deployed on {symbol}. Frame skipped.")
        return

    candles, price = get_market_data(symbol)
    if not candles or len(candles) < 30:
        log.warning(f"Telemetry streams dropped for {symbol}. Moving to fallback loop.")
        return

    closes = [c["c"] for c in candles]
    vols = [c["v"] for c in candles]
    
    # Baseline Metrics
    bb = bollinger_bands(closes)
    rsi_v = calc_rsi(closes)
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    atr = calc_atr(candles)
    
    # Advanced Volatility Layer Check
    avg_vol = sum(vols[-20:-1]) / 19
    vol_r = round(vols[-1] / avg_vol, 2) if avg_vol > 0 else 1.0
    
    patterns = detect_patterns(candles, bb, vol_r)
    sig = get_ai_signal(symbol, price, bb, rsi_v, ema9, ema21, vol_r, patterns, atr)
    
    action, score = sig.get("action", "WAIT"), sig.get("score", 0)
    log.info(f"Asset: {symbol} | Price: ${price} | Volume Force: {vol_r}x | AI Rating: {action} ({score}/10)")

    if action in ["BUY", "SELL"] and score >= MIN_SCORE:
        sl = float(sig.get("sl", price - (1.5 * atr) if action == "BUY" else price + (1.5 * atr)))
        tp1 = float(sig.get("tp1", bb["middle"]))
        
        # Risk Guardrail: Verify Risk-to-Reward parameters before releasing capital
        risk = abs(price - sl)
        reward = abs(tp1 - price)
        if reward < (risk * 1.1):
            log.warning(f"Trade Blocked: Institutional Risk-to-Reward parameters too narrow ({round(reward/risk, 2)}:1)")
            return

        trade_result = execute_trade(symbol, action, price, sl, tp1)
        if trade_result:
            send_email(f"🐋 WHALE TRADER EXECUTED: {action} {symbol}", f"Bot entered market at ${price}. SL: ${sl} | TP: ${tp1}. Reason: {sig.get('reason')}")
    else:
        log.info("Frame complete. Market conditions non-conducive for high-conviction deployment.")

def main():
    log.info("WhaleTrader Pro V4 Engine Booted. Portfolio Tracking Engaged.")
    for sym in SYMBOLS:
        analyze(sym)
        time.sleep(5)

if __name__ == "__main__":
    main()
  
