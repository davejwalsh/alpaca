import os
import time
import threading
import numpy as np
import pandas as pd
from flask import Flask, jsonify
import alpaca_trade_api as tradeapi

# ================= CONFIG =================

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

SYMBOLS = [
    "AAPL","TSLA","NVDA","AMD","SPY",
    "PEP","KO","CRM","MRK","ABT","CVX","TMO","WMT","CSCO","MCD",
    "ACN","DHR","AMD","TXN","NEE","LIN","PM","UPS","ORCL","BMY",
    "QCOM","LOW","INTC","SPGI","CAT","GS","MS","BLK"
]

TIMEFRAME = "1Min"
LOOKBACK = 200

RISK_PER_TRADE = 0.01
STOP_LOSS_PCT = 0.005
COOLDOWN_SECONDS = 300
DAILY_LOSS_LIMIT = -0.03

# ================= APP =================

app = Flask(__name__)
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)

positions = {}
last_trade_time = {}

cached_equity = None
cached_account = None
daily_start_equity = None
lock = threading.Lock()

# ================= SAFE ACCOUNT THREAD =================

def account_updater():
    global cached_equity, cached_account, daily_start_equity

    while True:
        try:
            acc = api.get_account()

            with lock:
                cached_account = acc
                cached_equity = float(acc.equity)

                if daily_start_equity is None:
                    daily_start_equity = cached_equity

        except Exception as e:
            print("Account update error:", e)

        time.sleep(10)

# ================= HELPERS =================

def get_equity():
    with lock:
        return cached_equity

def get_account():
    with lock:
        return cached_account

def get_daily_pnl():
    eq = get_equity()
    if eq is None or daily_start_equity is None:
        return 0.0
    if daily_start_equity == 0:
        return 0.0
    return (eq - daily_start_equity) / daily_start_equity

# ================= DATA =================

def get_data(symbol):
    try:
        bars = api.get_bars(symbol, TIMEFRAME, limit=LOOKBACK).df
        if bars is None or bars.empty:
            return pd.DataFrame()

        if "symbol" in bars.columns:
            bars = bars[bars["symbol"] == symbol]

        return bars

    except Exception as e:
        print("Data error:", symbol, e)
        return pd.DataFrame()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"])
    df["ma_fast"] = df["close"].rolling(10).mean()
    df["ma_slow"] = df["close"].rolling(50).mean()
    df["volatility"] = df["close"].pct_change().rolling(20).std()
    df["volume_avg"] = df["volume"].rolling(20).mean()
    return df


def log_decision(symbol, price, signal, reason):
    print(f"""
==============================
SYMBOL: {symbol}
PRICE: {price:.2f}
SIGNAL: {signal}
REASON: {reason}
TIME: {time.strftime('%H:%M:%S')}
==============================
""", flush=True)


# ================= SIGNAL =================

def generate_signal(symbol, df):
    if df is None or df.empty or len(df) < 60:
        return None

    df = compute_indicators(df)
    latest = df.iloc[-1]

    price = latest["close"]

    try:
        spread = abs(df["high"].iloc[-1] - df["low"].iloc[-1]) / (price + 1e-9)
        if spread > 0.002:
            return None

        recent_move = abs(df["close"].pct_change().iloc[-1])
        if np.isnan(recent_move) or recent_move < 0.0015:
            return None

        if latest["volume"] < latest["volume_avg"]:
            return None

        trend_up = latest["ma_fast"] > latest["ma_slow"]
        trend_down = latest["ma_fast"] < latest["ma_slow"]

        rsi = latest["rsi"]
        volatility = df["volatility"].iloc[-1]

        compression = volatility < 0.01

        lookback_drop = (df["close"].iloc[-20] - price) / (df["close"].iloc[-20] + 1e-9)

        breakout_up = compression and lookback_drop > 0.02
        breakout_down = compression and lookback_drop < -0.02

        if breakout_up:
            return "BUY"
        if breakout_down:
            return "SELL"
        if trend_up and rsi < 40:
            return "BUY"
        if trend_down and rsi > 60:
            return "SELL"

    except Exception as e:
        print("Signal error:", symbol, e)
    log_decision(symbol, price, None, f"No signal (RSI={rsi:.2f})")
    return None

# ================= EXECUTION =================

def place_trade(symbol, signal, price):

    try:
        open_positions = [p.symbol for p in api.list_positions()]
    except:
        open_positions = []

    if symbol in open_positions:
        return

    if get_daily_pnl() < DAILY_LOSS_LIMIT:
        return

    now = time.time()
    if symbol in last_trade_time and now - last_trade_time[symbol] < COOLDOWN_SECONDS:
        return

    equity = get_equity()
    if not equity:
        return

    risk = equity * RISK_PER_TRADE
    stop_distance = price * STOP_LOSS_PCT

    if stop_distance <= 0:
        return

    qty = int(risk / stop_distance)
    if qty <= 0:
        return

    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy" if signal == "BUY" else "sell",
            type="market",
            time_in_force="gtc"
        )

        last_trade_time[symbol] = now
        print("TRADE:", signal, symbol, qty)

    except Exception as e:
        print("Order error:", e)
    print(f"EXECUTING TRADE: {signal} {symbol} qty={qty} price={price}", flush=True)

# ================= LOOP =================

def trading_loop():
    while True:
        try:
            for symbol in SYMBOLS:
                df = get_data(symbol)
                if df.empty:
                    continue

                signal = generate_signal(symbol, df)

                if signal:
                    place_trade(symbol, signal, df["close"].iloc[-1])

        except Exception as e:
            print("Loop error:", e)

        time.sleep(10)

# ================= ENDPOINTS =================

@app.route("/")
def home():
    return jsonify({"status": "running"})

@app.route("/equity")
def equity():
    eq = get_equity()
    if eq is None:
        return jsonify({"error": "no equity"}), 503

    return jsonify({
        "equity": eq,
        "start": daily_start_equity,
        "pnl": get_daily_pnl()
    })

@app.route("/account")
def account():
    acc = get_account()
    if not acc:
        return jsonify({"error": "no account"}), 503

    return jsonify({
        "equity": float(acc.equity),
        "cash": float(acc.cash),
        "buying_power": float(acc.buying_power)
    })

@app.route("/portfolio")
def portfolio():
    try:
        pos = api.list_positions()
        return jsonify([
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "unrealized": float(p.unrealized_pl)
            }
            for p in pos
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= START =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 🚨 FIX FOR RAILWAY

    threading.Thread(target=account_updater, daemon=True).start()
    threading.Thread(target=trading_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=port)
