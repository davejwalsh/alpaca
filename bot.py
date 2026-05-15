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

SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMD", "SPY",   
           "PEP","KO","CRM","MRK","ABT","CVX","TMO","WMT","CSCO","MCD",     
           "ACN","DHR","AMD","TXN","NEE","LIN","PM","UPS","ORCL","BMY",     
           "QCOM","LOW","INTC","SPGI","CAT","GS","MS","BLK"]

TIMEFRAME = "1Min"
LOOKBACK = 200

RISK_PER_TRADE = 0.01
STOP_LOSS_PCT = 0.005
TAKE_PROFIT_PCT = 0.01

MIN_MOVE = 0.0015
MAX_SPREAD = 0.002
COOLDOWN_SECONDS = 300
DAILY_LOSS_LIMIT = -0.03

# ==========================================

app = Flask(__name__)
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)

positions = {}
last_trade_time = {}
daily_start_equity = None
account_cache = {"equity": None, "ts": 0}

# ============== SAFE API WRAPPER ===================

def safe_get_account():
    try:
        return api.get_account()
    except Exception as e:
        print("Account fetch error:", e)
        return None

def get_equity():
    acc = safe_get_account()
    if not acc:
        return None
    try:
        return float(acc.equity)
    except:
        return None

def get_daily_pnl():
    global daily_start_equity

    equity = get_equity()
    if equity is None:
        return 0.0

    if daily_start_equity is None:
        daily_start_equity = equity
        return 0.0

    if daily_start_equity == 0:
        return 0.0

    return (equity - daily_start_equity) / daily_start_equity

# ============== DATA ===================

def get_data(symbol):
    try:
        bars = api.get_bars(symbol, TIMEFRAME, limit=LOOKBACK).df
    except Exception as e:
        print("Data error:", symbol, e)
        return pd.DataFrame()

    if bars is None or bars.empty:
        return pd.DataFrame()

    if "symbol" in bars.columns:
        bars = bars[bars["symbol"] == symbol]

    return bars

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df = df.copy()

    df['rsi'] = compute_rsi(df['close'])
    df['ma_fast'] = df['close'].rolling(10).mean()
    df['ma_slow'] = df['close'].rolling(50).mean()
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['volume_avg'] = df['volume'].rolling(20).mean()

    return df

# ============== SIGNAL ENGINE ===================

def generate_signal(symbol, df):

    if df is None or df.empty or len(df) < 60:
        return None

    df = compute_indicators(df)
    latest = df.iloc[-1]

    if pd.isna(latest["close"]):
        return None

    price = latest["close"]

    # spread filter
    spread = abs(df["high"].iloc[-1] - df["low"].iloc[-1]) / (price + 1e-9)
    if spread > MAX_SPREAD:
        return None

    # min move
    recent_move = abs(df["close"].pct_change().iloc[-1])
    if pd.isna(recent_move) or recent_move < MIN_MOVE:
        return None

    # volume filter
    if latest["volume"] < latest["volume_avg"]:
        return None

    trend_up = latest["ma_fast"] > latest["ma_slow"]
    trend_down = latest["ma_fast"] < latest["ma_slow"]

    rsi = latest["rsi"]
    volatility = df["volatility"].iloc[-1]

    compression = volatility < 0.01

    lookback_drop = (df["close"].iloc[-20] - price) / (df["close"].iloc[-20] + 1e-9)

    breakout_up = (
        compression and
        lookback_drop > 0.02 and
        price > df["close"].rolling(10).mean().iloc[-1] and
        df["close"].iloc[-1] > df["close"].iloc[-2]
    )

    breakout_down = (
        compression and
        lookback_drop < -0.02 and
        price < df["close"].rolling(10).mean().iloc[-1]
    )

    signal = None

    if breakout_up:
        signal = "BUY"
    elif breakout_down:
        signal = "SELL"
    elif trend_up and rsi < 40:
        signal = "BUY"
    elif trend_down and rsi > 60:
        signal = "SELL"

    print(f"[DEBUG] {symbol} price={price:.2f} signal={signal} rsi={rsi:.2f}")

    return signal

# ============== EXECUTION ===================

def place_trade(symbol, signal, price):

    open_positions = []
    try:
        open_positions = [p.symbol for p in api.list_positions()]
    except:
        pass

    if symbol in open_positions:
        return

    if get_daily_pnl() < DAILY_LOSS_LIMIT:
        print("Daily loss limit hit.")
        return

    now = time.time()
    if symbol in last_trade_time and now - last_trade_time[symbol] < COOLDOWN_SECONDS:
        return

    equity = get_equity()
    if equity is None:
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
        print(f"TRADE: {signal} {symbol} qty={qty}")

    except Exception as e:
        print("Order error:", e)

# ============== LOOP ===================

def trading_loop():
    while True:
        try:
            for symbol in SYMBOLS:
                df = get_data(symbol)

                if df.empty:
                    continue

                signal = generate_signal(symbol, df)

                if signal:
                    price = df["close"].iloc[-1]
                    place_trade(symbol, signal, price)

        except Exception as e:
            print("Loop error:", e)

        time.sleep(10)

# ============== API ===================

@app.route("/")
def home():
    return jsonify({"status": "running"})

@app.route("/pnl")
def pnl():
    return jsonify({"daily_pnl": get_daily_pnl()})

@app.route("/account")
def account():
    acc = safe_get_account()
    if not acc:
        return jsonify({"error": "account unavailable"}), 503

    return jsonify({
        "equity": float(acc.equity),
        "cash": float(acc.cash),
        "buying_power": float(acc.buying_power),
        "portfolio_value": float(acc.portfolio_value),
        "status": acc.status
    })

@app.route("/equity")
def equity():
    equity = get_equity()
    if equity is None:
        return jsonify({"error": "equity unavailable"}), 503

    return jsonify({
        "current_equity": equity,
        "start_equity": daily_start_equity,
        "daily_pnl_pct": get_daily_pnl()
    })

@app.route("/portfolio")
def portfolio():
    try:
        positions = api.list_positions()
        data = []

        for p in positions:
            data.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc)
            })

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============== START ===================

if __name__ == "__main__":
    threading.Thread(target=trading_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
