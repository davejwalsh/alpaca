import os
import time
import threading
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from flask import Flask, jsonify, send_file

import matplotlib.pyplot as plt
import io

print("🔥 FILE STARTED")

# =========================================================
# CONFIG
# =========================================================
DEBUG_MODE = True
CHECK_INTERVAL = 30

# =========================================================
# API
# =========================================================
api = tradeapi.REST(
    os.getenv("APCA_API_KEY_ID"),
    os.getenv("APCA_API_SECRET_KEY"),
    "https://paper-api.alpaca.markets"
)

app = Flask(__name__)

# =========================================================
# STATE
# =========================================================
paused = False
first_trade_done = False

equity_curve = deque(maxlen=1000)

strategy_weight = {
    "rules": 0.5,
    "ml": 0.5
}

ml_weights = defaultdict(lambda: np.random.uniform(-0.3, 0.3))

# =========================================================
# UNIVERSE
# =========================================================
SYMBOLS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","V","UNH",
    "HD","PG","MA","DIS","BAC","XOM","AVGO","LLY","ADBE","COST",
    "PEP","KO","CRM","MRK","ABT","CVX","TMO","WMT","CSCO","MCD",
    "ACN","DHR","AMD","TXN","NEE","LIN","PM","UPS","ORCL","BMY",
    "QCOM","LOW","INTC","SPGI","CAT","GS","MS","BLK"
]

# =========================================================
# ACCOUNT
# =========================================================
def equity():
    return float(api.get_account().equity)

def positions():
    return api.list_positions()

# =========================================================
# DATA
# =========================================================
def get_data(symbol):
    return api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=80).df

def returns(series, n):
    return series.iloc[-1] / series.iloc[-n] - 1

# =========================================================
# REGIME
# =========================================================
def regime(df):
    vol = df["close"].pct_change().rolling(20).std().iloc[-1]
    trend = returns(df["close"], 10)

    if vol > 0.03:
        return "stress"
    if trend > 0.01:
        return "trend_up"
    if trend < -0.01:
        return "trend_down"
    return "neutral"

# =========================================================
# SIGNALS
# =========================================================
def rules_signal(df):
    return returns(df["close"], 5) * 0.6 + returns(df["close"], 20) * 0.4

def ml_features(df):
    return {
        "r5": returns(df["close"], 5),
        "r10": returns(df["close"], 10),
        "r20": returns(df["close"], 20)
    }

def ml_signal(symbol, df):
    f = ml_features(df)
    return sum(f[k] * ml_weights[f"{symbol}_{k}"] for k in f)

def ml_learn(symbol, df, reward):
    f = ml_features(df)
    for k in f:
        ml_weights[f"{symbol}_{k}"] += 0.01 * reward * f[k]

# =========================================================
# POSITION
# =========================================================
def position(symbol):
    try:
        p = api.get_position(symbol)
        return float(p.avg_entry_price), int(p.qty)
    except:
        return None, 0

# =========================================================
# EXECUTION
# =========================================================
def trade(symbol, signal, df):
    global first_trade_done

    entry, qty = position(symbol)
    price = df["close"].iloc[-1]

    print(f"📊 {symbol} | signal={signal:.4f} | price={price:.2f}")

    # FORCE TEST TRADE
    if DEBUG_MODE and not first_trade_done:
        print("🚀 FORCED TEST TRADE")
        api.submit_order(symbol="AAPL", qty=1, side="buy", type="market", time_in_force="gtc")
        first_trade_done = True
        return

    threshold = 0.1 if DEBUG_MODE else 0.5

    if qty == 0 and signal > threshold:
        size = 1 if DEBUG_MODE else int(100 * abs(signal))
        print(f"✅ BUY {symbol} size={size}")
        api.submit_order(symbol, size, "buy", "market", "gtc")

    if qty > 0:
        pnl = (price - entry) / entry
        print(f"💰 {symbol} PnL={pnl:.3f}")

        if pnl < -0.02:
            print(f"🛑 STOP LOSS {symbol}")
            api.submit_order(symbol, qty, "sell", "market", "gtc")

        elif pnl > 0.05 and DEBUG_MODE:
            print(f"🎯 QUICK TAKE PROFIT {symbol}")
            api.submit_order(symbol, qty, "sell", "market", "gtc")

        elif pnl > 0.18:
            print(f"🏆 FULL TAKE PROFIT {symbol}")
            api.submit_order(symbol, qty, "sell", "market", "gtc")

# =========================================================
# ENGINE
# =========================================================
def engine():
    global paused

    while True:
        print("\n🔁 NEW SCAN CYCLE")

        if paused:
            print("⏸️ PAUSED")
            time.sleep(5)
            continue

        for s in SYMBOLS:
            try:
                df = get_data(s)

                if len(df) < 40:
                    print(f"⚠️ Not enough data for {s}")
                    continue

                r = rules_signal(df)
                m = ml_signal(s, df)
                signal = strategy_weight["rules"] * r + strategy_weight["ml"] * m

                print(f"🔍 {s} | rules={r:.4f} ml={m:.4f} combined={signal:.4f}")

                trade(s, signal, df)

                reward = returns(df["close"], 3)
                ml_learn(s, df, reward)

            except Exception as e:
                print(f"❌ ERROR {s}: {e}")

        eq = equity()
        equity_curve.append(eq)
        print(f"💼 Equity: {eq}")

        time.sleep(CHECK_INTERVAL)

# =========================================================
# SAFE STARTUP (CRITICAL FIX)
# =========================================================
def start_background():
    print("⏳ Delaying engine start (Railway health check)...")
    time.sleep(5)  # <-- critical for Railway

    print("🚀 Starting engine thread")
    threading.Thread(target=engine, daemon=True).start()

# =========================================================
# API
# =========================================================
@app.route("/")
def home():
    return {"status": "ok"}

@app.route("/status")
def status():
    acc = api.get_account()
    return jsonify({
        "equity": float(acc.equity),
        "cash": float(acc.cash),
        "weights": strategy_weight,
        "paused": paused
    })

@app.route("/report")
def report():
    acc = api.get_account()
    pos = api.list_positions()

    return jsonify({
        "equity": float(acc.equity),
        "positions": [p.symbol for p in pos]
    })
@app.route("/trades")
def trades():
    try:
        activities = api.get_activities(activity_types="FILL")

        trades = []
        for a in activities[:20]:  # last 20 trades
            trades.append({
                "symbol": a.symbol,
                "side": a.side,
                "qty": a.qty,
                "price": a.price,
                "time": a.transaction_time
            })

        return jsonify(trades)

    except Exception as e:
        return {"error": str(e)}

@app.route("/equity-graph.png")
def equity_graph():
    if len(equity_curve) < 5:
        return {"error": "not enough data"}

    plt.figure()
    plt.plot(list(equity_curve))
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype="image/png")

@app.route("/equity-history")
def equity_history():
    return {
        "equity": list(equity_curve)
    }

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("🚀 DEBUG MODE ACTIVE")

    # start background AFTER flask is ready
    threading.Thread(target=start_background, daemon=True).start()

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
