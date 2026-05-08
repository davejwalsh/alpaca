import os
import time
import threading
from collections import deque, defaultdict

import numpy as np
import alpaca_trade_api as tradeapi
from flask import Flask, jsonify

print("🔥 HEDGE FUND BOT STARTING")

# =========================================================
# CONFIG
# =========================================================
DEBUG_MODE = True
CHECK_INTERVAL = 30
COOLDOWN = 300

MAX_POSITIONS = 12
MAX_EXPOSURE = 0.80
MIN_SIGNAL = 0.02

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
equity_curve = deque(maxlen=1000)
last_trade_time = {}

strategy_weight = {"rules": 0.5, "ml": 0.5}
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
# ACCOUNT HELPERS
# =========================================================
def equity():
    return float(api.get_account().equity)

def positions():
    return api.list_positions()

def position(symbol):
    try:
        p = api.get_position(symbol)
        return float(p.avg_entry_price), int(p.qty)
    except:
        return None, 0

# =========================================================
# DATA
# =========================================================
def get_data(symbol):
    return api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=80).df

def returns(series, n):
    return series.iloc[-1] / series.iloc[-n] - 1

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
# ENGINE (PORTFOLIO ALLOCATOR)
# =========================================================
def engine():
    global paused

    while True:
        print("\n🔁 PORTFOLIO CYCLE")

        if paused:
            print("⏸️ PAUSED")
            time.sleep(5)
            continue

        acc = api.get_account()
        eq = float(acc.equity)

        current_positions = {p.symbol: p for p in positions()}
        current_exposure = sum(float(p.market_value) for p in current_positions.values()) / eq

        scored = []

        # =====================================================
        # SCAN UNIVERSE
        # =====================================================
        for s in SYMBOLS:
            try:
                df = get_data(s)

                if len(df) < 40:
                    continue

                r = rules_signal(df)
                m = ml_signal(s, df)

                raw_signal = strategy_weight["rules"] * r + strategy_weight["ml"] * m
                momentum = returns(df["close"], 1)

                vol = df["close"].pct_change().rolling(20).std().iloc[-1]
                vol = max(vol, 1e-6)

                signal = (raw_signal + momentum * 0.3) / vol

                if abs(signal) < MIN_SIGNAL:
                    continue

                scored.append((s, signal, df, vol))

            except Exception as e:
                print(f"❌ {s}: {e}")

        if not scored:
            time.sleep(CHECK_INTERVAL)
            continue

        # =====================================================
        # RANK OPPORTUNITIES
        # =====================================================
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        top = scored[:MAX_POSITIONS]

        print(f"📊 TOP: {[x[0] for x in top]}")

        available_capital = max(0, (MAX_EXPOSURE - current_exposure)) * eq

        if available_capital <= 0:
            print("⚠️ Fully invested")
            time.sleep(CHECK_INTERVAL)
            continue

        total_strength = sum(abs(x[1]) for x in top)

        # =====================================================
        # TRADE PORTFOLIO
        # =====================================================
        for symbol, signal, df, vol in top:

            price = df["close"].iloc[-1]
            entry, qty = position(symbol)

            weight = abs(signal) / total_strength if total_strength else 0
            allocation = available_capital * weight

            size = int(allocation / (price * vol))
            size = max(0, min(size, 10))

            print(f"📌 {symbol} sig={signal:.3f} size={size}")

            now = time.time()

            # =================================================
            # ENTRY
            # =================================================
            if qty == 0 and signal > 0 and size > 0:

                if symbol in last_trade_time:
                    if now - last_trade_time[symbol] < COOLDOWN:
                        continue

                print(f"🟢 BUY {symbol} size={size}")
                api.submit_order(symbol, size, "buy", "market", "gtc")
                last_trade_time[symbol] = now

            # =================================================
            # EXIT
            # =================================================
            if qty > 0:
                entry_price = float(entry)
                pnl = (price - entry_price) / entry_price

                if pnl < -0.02:
                    print(f"🛑 STOP {symbol}")
                    api.submit_order(symbol, qty, "sell", "market", "gtc")

                elif pnl > 0.08:
                    print(f"📉 TAKE 8% {symbol}")
                    api.submit_order(symbol, max(1, qty // 2), "sell", "market", "gtc")

                elif pnl > 0.15:
                    print(f"📉 TAKE 15% {symbol}")
                    api.submit_order(symbol, max(1, qty // 2), "sell", "market", "gtc")

                elif pnl > 0.30:
                    print(f"🏆 EXIT {symbol}")
                    api.submit_order(symbol, qty, "sell", "market", "gtc")

        equity_curve.append(eq)

        print(f"💼 EQUITY: {eq:.2f} | EXP: {current_exposure:.2f}")

        time.sleep(CHECK_INTERVAL)

# =========================================================
# API
# =========================================================
@app.route("/")
def home():
    return {"status": "running"}

@app.route("/status")
def status():
    acc = api.get_account()
    return jsonify({
        "equity": float(acc.equity),
        "cash": float(acc.cash),
        "paused": paused
    })

@app.route("/report")
def report():
    pos = api.list_positions()
    return jsonify({
        "positions": [p.symbol for p in pos],
        "count": len(pos)
    })

@app.route("/equity")
def equity_api():
    return {"equity": list(equity_curve)}

# =========================================================
# START
# =========================================================
def start():
    time.sleep(5)
    threading.Thread(target=engine, daemon=True).start()

if __name__ == "__main__":
    print("🚀 STARTING HEDGE FUND BOT")
    threading.Thread(target=start, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
