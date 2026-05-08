import os
import time
import threading
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from flask import Flask, jsonify

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

# strategy tracking
strategy_pnl = {
    "rules": deque(maxlen=200),
    "ml": deque(maxlen=200)
}

strategy_weight = {
    "rules": 0.5,
    "ml": 0.5
}

ml_weights = defaultdict(lambda: np.random.uniform(-0.3, 0.3))

# =========================================================
# UNIVERSE
# =========================================================
SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG",
    "META", "NVDA", "TSLA", "BRK.B", "JPM",
    "V", "UNH", "HD", "PG", "MA",
    "DIS", "BAC", "XOM", "AVGO", "LLY",
    "ADBE", "COST", "PEP", "KO", "CRM",
    "MRK", "ABT", "CVX", "TMO", "WMT",
    "CSCO", "MCD", "ACN", "DHR", "AMD",
    "TXN", "NEE", "LIN", "PM", "UPS",
    "ORCL", "BMY", "QCOM", "LOW", "INTC",
    "SPGI", "CAT", "GS", "MS", "BLK"
]

# =========================================================
# ACCOUNT
# =========================================================
def account():
    return api.get_account()

def equity():
    return float(account().equity)

def positions():
    return api.list_positions()

def exposure():
    eq = equity()
    used = sum(float(p.market_value) for p in positions())
    return used / eq

# =========================================================
# DATA
# =========================================================
def get_data(symbol):
    return api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=80).df

def returns(series, n):
    return series.iloc[-1] / series.iloc[-n] - 1

# =========================================================
# REGIME DETECTION (risk engine)
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
# STRATEGY 1 — RULES
# =========================================================
def rules_signal(df):
    return returns(df["close"], 5) * 0.6 + returns(df["close"], 20) * 0.4

# =========================================================
# STRATEGY 2 — ML (adaptive weights)
# =========================================================
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
# POSITION SIZING (risk-aware)
# =========================================================
def position_size(signal, df):
    eq = equity()

    vol = df["close"].pct_change().rolling(20).std().iloc[-1]
    reg = regime(df)

    base_risk = 0.01

    if reg == "stress":
        base_risk *= 0.3
    elif reg == "trend_up":
        base_risk *= 1.2

    risk_budget = eq * base_risk * abs(signal)

    stop = max(vol * 2, 0.02)

    return max(int(risk_budget / stop), 0)

# =========================================================
# EXECUTION
# =========================================================
def position(symbol):
    try:
        p = api.get_position(symbol)
        return float(p.avg_entry_price), int(p.qty)
    except:
        return None, 0

def trade(symbol, signal, df):
    entry, qty = position(symbol)
    price = df["close"].iloc[-1]

    # ENTRY
    if qty == 0 and signal > 0.5:
        size = position_size(signal, df)
        if size > 0:
            api.submit_order(symbol, size, "buy", "market", "gtc")

    # EXIT
    if qty > 0:
        pnl = (price - entry) / entry

        if pnl < -0.02:
            api.submit_order(symbol, qty, "sell", "market", "gtc")

        if pnl > 0.18:
            api.submit_order(symbol, qty, "sell", "market", "gtc")

# =========================================================
# ENGINE LOOP
# =========================================================
def engine():
    global paused

    while True:
        if paused:
            time.sleep(2)
            continue

        for s in SYMBOLS:
            df = get_data(s)
            if len(df) < 40:
                continue

            r = rules_signal(df)
            m = ml_signal(s, df)

            # ensemble allocation
            signal = strategy_weight["rules"] * r + strategy_weight["ml"] * m

            trade(s, signal, df)

            reward = returns(df["close"], 3)
            ml_learn(s, df, reward)

        equity_curve.append(equity())

        time.sleep(120)

# =========================================================
# WALK-FORWARD EVALUATION (LIVE PROXY BACKTEST)
# =========================================================
def evaluate():
    while True:
        if len(equity_curve) < 100:
            time.sleep(60)
            continue

        recent = list(equity_curve)[-100:]

        growth = recent[-1] - recent[0]
        volatility = np.std(recent)

        score = growth / (volatility + 1e-9)

        # adapt weights
        if score < 0:
            strategy_weight["ml"] += 0.05
            strategy_weight["rules"] -= 0.05
        else:
            strategy_weight["rules"] += 0.03
            strategy_weight["ml"] -= 0.03

        # clamp
        strategy_weight["rules"] = min(max(strategy_weight["rules"], 0.1), 0.9)
        strategy_weight["ml"] = 1 - strategy_weight["rules"]

        time.sleep(300)

# =========================================================
# DRAW DOWN PROTECTION (IMPORTANT)
# =========================================================
def drawdown_guard():
    peak = 0

    while True:
        eq = equity()
        global paused

        if eq > peak:
            peak = eq

        dd = (peak - eq) / peak

        if dd > 0.08:
            paused = True
        elif dd < 0.03:
            paused = False

        time.sleep(60)

# =========================================================
# API
# =========================================================
@app.route("/status")
def status():
    acc = account()
    return jsonify({
        "equity": float(acc.equity),
        "cash": float(acc.cash),
        "weights": strategy_weight,
        "paused": paused
    })

@app.route("/regime")
def get_regime():
    df = get_data("AAPL")
    return {"regime": regime(df)}

@app.route("/weights")
def weights():
    return dict(strategy_weight)

@app.route("/report")
def report():
    acc = api.get_account()
    pos = api.list_positions()

    positions = []
    for p in pos:
        positions.append({
            "symbol": p.symbol,
            "qty": int(p.qty),
            "avg_entry": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl)
        })

    return jsonify({
        "equity": float(acc.equity),
        "cash": float(acc.cash),
        "buying_power": float(acc.buying_power),
        "positions": positions
    })

equity_curve = []  # make sure your engine appends here

@app.route("/performance")
def performance():
    if len(equity_curve) < 10:
        return jsonify({"error": "not enough data"})

    curve = np.array(equity_curve)

    returns = np.diff(curve) / curve[:-1]
    total_return = (curve[-1] / curve[0]) - 1

    volatility = np.std(returns)
    sharpe = np.mean(returns) / (volatility + 1e-9)

    peak = np.maximum.accumulate(curve)
    drawdown = (curve - peak) / peak
    max_dd = np.min(drawdown)

    return jsonify({
        "total_return": float(total_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd)
    })
    
@app.route("/ml_state")
def ml_state():
    sample = dict(list(ml_weights.items())[:20])

    return jsonify({
        "strategy_weights": strategy_weight,
        "ml_sample_weights": sample
    })

@app.route("/debug/signal/<symbol>")
def debug_signal(symbol):
    df = get_data(symbol)

    if len(df) < 40:
        return jsonify({"error": "not enough data"})

    r = rules_signal(df)
    m = ml_signal(symbol, df)

    combined = strategy_weight["rules"] * r + strategy_weight["ml"] * m

    reg = regime(df)

    return jsonify({
        "symbol": symbol,
        "rules_signal": float(r),
        "ml_signal": float(m),
        "combined_signal": float(combined),
        "regime": reg,
        "ml_weight_rules": strategy_weight["rules"],
        "ml_weight_ml": strategy_weight["ml"]
    })

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("🚀 Quant v5 production system starting")

    threading.Thread(target=engine, daemon=True).start()
    threading.Thread(target=evaluate, daemon=True).start()
    threading.Thread(target=drawdown_guard, daemon=True).start()

    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
