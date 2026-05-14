import os
import time
import threading
import numpy as np
import pandas as pd
from flask import Flask, jsonify
import alpaca_trade_api as tradeapi

print("📈 MICRO-QUANT v17 (SMALL EQUITY OPTIMIZED)")

# =========================================================
# CONFIG
# =========================================================
SYMBOLS = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","V","UNH",     
           "HD","PG","MA","DIS","BAC","XOM","AVGO","LLY","ADBE","COST",     
           "PEP","KO","CRM","MRK","ABT","CVX","TMO","WMT","CSCO","MCD",     
           "ACN","DHR","AMD","TXN","NEE","LIN","PM","UPS","ORCL","BMY",     
           "QCOM","LOW","INTC","SPGI","CAT","GS","MS","BLK"]

TIMEFRAME = "1Min"
LOOKBACK = 200

CHECK_INTERVAL = 60

INITIAL_CAPITAL = 500
capital = INITIAL_CAPITAL

RISK_PER_TRADE = 0.005
TAKE_PROFIT_MULT = 2.5

ATR_PERIOD = 14
ATR_MIN = 0.1
ATR_MAX = 1.0

TRADE_COOLDOWN = 1800

# =========================================================
# PORT FIX
# =========================================================
PORT = int(os.getenv("PORT", 8080))

# =========================================================
# API
# =========================================================
api = tradeapi.REST(
    os.getenv("APCA_API_KEY_ID"),
    os.getenv("APCA_API_SECRET_KEY"),
    base_url="https://paper-api.alpaca.markets"
)

# =========================================================
# STATE
# =========================================================
equity_curve = []
cash_curve = []

positions = {}
last_trade_time = {}
last_prices = {}
regime = "UNKNOWN"

trade_journal = []

lock = threading.Lock()

# =========================================================
# INDICATORS
# =========================================================
def compute_indicators(df):
    df['ma9'] = df['close'].rolling(9).mean()
    df['ma50'] = df['close'].rolling(50).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()

    return df

# =========================================================
# VOL FILTER
# =========================================================
def volatility_ok(df):
    latest = df.iloc[-1]
    atr_pct = (latest['atr'] / latest['close']) * 100
    return ATR_MIN <= atr_pct <= ATR_MAX

# =========================================================
# REGIME
# =========================================================
def update_regime():
    global regime
    df = fetch_data("SPY")
    latest = df.iloc[-1]
    regime = "BULL" if latest['close'] > latest['ma50'] else "BEAR"

# =========================================================
# SIGNAL
# =========================================================
def generate_signal(df):
    latest = df.iloc[-1]

    if np.isnan(latest['ma50']) or np.isnan(latest['rsi']) or np.isnan(latest['atr']):
        return None

    if not volatility_ok(df):
        return None

    if latest['close'] > latest['ma50']:
        if latest['rsi'] < 35 and latest['close'] > latest['ma9']:
            return "LONG"

    if latest['close'] < latest['ma50']:
        if latest['rsi'] > 65 and latest['close'] < latest['ma9']:
            return "SHORT"

    return None

# =========================================================
# DATA
# =========================================================
def fetch_data(symbol):
    bars = api.get_bars(symbol, TIMEFRAME, limit=LOOKBACK).df
    df = bars[['open', 'high', 'low', 'close', 'volume']].copy()
    return compute_indicators(df)

# =========================================================
# ANALYTICS
# =========================================================
def log_trade(symbol, side, entry, exit_price, pnl):
    trade_journal.append({
        "symbol": symbol,
        "side": side,
        "entry": entry,
        "exit": exit_price,
        "pnl": pnl,
        "time": time.time()
    })

def analytics():
    if not trade_journal:
        return {"status": "no trades yet"}

    df = pd.DataFrame(trade_journal)

    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    win_rate = len(wins) / len(df)

    avg_win = wins["pnl"].mean() if len(wins) else 0
    avg_loss = losses["pnl"].mean() if len(losses) else 0

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    return {
        "trades": len(df),
        "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 3),
        "avg_loss": round(avg_loss, 3),
        "expectancy": round(expectancy, 4)
    }

# =========================================================
# EXECUTION
# =========================================================
def can_trade(symbol):
    if symbol in positions:
        return False

    if symbol not in last_trade_time:
        return True

    return (time.time() - last_trade_time[symbol]) > TRADE_COOLDOWN

def execute_trade(symbol, signal, price, df):
    global capital

    risk_amount = capital * RISK_PER_TRADE

    atr = df['atr'].iloc[-1]

    min_stop_pct = 0.002
    stop_distance = max(atr * 1.2, price * min_stop_pct)

    target_distance = stop_distance * 2.5

    if signal == "LONG":
        stop = price - stop_distance
        target = price + target_distance
    else:
        stop = price + stop_distance
        target = price - target_distance

    positions[symbol] = {
        "side": signal,
        "entry": price,
        "stop": stop,
        "target": target,
        "risk": risk_amount
    }

    last_trade_time[symbol] = time.time()

# =========================================================
# POSITION MANAGEMENT
# =========================================================
def update_positions():
    global capital

    to_close = []

    for symbol, pos in list(positions.items()):
        price = last_prices.get(symbol)
        if price is None:
            continue

        pnl = 0

        if pos["side"] == "LONG":
            if price <= pos["stop"]:
                pnl = -pos["risk"]
                to_close.append((symbol, pnl, price))

            elif price >= pos["target"]:
                pnl = pos["risk"] * TAKE_PROFIT_MULT
                to_close.append((symbol, pnl, price))

        else:
            if price >= pos["stop"]:
                pnl = -pos["risk"]
                to_close.append((symbol, pnl, price))

            elif price <= pos["target"]:
                pnl = pos["risk"] * TAKE_PROFIT_MULT
                to_close.append((symbol, pnl, price))

    for symbol, pnl, exit_price in to_close:
        entry = positions[symbol]["entry"]

        log_trade(symbol, positions[symbol]["side"], entry, exit_price, pnl)

        capital += pnl
        del positions[symbol]

# =========================================================
# MAIN LOOP
# =========================================================
def run():
    global capital

    while True:
        try:
            update_regime()

            for symbol in SYMBOLS:
                df = fetch_data(symbol)
                price = df["close"].iloc[-1]
                last_prices[symbol] = price

                signal = generate_signal(df)

                if signal and can_trade(symbol):
                    if (regime == "BULL" and signal == "LONG") or \
                       (regime == "BEAR" and signal == "SHORT"):
                        execute_trade(symbol, signal, price, df)

            update_positions()

            equity_curve.append(round(capital, 2))
            cash_curve.append(round(capital, 2))

            print(f"💰 {capital:.2f} | Pos: {len(positions)} | Regime: {regime}")

        except Exception as e:
            print("Error:", e)

        time.sleep(CHECK_INTERVAL)

# =========================================================
# API
# =========================================================
app = Flask(__name__)

@app.route("/equity")
def equity():
    return jsonify(equity_curve)

@app.route("/cash")
def cash():
    return jsonify(cash_curve)

@app.route("/positions")
def get_positions():
    return jsonify(positions)

@app.route("/portfolio")
def portfolio():
    return jsonify({
        "capital": round(capital, 2),
        "positions": len(positions),
        "symbols": list(positions.keys())
    })

@app.route("/status")
def status():
    return jsonify({
        "capital": round(capital, 2),
        "positions": len(positions),
        "regime": regime
    })

@app.route("/analytics")
def analytics_route():
    return jsonify(analytics())

# =========================================================
# START
# =========================================================
if __name__ == "__main__":
    t = threading.Thread(target=run, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=PORT)
