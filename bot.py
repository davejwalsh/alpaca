import os
import time
import threading
import numpy as np
import pandas as pd
from flask import Flask, jsonify
import alpaca_trade_api as tradeapi

print("📈 MICRO-QUANT v29 (REALISTIC RISK + STABLE LEARNING CORE)")

# =========================================================
SYMBOLS = ["AAPL","MSFT","NVDA","TSLA","AMD","META","AMZN","GOOGL"]

TIMEFRAME = "1Min"
LOOKBACK = 200
CHECK_INTERVAL = 20

INITIAL_CAPITAL = 500
capital = INITIAL_CAPITAL

RISK_PER_TRADE = 0.01          # 1% risk per trade (realistic micro-account)
MAX_POSITIONS = 5

ATR_PERIOD = 14
ATR_MULT_STOP = 1.5
ATR_MULT_TARGET = 3.0

PORT = int(os.getenv("PORT", 8080))

api = tradeapi.REST(
    os.getenv("APCA_API_KEY_ID"),
    os.getenv("APCA_API_SECRET_KEY"),
    base_url="https://paper-api.alpaca.markets"
)

# =========================================================
positions = {}
trade_journal = []
equity_curve = []

# =========================================================
# WEIGHTS (regime-aware learning system preserved)
# =========================================================
weights = {
    "TREND": {"trend": 1.0, "compression": 1.0, "volume": 1.0, "breakout": 1.0, "volatility": 1.0},
    "CHOP":  {"trend": 1.0, "compression": 1.0, "volume": 1.0, "breakout": 1.0, "volatility": 1.0},
    "VOL":   {"trend": 1.0, "compression": 1.0, "volume": 1.0, "breakout": 1.0, "volatility": 1.0}
}

stats = {"TREND": {"n": 0}, "CHOP": {"n": 0}, "VOL": {"n": 0}}

# =========================================================
def compute_indicators(df):
    df["ma9"] = df["close"].rolling(9).mean()
    df["ma50"] = df["close"].rolling(50).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift())
        )
    )

    df["atr"] = pd.Series(tr).rolling(ATR_PERIOD).mean()
    df["vol_mean"] = df["volume"].rolling(20).mean()

    return df

# =========================================================
def fetch(symbol):
    bars = api.get_bars(symbol, TIMEFRAME, limit=LOOKBACK).df
    return compute_indicators(bars)

# =========================================================
def classify_regime(df):
    close = df["close"].iloc[-1]
    atr = df["atr"].iloc[-1]

    vol = (atr / (close + 1e-9)) * 100

    # smoothed regime (prevents flicker)
    prev_close = df["close"].iloc[-2] if len(df) > 2 else close
    prev_atr = df["atr"].iloc[-2] if len(df) > 2 else atr
    prev_vol = (prev_atr / (prev_close + 1e-9)) * 100

    if vol > 2.0 and prev_vol > 2.0:
        return "VOL"

    return "TREND" if close > df["ma50"].iloc[-1] else "CHOP"

# =========================================================
def features(df):
    close = df["close"].iloc[-1]

    vol_mean = df["vol_mean"].iloc[-1]
    vol_now = df["volume"].iloc[-1]

    return {
        "trend": 1 if close > df["ma50"].iloc[-1] else 0,
        "compression": 1 if df["close"].pct_change().rolling(25).std().iloc[-1] < 0.007 else 0,
        "volume": 1 if (not np.isnan(vol_mean) and vol_now > vol_mean * 1.2) else 0,
        "breakout": 1 if close > df["ma9"].iloc[-1] else 0,
        "volatility": 1 if 0.02 <= (df["atr"].iloc[-1] / (close + 1e-9)) * 100 <= 2 else 0
    }

# =========================================================
def score(f, regime):
    return sum(f[k] * weights[regime][k] for k in f)

# =========================================================
def generate_signal(df):
    reg = classify_regime(df)
    f = features(df)

    if any(pd.isna(v) for v in f.values()):
        return None, None, reg

    s = score(f, reg)

    n = stats[reg]["n"]
    threshold = 2.6 if n < 20 else 2.85

    if s < threshold:
        return None, None, reg

    if f["trend"] and f["breakout"]:
        return "LONG", f, reg

    if not f["trend"] and f["breakout"]:
        return "SHORT", f, reg

    return None, f, reg

# =========================================================
def position_size(price, atr):
    risk_dollars = capital * RISK_PER_TRADE
    stop_distance = atr * ATR_MULT_STOP

    qty = risk_dollars / (stop_distance + 1e-9)
    return max(qty, 0)

# =========================================================
def execute(symbol, side, df, f, reg):
    global capital

    price = df["close"].iloc[-1]
    atr = df["atr"].iloc[-1]

    qty = position_size(price, atr)

    if qty <= 0:
        return

    stop = price - atr * ATR_MULT_STOP if side == "LONG" else price + atr * ATR_MULT_STOP
    target = price + atr * ATR_MULT_TARGET if side == "LONG" else price - atr * ATR_MULT_TARGET

    positions[symbol] = {
        "side": side,
        "entry": price,
        "qty": qty,
        "stop": stop,
        "target": target,
        "atr_entry": atr,
        "regime": reg,
        "features": f
    }

    print(f"📌 ENTRY {symbol} {side} qty={qty:.2f} @ {price:.2f}")

# =========================================================
def update_weights(regime, f, pnl, atr_entry):
    stats[regime]["n"] += 1
    n = stats[regime]["n"]

    confidence = min(1.0, n / 50)

    reward = pnl / (atr_entry + 1e-9)
    reward = np.tanh(reward)

    for k in weights[regime]:
        weights[regime][k] += 0.015 * confidence * reward * (f[k] - 0.5)
        weights[regime][k] = float(np.clip(weights[regime][k], 0.3, 2.0))

# =========================================================
def update_positions():
    global capital

    for symbol, pos in list(positions.items()):
        df = fetch(symbol)
        price = df["close"].iloc[-1]

        pnl_per_share = (price - pos["entry"]) if pos["side"] == "LONG" else (pos["entry"] - price)
        pnl = pnl_per_share * pos["qty"]

        # trailing stop + target
        if pos["side"] == "LONG":
            exit_cond = price <= pos["stop"] or price >= pos["target"]
        else:
            exit_cond = price >= pos["stop"] or price <= pos["target"]

        if not exit_cond:
            continue

        capital += pnl

        update_weights(pos["regime"], pos["features"], pnl, pos["atr_entry"])

        trade_journal.append({
            "symbol": symbol,
            "pnl": pnl,
            "regime": pos["regime"],
            "time": time.time()
        })

        print(f"📤 EXIT {symbol} pnl={pnl:.2f}")

        del positions[symbol]

# =========================================================
def run():
    while True:
        try:
            for symbol in SYMBOLS:
                df = fetch(symbol)

                sig, f, reg = generate_signal(df)

                if sig and symbol not in positions and len(positions) < MAX_POSITIONS:
                    execute(symbol, sig, df, f, reg)

            update_positions()

            equity_curve.append(capital)

            print(f"💰 {capital:.2f} | POS {len(positions)}")

        except Exception as e:
            print("ERROR:", e)

        time.sleep(CHECK_INTERVAL)

# =========================================================
app = Flask(__name__)

@app.route("/status")
def status():
    return jsonify({
        "capital": capital,
        "positions": len(positions),
        "weights": weights,
        "stats": stats
    })

@app.route("/equity")
def equity():
    return jsonify(equity_curve)

@app.route("/positions")
def get_positions():
    return jsonify(positions)

@app.route("/analytics")
def analytics():
    if not trade_journal:
        return {"status": "no trades"}
    df = pd.DataFrame(trade_journal)
    return {
        "trades": len(df),
        "win_rate": float((df["pnl"] > 0).mean()),
        "avg_pnl": float(df["pnl"].mean())
    }

# =========================================================
if __name__ == "__main__":
    threading.Thread(target=run, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
