import os
import time
import threading
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from flask import Flask, jsonify

print("🔥 QUANT SYSTEM v6.1 (CLEAN PORTFOLIO ARCH)")

# =========================================================
# CONFIG
# =========================================================
CHECK_INTERVAL = 30
COOLDOWN = 300

MAX_POSITIONS = 12
MAX_EXPOSURE = 0.80
MIN_SIGNAL = 0.03

ML_LR = 0.002
MAX_WEIGHT = 1.0

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
equity_curve = deque(maxlen=2000)
cash_curve = deque(maxlen=2000)

last_trade_time = {}
paused = False

strategy_weight = {"rules": 0.45, "ml": 0.55}
ml_weights = defaultdict(float)

# =========================================================
# UNIVERSE
# =========================================================
SYMBOLS = list(set([
    "AAPL","TSLA","NVDA","AMD","SPY",
    "PEP","KO","CRM","MRK","ABT","CVX","TMO","WMT","CSCO","MCD",
    "ACN","DHR","TXN","NEE","LIN","PM","UPS","ORCL","BMY",
    "QCOM","LOW","INTC","SPGI","CAT","GS","MS","BLK",
    "F","SOFI","PBR","T","CMCSA","DKNG","HPQ",
    "NOK","BAC","WFC","C","CSX","KMI","VZ","UAL","DAL","CCL",
    "RIVN","LCID","PLTR","OPEN","CHWY","SNAP",
    "ROKU","COIN","AFRM","UPST","SHOP","SQ","PYPL",
    "RIOT","MARA","RUN","ENPH",
    "XOM","OXY","SLB","HAL","EOG",
    "ADBE","NOW","CRWD","ZS","OKTA","NET","DDOG",
    "JPM","SCHW","AXP",
    "NKE","SBUX","TGT","COST","HD",
    "GM",
    "PFE","JNJ","LLY","GILD","BIIB",
    "UBER","LYFT","ABNB","ETSY","EBAY"
]))

# =========================================================
# SAFE WRAPPERS (FIX)
# =========================================================
def safe_get_account():
    try:
        return api.get_account()
    except Exception as e:
        print("ACCOUNT FETCH ERROR:", e)
        return None


def safe_get_bars(symbol):
    try:
        return api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=120).df
    except Exception as e:
        print(f"BARS ERROR {symbol}: {e}")
        return pd.DataFrame()

# =========================================================
# HELPERS
# =========================================================
def equity():
    acc = safe_get_account()
    return float(acc.equity) if acc else 0.0


def cash():
    acc = safe_get_account()
    return float(acc.cash) if acc else 0.0


def positions():
    try:
        return api.list_positions()
    except Exception as e:
        print("POSITIONS ERROR:", e)
        return []


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
    df = safe_get_bars(symbol)
    if df is None or df.empty:
        return pd.DataFrame()
    return df


def returns(series, n):
    if len(series) < n:
        return 0
    return series.iloc[-1] / series.iloc[-n] - 1

# =========================================================
# REGIME (FIXED SAFE)
# =========================================================
def market_regime():
    try:
        df = get_data("SPY")

        if df is None or df.empty or len(df) < 30:
            return "neutral"

        close = df["close"]

        vol = close.pct_change().rolling(30).std().iloc[-1]
        trend = returns(close, 30)

        if np.isnan(vol) or np.isnan(trend):
            return "neutral"

        if vol > 0.035:
            return "stress"
        if vol > 0.02 and trend < 0:
            return "risk_off"
        if trend > 0.01:
            return "risk_on"

        return "neutral"

    except Exception as e:
        print("REGIME ERROR:", e)
        return "neutral"


def regime_multiplier(r):
    return {
        "risk_on": 1.25,
        "neutral": 1.0,
        "risk_off": 0.6,
        "stress": 0.25
    }.get(r, 1.0)

# =========================================================
# SIGNALS
# =========================================================
def rules_signal(df):
    r5 = returns(df["close"], 5)
    r20 = returns(df["close"], 20)
    vol = df["close"].pct_change().rolling(20).std().iloc[-1] + 1e-6
    return np.tanh((0.7 * r5 + 0.3 * r20) / vol)


def ml_features(df):
    return {
        "r5": returns(df["close"], 5),
        "r10": returns(df["close"], 10),
        "r20": returns(df["close"], 20),
    }


def ml_signal(symbol, df):
    f = ml_features(df)
    score = sum(f[k] * ml_weights[f"{symbol}_{k}"] for k in f)
    return np.tanh(score)

# =========================================================
# RISK
# =========================================================
def exposure():
    try:
        acc = safe_get_account()
        if not acc:
            return 0.0

        eq = float(acc.equity)
        pos = positions()

        return sum(float(p.market_value) for p in pos) / eq if eq else 0.0

    except Exception as e:
        print("EXPOSURE ERROR:", e)
        return 0.0

# =========================================================
# ML
# =========================================================
def ml_reward(entry, price, qty):
    if qty <= 0 or entry is None:
        return 0
    pnl = (price - entry) / entry
    return np.tanh(pnl * 6)


def update_ml(symbol, df, reward):
    f = ml_features(df)

    for k, v in f.items():
        key = f"{symbol}_{k}"
        grad = reward * v

        ml_weights[key] += ML_LR * np.clip(grad, -1, 1)
        ml_weights[key] = np.clip(ml_weights[key], -MAX_WEIGHT, MAX_WEIGHT)

# =========================================================
# EXECUTION
# =========================================================
def execute(symbol, signal, df, size):
    price = df["close"].iloc[-1]
    entry, qty = position(symbol)

    now = time.time()

    if symbol in last_trade_time:
        if now - last_trade_time[symbol] < COOLDOWN:
            return

    if qty == 0 and signal > 0 and size > 0:
        print(f"🟢 BUY {symbol} size={size}")
        api.submit_order(symbol, size, "buy", "market", "gtc")
        last_trade_time[symbol] = now

    if qty > 0:
        pnl = (price - entry) / entry

        if pnl < -0.02:
            api.submit_order(symbol, qty, "sell", "market", "gtc")
        elif pnl > 0.10:
            api.submit_order(symbol, qty // 2, "sell", "market", "gtc")
        elif pnl > 0.20:
            api.submit_order(symbol, qty, "sell", "market", "gtc")

# =========================================================
# ENGINE
# =========================================================
def engine():
    global paused

    while True:
        print("\n🔁 CYCLE RUNNING |", time.strftime("%H:%M:%S"), flush=True)

        if paused:
            time.sleep(5)
            continue

        try:
            acc = safe_get_account()
            if not acc:
                time.sleep(10)
                continue

            eq = float(acc.equity)
            reg = market_regime()
            reg_mult = regime_multiplier(reg)
            current_exp = exposure()

            scored = []

            for s in SYMBOLS:
                df = get_data(s)
                if df.empty or len(df) < 60:
                    continue

                r = rules_signal(df)
                m = ml_signal(s, df)

                signal = (strategy_weight["rules"] * r +
                          strategy_weight["ml"] * m)

                momentum = returns(df["close"], 1)

                signal = np.tanh((signal + 0.3 * momentum) * reg_mult)

                if abs(signal) < MIN_SIGNAL:
                    continue

                scored.append((s, signal, df))

            if not scored:
                time.sleep(CHECK_INTERVAL)
                continue

            scored.sort(key=lambda x: abs(x[1]), reverse=True)
            top = scored[:MAX_POSITIONS]

            available = max(0, (MAX_EXPOSURE - current_exp)) * eq
            total = sum(abs(x[1]) for x in top) + 1e-9

            print(f"📊 REGIME={reg} TOP={[x[0] for x in top]}")

            for symbol, signal, df in top:
                price = df["close"].iloc[-1]
                entry, qty = position(symbol)

                weight = abs(signal) / total
                allocation = available * weight

                size = int(allocation / price)
                size = max(0, min(size, 15))

                execute(symbol, signal, df, size)

                reward = ml_reward(entry, price, qty)
                update_ml(symbol, df, reward)

            equity_curve.append(eq)
            cash_curve.append(float(acc.cash))

            print(f"💼 EQ={eq:.2f} EXP={current_exp:.2f}")

        except Exception as e:
            print("ENGINE ERROR:", e)

        time.sleep(CHECK_INTERVAL)

# =========================================================
# API
# =========================================================
@app.route("/")
def home():
    return {"status": "running"}


@app.route("/status")
def status():
    acc = safe_get_account()
    if not acc:
        return jsonify({"error": "account unavailable"}), 503

    return jsonify({
        "equity": float(acc.equity),
        "cash": float(acc.cash),
        "regime": market_regime(),
        "exposure": exposure()
    })


@app.route("/portfolio")
def portfolio():
    try:
        return jsonify([
            {
                "symbol": p.symbol,
                "qty": int(p.qty),
                "value": float(p.market_value)
            }
            for p in positions()
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ml")
def ml_state():
    return {"sample_weights": dict(list(ml_weights.items())[:30])}


@app.route("/equity")
def eq():
    return {"equity": list(equity_curve)}


@app.route("/cash")
def cs():
    return {"cash": list(cash_curve)}

# =========================================================
# START
# =========================================================
def start():
    time.sleep(5)
    threading.Thread(target=engine, daemon=True).start()


if __name__ == "__main__":
    print("🚀 QUANT SYSTEM v6.1 START")
    threading.Thread(target=start, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
