import os
import time
import threading
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from flask import Flask, jsonify

print("🏛️ MICRO-QUANT v11 (WALK-FORWARD RESEARCH CORE)")

# =========================================================
# CONFIG
# =========================================================
CHECK_INTERVAL = 60
SCAN_LIMIT = 120
TOP_K = 10

MAX_GROSS = 1.0
MAX_POS = 0.12

MIN_HISTORY = 120
EPS = 1e-9

FEE = 0.001
SLIP = 0.0005
MAX_DRAWDOWN = 0.12

WINDOW = 200
TRAIN_SIZE = 120

CACHE = {}
CACHE_TTL = 30

# =========================================================
# API
# =========================================================
api = tradeapi.REST(
    os.getenv("APCA_API_KEY_ID"),
    os.getenv("APCA_API_SECRET_KEY"),
    "https://paper-api.alpaca.markets"
)

app = Flask(__name__)

lock = threading.Lock()

equity_curve = deque(maxlen=20000)
order_log = deque(maxlen=10000)

strategy_stats = defaultdict(lambda: {"sharpe": 0, "drift": 0, "count": 0})

# =========================================================
# UNIVERSE
# =========================================================
def universe():
    assets = api.list_assets()

    core = [
        a.symbol for a in assets
        if a.tradable and a.status == "active"
        and a.exchange in ["NASDAQ", "NYSE"]
    ]

    priority = ["AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","V","UNH","HD","PG","MA","DIS","BAC","XOM","AVGO","LLY","ADBE","COST","PEP","KO","CRM","MRK","ABT","CVX","TMO","WMT","CSCO","MCD","ACN","DHR","AMD","TXN","NEE","LIN","PM","UPS","ORCL","BMY","QCOM","LOW","INTC","SPGI","CAT","GS","MS","BLK"]

    return sorted(core, key=lambda x: (x not in priority, x))[:SCAN_LIMIT]


SYMBOLS = universe()

# =========================================================
# DATA
# =========================================================
def bars(symbol):
    now = time.time()

    if symbol in CACHE:
        ts, df = CACHE[symbol]
        if now - ts < CACHE_TTL:
            return df

    try:
        df = api.get_bars(symbol, "1Min", limit=400).df

        if df is None or len(df) < MIN_HISTORY:
            return None

        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)

        df = df.sort_index().dropna()

        CACHE[symbol] = (now, df)
        return df

    except:
        return None

# =========================================================
# FEATURES
# =========================================================
def features(close):
    df = pd.DataFrame(index=close.index)

    df["ret"] = close.pct_change()
    df["vol"] = df["ret"].rolling(20).std()
    df["mom"] = df["ret"].rolling(20).mean()
    df["high"] = close.rolling(20).max()

    return df.shift(1).dropna()

# =========================================================
# ALPHA (SAME CORE SIGNAL)
# =========================================================
def alpha(f, price):

    vol = max(f["vol"].iloc[-1], EPS)

    mom = f["mom"].iloc[-1] / vol
    rev = -f["ret"].iloc[-1] / vol
    brk = (price - f["high"].iloc[-1]) / (f["high"].iloc[-1] + EPS)

    a = 0.4 * mom + 0.4 * rev + 0.2 * brk

    return float(np.tanh(np.clip(a, -4, 4))), float(vol)

# =========================================================
# WALK FORWARD VALIDATION (KEY ADDITION)
# =========================================================
def walk_forward(df):

    close = df["close"].values

    if len(close) < WINDOW:
        return 0, 0

    train = close[:TRAIN_SIZE]
    test = close[TRAIN_SIZE:WINDOW]

    pnl = []

    for i in range(30, len(test)):

        window = pd.Series(np.concatenate([train, test[:i]]))

        f = features(window)

        if len(f) < 30:
            continue

        a, _ = alpha(f, window.iloc[-1])

        pnl.append(a)

    if len(pnl) < 10:
        return 0, 0

    pnl = np.array(pnl)

    sharpe = pnl.mean() / (pnl.std() + EPS)
    drift = abs(pnl[-1] - pnl.mean())

    return float(sharpe), float(drift)

# =========================================================
# REGIME DETECTION
# =========================================================
def market_regime(vols):

    v = np.mean(vols)

    if v < 0.01:
        return 1.0  # calm
    elif v < 0.02:
        return 0.6  # normal
    return 0.3      # stressed

# =========================================================
# PORTFOLIO
# =========================================================
def portfolio(alphas, vols, regime):

    alphas = np.array(alphas)

    z = (alphas - np.mean(alphas)) / (np.std(alphas) + EPS)

    w = z / (np.array(vols) + EPS)

    w = w / (np.sum(np.abs(w)) + EPS)

    w *= regime

    return np.clip(w, -MAX_POS, MAX_POS)

# =========================================================
# STATE
# =========================================================
def account():
    return api.get_account()


def positions():
    try:
        return {p.symbol: float(p.qty) for p in api.list_positions()}
    except:
        return {}

# =========================================================
# EXPOSURE
# =========================================================
def exposure(pos, prices, eq):

    gross = 0

    for s, q in pos.items():
        p = prices.get(s)
        if p is None:
            continue
        gross += abs(q * p)

    return gross / (eq + EPS)

# =========================================================
# ENGINE (NOW RESEARCH-AWARE)
# =========================================================
def engine():

    peak = 0

    while True:

        alphas, vols, syms, prices = [], [], [], []

        for s in SYMBOLS:

            df = bars(s)
            if df is None:
                continue

            f = features(df["close"])
            if len(f) < 30:
                continue

            a, v = alpha(f, df["close"].iloc[-1])

            sh, dr = walk_forward(df)

            # filter unstable strategies
            if sh < 0.2:
                continue

            alphas.append(a)
            vols.append(v)
            syms.append(s)
            prices.append(df["close"].iloc[-1])

        if not alphas:
            time.sleep(CHECK_INTERVAL)
            continue

        acc = account()
        pos = positions()

        eq = float(acc.equity)

        exp = exposure(pos, {s:p for s,p in zip(syms, prices)}, eq)

        regime = market_regime(vols)

        with lock:
            equity_curve.append(eq)

        peak = max(peak, eq)
        dd = (peak - eq) / (peak + EPS)

        if dd > MAX_DRAWDOWN:
            print("🚨 HALT")
            time.sleep(CHECK_INTERVAL)
            continue

        w = portfolio(alphas, vols, regime)

        ranked = sorted(zip(syms, w), key=lambda x: abs(x[1]), reverse=True)[:TOP_K]

        print(f"📊 EXP={exp:.2f} REGIME={regime:.2f} DD={dd:.2f}")

        for s, weight in ranked:

            price = prices[syms.index(s)]

            target = weight * eq
            current = pos.get(s, 0.0)

            desired = target / (price + EPS)
            delta = desired - current

            if abs(delta) < 0.5:
                continue

            qty = max(1, int(abs(delta)))
            side = "buy" if delta > 0 else "sell"

            try:
                api.submit_order(
                    symbol=s,
                    qty=qty,
                    side=side,
                    type="market",
                    time_in_force="gtc"
                )

                with lock:
                    order_log.append({
                        "symbol": s,
                        "qty": qty,
                        "side": side,
                        "time": time.time()
                    })

            except Exception as e:
                print("order fail", s, e)

        print(f"💼 EQUITY={eq:.2f}")

        time.sleep(CHECK_INTERVAL)

# =========================================================
# API
# =========================================================
@app.route("/")
def home():
    return {"status": "v11", "symbols": len(SYMBOLS)}


@app.route("/equity")
def eq():
    return {"equity": list(equity_curve)}

def cash_api():
def cs():
    return {"cash": list(cash_curve)}


@app.route("/orders")
def orders():
    return list(order_log)


@app.route("/debug/strategy")
def strat():
    return dict(strategy_stats)

@app.route("/portfolio")
def portfolio():
    return jsonify([
        {
            "symbol": p.symbol,
            "qty": int(p.qty),
            "value": float(p.market_value)
        }
        for p in positions()
    ])
    
@app.route("/ml")
def ml_state():
    return {
        "sample_weights": dict(list(ml_weights.items())[:30])
    }

@app.route("/status")
def status():
    acc = api.get_account()
    return jsonify({
        "equity": float(acc.equity),
        "cash": float(acc.cash),
        "regime": market_regime(),
        "exposure": exposure()
    })

# =========================================================
# START
# =========================================================
def start():
    threading.Thread(target=engine, daemon=True).start()


if __name__ == "__main__":
    print("🏛️ STARTING v11 WALK-FORWARD CORE")
    start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
