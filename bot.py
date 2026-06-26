import os
import time
import threading
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from collections import deque
from flask import Flask

# =========================================================
# CONFIG
# =========================================================

SYMBOLS = [
    "AAPL","TSLA","NVDA","AMD","SPY","MSFT","AMZN","META","GOOGL",
    "JPM","XOM","KO","PEP","WMT","COST","CRM","UBER","PLTR","COIN"
]

LOOKAHEAD = 12

MAX_POSITIONS = 5
MAX_EXPOSURE = 0.60

BUY_THRESHOLD = 0.64
SELL_THRESHOLD = 0.46
HYSTERESIS = 0.08

MIN_HOLD_TIME = 20 * 60
TRADE_COOLDOWN = 10 * 60

STOP_LOSS = -0.025
TAKE_PROFIT = 0.035

RETRAIN_EVERY = 300
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

model = None
scaler = None
model_lock = threading.Lock()
is_trained = False

step_counter = 0

portfolio_state = {
    "bought_at": {},
    "entry_prices": {},
    "last_trade": {}
}

# =========================================================
# DATA
# =========================================================

def get_bars(symbol, limit=500):
    try:
        bars = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit)
        return bars.df.sort_index()
    except:
        return pd.DataFrame()

# =========================================================
# REGIME FILTER (ANTI-CHOP / ANTI-CHURN)
# =========================================================

def market_regime_ok(df):
    try:
        vol = df["close"].pct_change().rolling(30).std().iloc[-1]
        return vol < 0.03
    except:
        return False

# =========================================================
# FEATURES
# =========================================================

def features(df):
    c = df["close"]

    r5 = c.pct_change(5)
    r10 = c.pct_change(10)
    vol = c.pct_change().rolling(20).std()

    ma10 = c.rolling(10).mean()
    ma30 = c.rolling(30).mean()

    trend = (ma10 - ma30) / (ma30 + 1e-9)

    x = np.array([
        r5.iloc[-1],
        r10.iloc[-1],
        vol.iloc[-1],
        trend.iloc[-1]
    ])

    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return np.zeros(4)

    return x

# =========================================================
# DATASET
# =========================================================

def build_dataset(df):
    X, y = [], []

    for i in range(60, len(df) - LOOKAHEAD, 5):
        x = features(df.iloc[:i])
        if np.any(np.isnan(x)):
            continue

        current = df["close"].iloc[i]
        future = df["close"].iloc[i + LOOKAHEAD]

        ret = (future / current) - 1
        vol = df["close"].pct_change().rolling(20).std().iloc[i]

        threshold = max(0.001, vol * 0.7)

        X.append(x)
        y.append(1 if ret > threshold else 0)

    return np.array(X), np.array(y)

# =========================================================
# TRAIN
# =========================================================

def train():
    global model, scaler, is_trained

    print("🧠 Training model...")

    X_all, y_all = [], []

    for s in SYMBOLS:
        df = get_bars(s, 800)
        if df.empty:
            continue

        X, y = build_dataset(df)
        if len(X) == 0:
            continue

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        print("⚠️ No data")
        return

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    if len(np.unique(y_all)) < 2:
        print("⚠️ Bad dataset")
        return

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    base = SGDClassifier(loss="log_loss", class_weight="balanced")
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_all, y_all)

    is_trained = True
    print("✅ Training complete")

# =========================================================
# PREDICT (SAFE)
# =========================================================

def predict(df):
    with model_lock:
        if model is None or scaler is None:
            return 0.0

        x = features(df).reshape(1, -1)
        if np.any(np.isnan(x)):
            return 0.0

        return float(model.predict_proba(scaler.transform(x))[0, 1])

# =========================================================
# POSITION HELPERS
# =========================================================

def get_positions():
    try:
        return {p.symbol: p for p in api.list_positions()}
    except:
        return {}

def get_position_count(positions):
    return len([p for p in positions.values() if int(p.qty) > 0])

def current_exposure(positions):
    total = 0.0
    for p in positions.values():
        try:
            total += int(p.qty) * float(p.avg_entry_price)
        except:
            pass
    return total

# =========================================================
# SYNC STATE (FIXED)
# =========================================================

def sync_state():
    try:
        positions = api.list_positions()
        now = time.time()

        for p in positions:
            if p.symbol not in portfolio_state["bought_at"]:
                portfolio_state["bought_at"][p.symbol] = now
                portfolio_state["entry_prices"][p.symbol] = float(p.avg_entry_price)
    except:
        pass

# =========================================================
# SAFE ORDER WRAPPER
# =========================================================

def safe_order(symbol, qty, side):
    try:
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day"
        )
        return True
    except Exception as e:
        print(f"ORDER ERROR {symbol}: {e}")
        return False

# =========================================================
# EXECUTION CORE (ANTI-CHURN + REGIME FILTER)
# =========================================================

def execute_portfolio(ranked):
    now = time.time()

    try:
        account = api.get_account()
        equity = float(account.equity)
    except:
        return

    positions = get_positions()

    for symbol, prob, df in ranked[:MAX_POSITIONS]:

        if not market_regime_ok(df):
            continue

        price = float(df["close"].iloc[-1])
        pos = positions.get(symbol)

        qty = int(pos.qty) if pos else 0
        entry = float(pos.avg_entry_price) if pos else None

        last_trade = portfolio_state["last_trade"].get(symbol, 0)

        if now - last_trade < TRADE_COOLDOWN:
            continue

        # =====================================================
        # HOLD LOGIC
        # =====================================================

        if qty > 0:
            hold_time = now - portfolio_state["bought_at"].get(symbol, now)
            pnl = (price - entry) / entry if entry else 0

            if pnl <= STOP_LOSS:
                if safe_order(symbol, qty, "sell"):
                    portfolio_state["last_trade"][symbol] = now
                continue

            if pnl >= TAKE_PROFIT:
                if safe_order(symbol, qty, "sell"):
                    portfolio_state["last_trade"][symbol] = now
                continue

            if hold_time < MIN_HOLD_TIME:
                continue

            if prob < (SELL_THRESHOLD - HYSTERESIS):
                if safe_order(symbol, qty, "sell"):
                    portfolio_state["last_trade"][symbol] = now
                    portfolio_state["bought_at"].pop(symbol, None)
                    portfolio_state["entry_prices"].pop(symbol, None)

            continue

        # =====================================================
        # BUY LOGIC
        # =====================================================

        if prob < BUY_THRESHOLD:
            continue

        if get_position_count(positions) >= MAX_POSITIONS:
            continue

        if current_exposure(positions) >= equity * MAX_EXPOSURE:
            continue

        allocation = (equity * MAX_EXPOSURE) / MAX_POSITIONS
        qty = int(allocation / price)

        if qty < 1:
            continue

        if safe_order(symbol, qty, "buy"):
            portfolio_state["bought_at"][symbol] = now
            portfolio_state["entry_prices"][symbol] = price
            portfolio_state["last_trade"][symbol] = now

# =========================================================
# ENGINE
# =========================================================

def engine():
    global step_counter

    print("🔥 Engine started")

    while True:
        try:
            if not is_trained:
                time.sleep(5)
                continue

            ranked = []

            for s in SYMBOLS:
                df = get_bars(s)
                if df.empty or len(df) < 120:
                    continue

                ranked.append((s, predict(df), df))

            ranked.sort(key=lambda x: x[1], reverse=True)

            execute_portfolio(ranked)

            step_counter += 1

            if step_counter % RETRAIN_EVERY == 0:
                train()

            time.sleep(CHECK_INTERVAL)

        except Exception as e:
            print("ENGINE ERROR:", e)
            time.sleep(10)

# =========================================================
# START
# =========================================================

def start():
    print("🚀 Booting system...")

    train()
    sync_state()

    t = threading.Thread(target=engine, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=8080, debug=False)


if __name__ == "__main__":
    start()
