import os
import time
import threading
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from collections import deque
from flask import Flask

# =========================================================
# CONFIG
# =========================================================
SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "META", "SPY"]

LOOKAHEAD = 12
FEE = 0.001
SLIPPAGE = 0.001

MAX_POSITIONS = 5
MAX_EXPOSURE = 0.80

THRESHOLD = 0.60
EXIT_THRESHOLD = 0.45

RETRAIN_EVERY = 200
CHECK_INTERVAL = 30

MIN_TRAIN_SAMPLES = 200

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
equity_curve = deque(maxlen=5000)

model = None
scaler = None
model_lock = threading.Lock()

is_trained = False
step_counter = 0

# =========================================================
# DATA
# =========================================================
def get_bars(symbol, limit=1000):
    try:
        bars = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit)
        df = bars.df.sort_index()
        return df
    except Exception as e:
        print(f"[data] {symbol} error")
        return pd.DataFrame()

# =========================================================
# FEATURES
# =========================================================
def features(df):
    close = df["close"]

    r5 = close.pct_change(5)
    r10 = close.pct_change(10)
    r20 = close.pct_change(20)

    vol = close.pct_change().rolling(20).std()

    ma10 = close.rolling(10).mean()
    ma30 = close.rolling(30).mean()

    drawdown = (close - close.rolling(30).max()) / close.rolling(30).max()
    recovery = close.pct_change(3)
    trend = (ma10 - ma30) / ma30

    return np.array([
        r5.iloc[-1],
        r10.iloc[-1],
        r20.iloc[-1],
        vol.iloc[-1],
        drawdown.iloc[-1],
        recovery.iloc[-1],
        trend.iloc[-1]
    ])

# =========================================================
# FILTER (IMPORTANT)
# =========================================================
def passes_filter(df):
    close = df["close"]

    vol = close.pct_change().rolling(20).std().iloc[-1]
    drawdown = (close.iloc[-1] - close.rolling(30).max().iloc[-1]) / close.rolling(30).max().iloc[-1]

    if np.isnan(vol) or np.isnan(drawdown):
        return False

    return vol > 0.002 and drawdown < -0.01  # bounce setup

# =========================================================
# DATASET
# =========================================================
def build_dataset(df, min_window=60):
    X, y = [], []

    for i in range(min_window, len(df) - LOOKAHEAD):
        window = df.iloc[:i]
        x = features(window)

        if np.isnan(x).any():
            continue

        current = df["close"].iloc[i]
        future = df["close"].iloc[i + LOOKAHEAD]
        ret = future / current - 1

        # 3-class labeling
        if ret > 0.003:
            label = 1
        elif ret < -0.003:
            label = -1
        else:
            continue  # remove noise

        X.append(x)
        y.append(label)

    if not X:
        return None, None

    X = np.array(X)
    y = np.array(y)

    # convert to binary
    y = (y == 1).astype(int)

    return X, y

# =========================================================
# TRAIN
# =========================================================
def train():
    global model, scaler, is_trained

    X_all, y_all = [], []

    for s in SYMBOLS:
        df = get_bars(s, 800)
        if df.empty:
            continue

        X, y = build_dataset(df)
        if X is None:
            continue

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        print("⚠️ no data")
        return

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    if len(y_all) < MIN_TRAIN_SAMPLES or len(np.unique(y_all)) < 2:
        print("⚠️ bad training data")
        return

    scaler_new = StandardScaler()
    X_scaled = scaler_new.fit_transform(X_all)

    classes = np.unique(y_all)
    weights = compute_class_weight("balanced", classes=classes, y=y_all)

    model_new = SGDClassifier(loss="log_loss", max_iter=1000)
    model_new.fit(X_scaled, y_all)

    with model_lock:
        model = model_new
        scaler = scaler_new
        is_trained = True

    print("✅ trained")

# =========================================================
# PREDICT
# =========================================================
def predict(df):
    with model_lock:
        if model is None:
            return 0.0

        x = features(df).reshape(1, -1)

        if np.isnan(x).any():
            return 0.0

        x = scaler.transform(x)
        return model.predict_proba(x)[0, 1]

# =========================================================
# RANK
# =========================================================
def rank_market():
    scores = []

    for s in SYMBOLS:
        df = get_bars(s)

        if df.empty or len(df) < 100:
            continue

        if not passes_filter(df):
            continue

        prob = predict(df)
        scores.append((s, prob, df))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:MAX_POSITIONS]

# =========================================================
# EXECUTION
# =========================================================
def execute_portfolio(ranked):
    try:
        equity = float(api.get_account().equity)
    except:
        return

    alloc = (equity * MAX_EXPOSURE) / MAX_POSITIONS

    for symbol, prob, df in ranked:

        price = df["close"].iloc[-1]

        try:
            pos = api.get_position(symbol)
            qty = int(pos.qty)
        except:
            qty = 0

        target = int(alloc / price)

        if qty == 0 and prob > THRESHOLD:
            print(f"BUY {symbol} {prob:.2f}")
            api.submit_order(symbol=symbol, qty=target, side="buy", type="market", time_in_force="gtc")

        elif qty > 0 and prob < EXIT_THRESHOLD:
            print(f"SELL {symbol} {prob:.2f}")
            api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="gtc")

# =========================================================
# ENGINE
# =========================================================
def engine():
    global step_counter

    while True:
        if not is_trained:
            time.sleep(5)
            continue

        ranked = rank_market()
        execute_portfolio(ranked)

        step_counter += 1

        if step_counter % RETRAIN_EVERY == 0:
            train()

        time.sleep(CHECK_INTERVAL)

# =========================================================
# BACKTEST (SIMPLIFIED SAFE)
# =========================================================
def backtest():
    print("backtest disabled in v6 (safe mode)")
    return []

# =========================================================
# API
# =========================================================
@app.route("/status")
def status():
    return {"trained": is_trained, "steps": step_counter}

# =========================================================
# START
# =========================================================
def start():
    time.sleep(3)
    train()
    threading.Thread(target=engine, daemon=True).start()

if __name__ == "__main__":
    print("🚀 V6 STABLE")
    threading.Thread(target=start, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
