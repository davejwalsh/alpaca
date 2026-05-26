import os
import time
import threading
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from collections import deque
from flask import Flask

import joblib
import io
from supabase import create_client

from datetime import datetime
import base64

# =========================================================
# CONFIG
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

LOOKAHEAD = 12
FEE = 0.001
SLIPPAGE = 0.001

MAX_POSITIONS = 5
MAX_EXPOSURE = 0.80

THRESHOLD = 0.52

RETRAIN_EVERY = 500
SAVE_WEIGHTS_EVERY = 500
CHECK_INTERVAL = 30

BACKTEST_TRAIN_END = 400

BAR_CACHE = {}
BAR_CACHE_TS = {}
BAR_CACHE_TTL = 60

SIGNAL_CACHE = {}
SIGNAL_CACHE_TS = {}
SIGNAL_CACHE_TTL = 60  # seconds (tune this)

clock_lock = threading.Lock()
_last_clock_check = 0
_cached_open = False

# =========================================================
# API (PAPER ACCOUNT)
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

portfolio_state = {
    "positions": {},
    "last_probs": {},
    "last_prices": {},
    "equity": 0.0,
    "cash": 0.0
}

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

BUCKET = "weights"

# =========================================================
# WEIGHTS SAVE/LOAD
# =========================================================
def save_weights_to_supabase():
    global model, scaler

    with model_lock:
        if model is None or scaler is None:
            print("⚠️ No model to save")
            return

        buffer = io.BytesIO()
        joblib.dump({"model": model, "scaler": scaler}, buffer)
        buffer.seek(0)

        data = buffer.read()   # ✅ FORCE BYTES CLEANLY

    try:
        res = supabase.storage.from_(BUCKET).upload(
            "model_bundle.pkl",
            data,
            file_options={
                "content-type": "application/octet-stream",
                "upsert": "true"
            }
        )
        print("✅ Weights saved to Supabase")

    except Exception as e:
        print("❌ Supabase save error:", e)


def load_weights_from_supabase():
    global model, scaler, is_trained
    print("🧠 Attempting to load weights")

    try:
        files = supabase.storage.from_(BUCKET).list()
        exists = any(f["name"] == "model_bundle.pkl" for f in files)
        if not exists:
            print("🧠 No weights found")
            return

        res = supabase.storage.from_(BUCKET).download("model_bundle.pkl")
        if not res:
            print("⚠️ No saved weights found")
            return

        data = res.read() if hasattr(res, "read") else res
        obj = joblib.load(io.BytesIO(data))

        with model_lock:
            model = obj["model"]
            scaler = obj["scaler"]
            is_trained = True

        print("✅ Weights loaded from Supabase")

    except Exception as e:
        print("⚠️ Could not load weights:", e)
        model = None
        scaler = None
        is_trained = False

# =========================================================
# MARKET CHECK
# =========================================================
def market_is_open():
    global _last_clock_check, _cached_open

    with clock_lock:
        now = time.time()
        if now - _last_clock_check < 60:
            return _cached_open

        try:
            clock = api.get_clock()
            _cached_open = clock.is_open
            _last_clock_check = now
            return _cached_open
        except:
            return False

# =========================================================
# SAFE DATA
# =========================================================
def get_bars(symbol, limit=1000):
    now = time.time()

    if len(BAR_CACHE) > 500:
        BAR_CACHE.clear()
        BAR_CACHE_TS.clear()

    if symbol in BAR_CACHE and now - BAR_CACHE_TS[symbol] < BAR_CACHE_TTL:
        return BAR_CACHE[symbol]

    try:
        bars = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit)
        df = bars.df.sort_index()

        BAR_CACHE[symbol] = df
        BAR_CACHE_TS[symbol] = now

        return df

    except Exception as e:
        print(f"[get_bars] error for {symbol}: {e}")
        return pd.DataFrame()

# =========================================================
# FEATURE ENGINE
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
    trend_strength = (ma10 - ma30) / (ma30 + 1e-9)

    return np.array([
        r5.iloc[-1],
        r10.iloc[-1],
        r20.iloc[-1],
        vol.iloc[-1],
        drawdown.iloc[-1],
        recovery.iloc[-1],
        trend_strength.iloc[-1]
    ])

# =========================================================
# DATASET BUILDER
# =========================================================
def build_dataset(df, min_window=60, max_index=None):
    if max_index is None:
        max_index = len(df) - LOOKAHEAD

    X, y = [], []
    STEP = 5
    for i in range(min_window, max_index, STEP):
        if i + LOOKAHEAD >= len(df):
            break

        window = df.iloc[:i]
        x = features(window)

        if np.isnan(x).any():
            continue

        current = df["close"].iloc[i]
        future = df["close"].iloc[i + LOOKAHEAD]
        ret = future / current - 1

        volatility = df["close"].pct_change().rolling(20).std().iloc[i]
        
        threshold = max(0.0005, volatility * 0.5)
        
        label = 1 if ret > threshold else 0

        X.append(x)
        y.append(label)

    if len(X) == 0:
        return np.empty((0, 7)), np.empty((0,))

    return np.array(X), np.array(y)

# =========================================================
# TRAINER
# =========================================================
def train_on_data(data_dict, end_idx=None):
    X_all, y_all = [], []

    for s, df in data_dict.items():
        max_index = min(end_idx, len(df) - LOOKAHEAD) if end_idx else None
        X, y = build_dataset(df, max_index=max_index)
        if len(X) == 0:
            continue

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        return None, None

    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)

    if len(np.unique(y_all)) < 2:
        print("⚠️ Only one class in training data. Skipping training.")
        return None, None

    bt_scaler = StandardScaler()
    bt_scaler.fit(X_all)
    X_all_scaled = bt_scaler.transform(X_all)

    bt_model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, class_weight="balanced")
    bt_model.fit(X_all_scaled, y_all)

    return bt_model, bt_scaler

# =========================================================
# LIVE TRAIN
# =========================================================
def train():
    global model, scaler, is_trained

    print("🧠 TRAINING STARTED")

    data = {}
    for s in SYMBOLS:
        df = get_bars(s, 800)
        if not df.empty:
            data[s] = df

    if not data:
        print("⚠️ No data to train on.")
        return

    new_model, new_scaler = train_on_data(data)

    if new_model is None:
        print("⚠️ Training failed")
        return

    with model_lock:
        model = new_model
        scaler = new_scaler
        is_trained = True

    print("✅ TRAINING COMPLETE")

# =========================================================
# PREDICT
# =========================================================
def predict(df):
    with model_lock:
        if model is None or scaler is None:
            return 0.0

        x = features(df).reshape(1, -1)
        if np.isnan(x).any():
            return 0.0

        x_scaled = scaler.transform(x)
        return float(model.predict_proba(x_scaled)[0, 1])

# =========================================================
# RANK
# =========================================================
def rank_market():
    scores = []
    cached = {}

    now = time.time()

    for s in SYMBOLS:
        df = get_bars(s)
        if df.empty or len(df) < 100:
            continue

        cached[s] = df

    for s, df in cached.items():

        # =====================================================
        # CACHE CHECK (NEW LOGIC)
        # =====================================================
        if s in SIGNAL_CACHE:
            if now - SIGNAL_CACHE_TS.get(s, 0) < SIGNAL_CACHE_TTL:
                prob = SIGNAL_CACHE[s]
                scores.append((s, prob, df))
                continue

        # =====================================================
        # COMPUTE SIGNAL (ONLY IF NEEDED)
        # =====================================================
        prob = predict(df)

        SIGNAL_CACHE[s] = prob
        SIGNAL_CACHE_TS[s] = now

        scores.append((s, prob, df))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:MAX_POSITIONS]

# =========================================================
# EXECUTION
# =========================================================
def execute_portfolio(ranked):
    try:
        account = api.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
    except Exception as e:
        print(f"[execute_portfolio] account error: {e}")
        return

    portfolio_state["equity"] = equity
    portfolio_state["cash"] = cash

    max_alloc = equity * MAX_EXPOSURE
    allocation_per = max_alloc / MAX_POSITIONS

    positions_snapshot = {}
    probs_snapshot = {}
    prices_snapshot = {}

    for symbol, prob, df in ranked:

        try:
            trade = api.get_latest_trade(symbol)
            live_price = float(trade.price)
        except Exception as e:
            print(f"[price error] {symbol}: {e}")
            continue

        model_price = float(df["close"].iloc[-1])
        drift = abs(live_price - model_price) / model_price

        if drift > 0.003:
            print(f"⚠️ SKIP {symbol} price moved too fast ({drift:.3f})")
            continue

        probs_snapshot[symbol] = prob
        prices_snapshot[symbol] = live_price

        try:
            pos = api.get_position(symbol)
            qty = int(pos.qty)
        except Exception:
            qty = 0

        positions_snapshot[symbol] = qty

        if live_price <= 0:
            continue

        target_qty = int(allocation_per / live_price)
        if target_qty < 1:
            continue

        if qty == 0 and prob >= THRESHOLD:
            limit_price = live_price * 1.001
            print(f"🟢 BUY {symbol} prob={prob:.2f} @ {limit_price:.2f}")

            try:
                api.submit_order(
                    symbol=symbol,
                    qty=target_qty,
                    side="buy",
                    type="limit",
                    time_in_force="gtc",
                    limit_price=round(limit_price, 2)
                )
            except Exception as e:
                print(f"[BUY ERROR] {symbol}: {e}")

        elif qty > 0 and prob < 0.5:
            limit_price = live_price * 0.999
            print(f"🔴 SELL {symbol} prob={prob:.2f} @ {limit_price:.2f}")

            try:
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="limit",
                    time_in_force="gtc",
                    limit_price=round(limit_price, 2)
                )
            except Exception as e:
                print(f"[SELL ERROR] {symbol}: {e}")

    portfolio_state.update({
        "positions": positions_snapshot,
        "last_probs": probs_snapshot,
        "last_prices": prices_snapshot,
        "equity": equity,
        "cash": cash
    })

    print(f"💼 Equity: {equity:.2f} | Cash: {cash:.2f}")

# =========================================================
# ENGINE
# =========================================================
# =========================================================
# ENGINE
# =========================================================
def engine():

    global step_counter
    global model
    global scaler

    print("🔥 ENGINE STARTED")

    while True:

        try:

            print("🔁 ENGINE LOOP")

            if not market_is_open():
                print("🌙 Market closed — sleeping")
                time.sleep(60)
                continue

            if not is_trained:
                print("⚠️ Model not trained yet")
                time.sleep(5)
                continue

            ranked = rank_market()

            print(f"📊 Ranked symbols: {len(ranked)}")

            for s, p, _ in ranked:
                print(f"   {s}: {p:.3f}")

            execute_portfolio(ranked)

            equity_curve.append(portfolio_state["equity"])

            step_counter += 1

            # ============================================
            # RETRAIN
            # ============================================
            if step_counter % RETRAIN_EVERY == 0:

                print("🧠 Retraining model...")

                train()

            # ============================================
            # SAVE WEIGHTS
            # ============================================
            if step_counter % SAVE_WEIGHTS_EVERY == 0:

                print("💾 Saving weights...")

                save_weights_to_supabase()

            time.sleep(CHECK_INTERVAL)

        except Exception as e:

            print(f"❌ ENGINE ERROR: {e}")

            import traceback
            traceback.print_exc()

            time.sleep(10)
# =========================================================
# BACKTEST
# =========================================================
def backtest():
    print("▶️ Starting backtest...")

    data = {}
    min_len = None

    for s in SYMBOLS:
        df = get_bars(s, 1000)
        if df.empty or len(df) <= BACKTEST_TRAIN_END + LOOKAHEAD:
            continue
        data[s] = df
        min_len = len(df) if min_len is None else min(min_len, len(df))

    if not data:
        return []

    bt_model, bt_scaler = train_on_data(data, end_idx=BACKTEST_TRAIN_END)
    if bt_model is None:
        return []

    cash = 100000.0
    positions = {}
    equity_curve.clear()

    cost_factor = 1 + FEE + SLIPPAGE
    sell_factor = 1 - FEE - SLIPPAGE

    for i in range(BACKTEST_TRAIN_END, min_len - LOOKAHEAD):

        scores = []

        for s, df in data.items():
            window = df.iloc[:i]
            x = features(window)
            if np.isnan(x).any():
                continue

            x_scaled = bt_scaler.transform(x.reshape(1, -1))
            prob = bt_model.predict_proba(x_scaled)[0, 1]
            price = df["close"].iloc[i]

            scores.append((s, prob, price))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:MAX_POSITIONS]

        top_symbols = {t[0] for t in top}

        for s in list(positions.keys()):
            if s not in top_symbols:
                price = data[s]["close"].iloc[i]
                cash += positions[s] * price * sell_factor
                del positions[s]

        alloc = cash * MAX_EXPOSURE / MAX_POSITIONS

        for s, prob, price in top:
            if prob < THRESHOLD:
                continue

            raw_qty = alloc / price
            if raw_qty < 1:
                continue

            qty = int(raw_qty)
            cost = qty * price * cost_factor

            if cash >= cost:
                cash -= cost
                positions[s] = qty

        equity = cash + sum(
            positions[s] * data[s]["close"].iloc[i]
            for s in positions
        )

        equity_curve.append(equity)

    return list(equity_curve)

# =========================================================
# API ROUTES
# =========================================================
@app.route("/status")
def status():
    return {"trained": is_trained, "steps": step_counter}

@app.route("/portfolio")
def portfolio():
    return portfolio_state

@app.route("/equity")
def equity():
    return {"equity_curve": list(equity_curve)}

@app.route("/backtest")
def run_backtest():
    return {"equity_curve": backtest()}

@app.route("/weights/base64", methods=["GET"])
def get_weights_base64():
    try:
        res = supabase.storage.from_(BUCKET).download("model_bundle.pkl")
        if not res:
            return {"error": "no weights found"}, 404

        data = res.read() if hasattr(res, "read") else res
        encoded = base64.b64encode(data).decode("utf-8")

        return {"filename": "model_bundle.pkl", "content_base64": encoded}

    except Exception as e:
        return {"error": str(e)}, 500

@app.route("/signals")
def signals():

    if not is_trained:
        return {"error": "model not trained"}

    ranked = rank_market()

    output = []

    for symbol, prob, df in ranked:

        try:
            latest_price = float(df["close"].iloc[-1])
        except:
            latest_price = None

        output.append({
            "symbol": symbol,
            "probability": round(prob, 4),
            "price": latest_price
        })

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "threshold": THRESHOLD,
        "signals": output
    }
    
# =========================================================
# START
# =========================================================
def run_engine():
    engine()


def run_flask():
    print("🌐 Flask API starting on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)


def start():
    print("🚀 SYSTEM BOOTING...")

    time.sleep(3)

    # Load weights first
    load_weights_from_supabase()

    if model is None or scaler is None:
        print("🧠 No weights found → training fresh model")
        train()
        save_weights_to_supabase()

    # Start engine thread (TRADING LOOP)
    engine_thread = threading.Thread(target=run_engine, daemon=True)
    engine_thread.start()
    print("⚙️ Trading engine thread started")

    # Start Flask in main thread
    run_flask()


if __name__ == "__main__":
    start()
