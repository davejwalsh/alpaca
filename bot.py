import os
import time
import threading
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

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

MAX_POSITIONS = 7
MAX_EXPOSURE = 0.50

THRESHOLD = 0.55

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

LAST_TRADE_TS = {}
TRADE_COOLDOWN = 300
MIN_HOLD_TIME = 3600

STALE_TIME = 60 * 60 * 12
STALE_THRESHOLD = 0.01
TAKE_PROFIT = 0.03

STOP_LOSS = -0.03

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
    "entry_prices": {},
    "bought_at": {},
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


def reload_portfolio_state():

    print("🔄 Reloading portfolio state...")

    try:

        positions = api.list_positions()

    except Exception as e:
        print(f"[reload_portfolio_state] {e}")
        return

    now = time.time()

    portfolio_state["positions"].clear()
    portfolio_state["entry_prices"].clear()

    # We keep bought_at, but we will selectively prune it
    current_symbols = set()
    
    for pos in positions:

        try:

            symbol = pos.symbol

            qty = int(pos.qty)

            avg_entry = float(pos.avg_entry_price)
            current_symbols.add(symbol)
            portfolio_state["positions"][symbol] = qty

            # restore entry price
            portfolio_state["entry_prices"][symbol] = avg_entry

            # restore bought time if missing
            if symbol not in portfolio_state["bought_at"]:

                # assume recently loaded
                portfolio_state["bought_at"][symbol] = now

            print(
                f"✅ Restored {symbol} "
                f"qty={qty} entry={avg_entry}"
            )

        except Exception as e:

            print(f"[reload symbol error] {e}")

    for symbol in list(portfolio_state["bought_at"].keys()):
        if symbol not in current_symbols:
            portfolio_state["bought_at"].pop(symbol, None)

    print(f"🔄 Sync complete. Positions: {len(current_symbols)}")
# =========================================================
# MARKET CHECK
# # =========================================================
# def market_is_open():
#     global _last_clock_check, _cached_open
#     print("Checking if open")
#     with clock_lock:
#         print("in with")
#         now = time.time()
#         if now - _last_clock_check < 60:
#             return _cached_open
#         print("Continue 1")
#         try:
#             print("Continue 1.0")
#             clock = api.get_clock()
#             print("Continue 1.1")
#             _cached_open = clock.is_open
#             print("Continue 1.2")
#             _last_clock_check = now
#             print(f"Continue 2 {_cached_open}")
#             return _cached_open
#         except:
#             print("Continue 3")
#             return False

def market_is_open():
    return True
    # global _last_clock_check, _cached_open

    # print("Checking if open")

    # now = time.time()

    # # fast path (no lock needed if stale check fails)
    # if now - _last_clock_check < 60:
    #     return _cached_open

    # try:
    #     print("Calling Alpaca clock API")
    #     clock = api.get_clock()
    #     result = clock.is_open
    # except Exception as e:
    #     print("Clock API failed:", e)
    #     return False

    # # ONLY lock shared state update
    # with clock_lock:
    #     _cached_open = result
    #     _last_clock_check = now
    # print("returning")
    # return result

# =========================================================
# SAFE DATA
# =========================================================
# def get_bars(symbol, limit=1000):
#     now = time.time()

#     if len(BAR_CACHE) > 500:
#         BAR_CACHE.clear()
#         BAR_CACHE_TS.clear()

#     if symbol in BAR_CACHE and now - BAR_CACHE_TS[symbol] < BAR_CACHE_TTL:
#         return BAR_CACHE[symbol]

#     try:
#         bars = api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=limit)
#         df = bars.df.sort_index()

#         BAR_CACHE[symbol] = df
#         BAR_CACHE_TS[symbol] = now

#         return df

#     except Exception as e:
#         print(f"[get_bars] error for {symbol}: {e}")
#         return pd.DataFrame()

def get_bars(symbol, limit=1000):
    now = time.time()

    if symbol in BAR_CACHE:
        age = now - BAR_CACHE_TS.get(symbol, 0)

        if age < BAR_CACHE_TTL:
            print(f"⚡ BAR CACHE {symbol}", flush=True)
            return BAR_CACHE[symbol]

    try:
        print(f"🌐 Alpaca request {symbol}", flush=True)

        start = time.time()

        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Minute,
            limit=limit
        )

        elapsed = time.time() - start

        print(
            f"✅ Alpaca response {symbol} "
            f"({elapsed:.2f}s)",
            flush=True
        )

        df = bars.df.sort_index()

        BAR_CACHE[symbol] = df
        BAR_CACHE_TS[symbol] = now

        return df

    except Exception as e:

        print(
            f"❌ get_bars failed {symbol}: {e}",
            flush=True
        )

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

        future = df["close"].iloc[i + LOOKAHEAD]
        current = df["close"].iloc[i]
        ret = (future / current) - 1
        
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

    base = SGDClassifier(loss="log_loss", class_weight="balanced")
    bt_model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
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
# def rank_market():
#     scores = []
#     cached = {}

#     now = time.time()

#     for s in SYMBOLS:
#         df = get_bars(s)
#         if df.empty or len(df) < 100:
#             continue

#         cached[s] = df

#     for s, df in cached.items():

#         # =====================================================
#         # CACHE CHECK (NEW LOGIC)
#         # =====================================================
#         if s in SIGNAL_CACHE:
#             if now - SIGNAL_CACHE_TS.get(s, 0) < SIGNAL_CACHE_TTL:
#                 prob = SIGNAL_CACHE[s]
#                 scores.append((s, prob, df))
#                 continue

#         # =====================================================
#         # COMPUTE SIGNAL (ONLY IF NEEDED)
#         # =====================================================
#         prob = predict(df)

#         SIGNAL_CACHE[s] = prob
#         SIGNAL_CACHE_TS[s] = now

#         scores.append((s, prob, df))

#     scores.sort(key=lambda x: x[1], reverse=True)
#     filtered = [
#         x for x in scores
#         if x[1] >= THRESHOLD
#     ]

#     print("SYMBOLS CHECKED:", len(SYMBOLS))
#     print("SCORES BEFORE FILTER:", len(scores))
#     print("THRESHOLD:", THRESHOLD)
    
#     return filtered[:MAX_POSITIONS]

def rank_market():
    print("📊 rank_market START", flush=True)

    start_total = time.time()

    scores = []
    cached = {}

    now = time.time()

    total_symbols = len(SYMBOLS)

    # =====================================================
    # FETCH DATA
    # =====================================================
    for i, s in enumerate(SYMBOLS, start=1):

        symbol_start = time.time()

        try:
            print(
                f"📥 [{i}/{total_symbols}] Fetching bars for {s}",
                flush=True
            )

            df = get_bars(s)

            elapsed = time.time() - symbol_start

            if df.empty:
                print(
                    f"⚠️ [{i}/{total_symbols}] {s} EMPTY "
                    f"({elapsed:.2f}s)",
                    flush=True
                )
                continue

            if len(df) < 100:
                print(
                    f"⚠️ [{i}/{total_symbols}] {s} "
                    f"only {len(df)} bars ({elapsed:.2f}s)",
                    flush=True
                )
                continue

            cached[s] = df

            print(
                f"✅ [{i}/{total_symbols}] {s} "
                f"{len(df)} bars ({elapsed:.2f}s)",
                flush=True
            )

        except Exception as e:

            print(
                f"❌ [{i}/{total_symbols}] {s} fetch failed: {e}",
                flush=True
            )

    print(
        f"📦 Cached symbols: {len(cached)}",
        flush=True
    )

    # =====================================================
    # SCORE SYMBOLS
    # =====================================================
    for s, df in cached.items():

        try:

            # -------------------------------------------------
            # CACHE HIT
            # -------------------------------------------------
            if s in SIGNAL_CACHE:

                cache_age = now - SIGNAL_CACHE_TS.get(s, 0)

                if cache_age < SIGNAL_CACHE_TTL:

                    prob = SIGNAL_CACHE[s]

                    scores.append((s, prob, df))

                    print(
                        f"⚡ CACHE {s} "
                        f"prob={prob:.3f}",
                        flush=True
                    )

                    continue

            # -------------------------------------------------
            # COMPUTE SIGNAL
            # -------------------------------------------------
            pred_start = time.time()

            prob = predict(df)

            pred_elapsed = time.time() - pred_start

            SIGNAL_CACHE[s] = prob
            SIGNAL_CACHE_TS[s] = now

            scores.append((s, prob, df))

            print(
                f"🧠 PREDICT {s} "
                f"prob={prob:.3f} "
                f"({pred_elapsed:.3f}s)",
                flush=True
            )

        except Exception as e:

            print(
                f"❌ Prediction failed for {s}: {e}",
                flush=True
            )

    # =====================================================
    # SORT + FILTER
    # =====================================================
    scores.sort(key=lambda x: x[1], reverse=True)

    filtered = [
        x for x in scores
        if x[1] >= THRESHOLD
    ]

    total_elapsed = time.time() - start_total

    print("======================================", flush=True)
    print(f"SYMBOLS CONFIGURED : {len(SYMBOLS)}", flush=True)
    print(f"SYMBOLS WITH DATA  : {len(cached)}", flush=True)
    print(f"SCORES GENERATED   : {len(scores)}", flush=True)
    print(f"PASSED THRESHOLD   : {len(filtered)}", flush=True)
    print(f"THRESHOLD          : {THRESHOLD}", flush=True)
    print(f"TOTAL TIME         : {total_elapsed:.2f}s", flush=True)
    print("======================================", flush=True)

    return filtered[:MAX_POSITIONS]

def get_open_count():
    try:
        return len(api.list_positions())
    except:
        return 0
        
# def execute_portfolio(ranked):

#     try:
#         account = api.get_account()
#         equity = float(account.equity)
#         cash = float(account.cash)
#     except Exception as e:
#         print(f"[execute_portfolio] account error: {e}")
#         return

#     portfolio_state["equity"] = equity
#     portfolio_state["cash"] = cash

#     usable_equity = equity * MAX_EXPOSURE

#     # =========================================================
#     # SOURCE OF TRUTH: ALPACA POSITIONS
#     # =========================================================
#     try:
#         alpaca_positions = {
#             p.symbol: p for p in api.list_positions()
#         }
#     except Exception as e:
#         print(f"[execute_portfolio] position fetch error: {e}")
#         alpaca_positions = {}

#     # num_open = len(alpaca_positions)
#     allocation_per = usable_equity / MAX_POSITIONS

#     # =========================================================
#     # OBSERVABILITY ONLY (NOT STATE)
#     # =========================================================
#     probs_snapshot = {}
#     prices_snapshot = {}

#     for symbol, prob, df in ranked:

#         try:

#             # =====================================================
#             # LIVE PRICE
#             # =====================================================
#             trade = api.get_latest_trade(symbol)
#             live_price = float(trade.price)

#             probs_snapshot[symbol] = prob
#             prices_snapshot[symbol] = live_price

#             # =====================================================
#             # GET REAL POSITION (ALPACA)
#             # =====================================================
#             pos = alpaca_positions.get(symbol)
#             qty = int(pos.qty) if pos else 0

#             entry_price = float(pos.avg_entry_price) if pos else None

#             # =====================================================
#             # STOP LOSS
#             # =====================================================
#             if qty > 0 and entry_price:

#                 pnl_pct = (live_price - entry_price) / entry_price

#                 if pnl_pct <= STOP_LOSS:

#                     print(f"🛑 STOP LOSS SELL {symbol} pnl={pnl_pct:.3f}")

#                     try:
#                         api.submit_order(
#                             symbol=symbol,
#                             qty=qty,
#                             side="sell",
#                             type="market",
#                             time_in_force="day"
#                         )

#                         portfolio_state["entry_prices"].pop(symbol, None)
#                         portfolio_state["bought_at"].pop(symbol, None)

#                         # num_open -= 1

#                     except Exception as e:
#                         print(f"[STOP SELL ERROR] {symbol}: {e}")

#                     continue

#             # =====================================================
#             # PRICE DRIFT CHECK
#             # =====================================================
#             model_price = float(df["close"].iloc[-1])
#             drift = abs(live_price - model_price) / model_price

#             if drift > 0.005:
#                 print(f"⚠️ SKIP {symbol} drift={drift:.3f}")
#                 continue

#             if live_price <= 0:
#                 continue

#             target_qty = int(allocation_per / live_price)
#             if target_qty < 1:
#                 continue

#             # =====================================================
#             # STALE CHECK
#             # =====================================================
#             # bought_at = portfolio_state["bought_at"].get(symbol)

#             alpaca_positions = {
#                 p.symbol: p for p in api.list_positions()
#             }
            
#             bought_at = None
#             # for p in api.list_positions():
#             if p.symbol == symbol:
#                     # Alpaca does not store entry time → fallback safely
#                 bought_at = portfolio_state["bought_at"].get(symbol)
#                 break

#             if qty > 0 and bought_at and entry_price:

#                 held_time = time.time() - bought_at
#                 pnl_pct = (live_price - entry_price) / entry_price

#                 if (
#                     held_time > STALE_TIME
#                     and -STALE_THRESHOLD < pnl_pct < STALE_THRESHOLD
#                 ):

#                     print(f"🔴 STALE SELL {symbol} pnl={pnl_pct:.3f}")

#                     try:
#                         api.submit_order(
#                             symbol=symbol,
#                             qty=qty,
#                             side="sell",
#                             type="market",
#                             time_in_force="day"
#                         )

#                         portfolio_state["entry_prices"].pop(symbol, None)
#                         portfolio_state["bought_at"].pop(symbol, None)

#                         # num_open -= 1

#                     except Exception as e:
#                         print(f"[STALE SELL ERROR] {symbol}: {e}")

#                     continue

#             # =====================================================
#             # BUY
#             # =====================================================
#             if qty == 0 and prob >= THRESHOLD:

#                 last_trade = LAST_TRADE_TS.get(symbol, 0)

#                 if time.time() - last_trade < TRADE_COOLDOWN:
#                     continue

#                 if get_open_count()  >= MAX_POSITIONS:
#                     continue

#                 if prob < 0.60:
#                     continue

#                 print(f"🟢 BUY {symbol} prob={prob:.3f}")

#                 try:
#                     api.submit_order(
#                         symbol=symbol,
#                         qty=target_qty,
#                         side="buy",
#                         type="market",
#                         time_in_force="day"
#                     )

#                     # NOTE: still imperfect (fill price unknown), but acceptable for now
#                     # portfolio_state["entry_prices"][symbol] = live_price
#                     portfolio_state["bought_at"][symbol] = time.time()

#                     LAST_TRADE_TS[symbol] = time.time()
#                     # num_open += 1

#                 except Exception as e:
#                     print(f"[BUY ERROR] {symbol}: {e}")

#             # =====================================================
#             # SELL
#             # =====================================================
#             elif qty > 0 and (
#                 prob < 0.48
#                 or (
#                     entry_price and
#                     (live_price - entry_price) / entry_price >= TAKE_PROFIT
#                 )
#             ):

#                 print(f"🔴 SELL {symbol} prob={prob:.3f}")

                
#                 try:
#                     api.submit_order(
#                         symbol=symbol,
#                         qty=qty,
#                         side="sell",
#                         type="market",
#                         time_in_force="day"
#                     )

#                     portfolio_state["entry_prices"].pop(symbol, None)
#                     portfolio_state["bought_at"].pop(symbol, None)

#                     LAST_TRADE_TS[symbol] = time.time()
#                     # num_open -= 1

#                 except Exception as e:
#                     print(f"[SELL ERROR] {symbol}: {e}")

#         except Exception as e:
#             print(f"[symbol loop error] {symbol}: {e}")

#     # =========================================================
#     # FINAL STATE UPDATE (OBSERVATION ONLY)
#     # =========================================================
#     portfolio_state.update({
#         "positions": {
#             s: int(p.qty) for s, p in alpaca_positions.items()
#         },
#         "last_probs": probs_snapshot,
#         "last_prices": prices_snapshot,
#         "equity": equity,
#         "cash": cash
#     })

#     print(f"💼 Equity: {equity:.2f} | Cash: {cash:.2f}")

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

    usable_equity = equity * MAX_EXPOSURE

    try:
        alpaca_positions = {p.symbol: p for p in api.list_positions()}
    except Exception as e:
        print(f"[execute_portfolio] position fetch error: {e}")
        alpaca_positions = {}

    now = time.time()

    probs_snapshot = {}
    prices_snapshot = {}

    current_count = len(alpaca_positions)

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    candidates = ranked[:MAX_POSITIONS]

    def position_score(pos, price):
        try:
            entry = float(pos.avg_entry_price)
            return (price - entry) / entry
        except:
            return 0.0

    weakest_positions = sorted(
        alpaca_positions.items(),
        key=lambda x: position_score(
            x[1],
            prices_snapshot.get(x[0], float(x[1].avg_entry_price))
        )
    )
    
    allocation_base = usable_equity / MAX_POSITIONS

    for symbol, prob, df in candidates:

        try:
            trade = api.get_latest_trade(symbol)
            live_price = float(trade.price)

            probs_snapshot[symbol] = prob
            prices_snapshot[symbol] = live_price

            pos = alpaca_positions.get(symbol)
            qty = int(pos.qty) if pos else 0
            entry_price = float(pos.avg_entry_price) if pos else None

            bought_at = portfolio_state["bought_at"].get(symbol)

            # -------------------------------------------------
            # HOLD LOGIC (ONLY AFFECTS SELLING)
            # -------------------------------------------------
            can_sell = True
            if qty > 0 and bought_at:
                held_time = now - bought_at
                if held_time < MIN_HOLD_TIME:
                    can_sell = False

            # -------------------------------------------------
            # EDGE + SIZE
            # -------------------------------------------------
            edge = prob - 0.5
            if edge <= 0:
                continue

            size_factor = min(max(edge * 2.5, 0.0), 1.0)
            target_value = allocation_base * size_factor
            target_qty = int(target_value / live_price)

            if target_qty < 1:
                continue

            # -------------------------------------------------
            # STOP LOSS (always allowed)
            # -------------------------------------------------
            if qty > 0 and entry_price:
                pnl_pct = (live_price - entry_price) / entry_price

                if pnl_pct <= STOP_LOSS:
                    print(f"🛑 STOP LOSS SELL {symbol} pnl={pnl_pct:.3f}")

                    try:
                        api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side="sell",
                            type="market",
                            time_in_force="day"
                        )

                        portfolio_state["entry_prices"].pop(symbol, None)
                        portfolio_state["bought_at"].pop(symbol, None)

                    except Exception as e:
                        print(f"[STOP LOSS ERROR] {symbol}: {e}")

                    continue

            # -------------------------------------------------
            # DRIFT FILTER
            # -------------------------------------------------
            model_price = float(df["close"].iloc[-1])
            drift = abs(live_price - model_price) / model_price

            if drift > 0.005:
                continue

            # =================================================
            # BUY LOGIC
            # =================================================
            if qty == 0:

                last_trade = LAST_TRADE_TS.get(symbol, 0)
                if now - last_trade < TRADE_COOLDOWN:
                    continue

                # replacement logic
                if current_count >= MAX_POSITIONS:

                    weakest_symbol, weakest_pos = weakest_positions[0]

                    weakest_entry = float(weakest_pos.avg_entry_price)
                    weakest_price = prices_snapshot.get(
                        weakest_symbol,
                        weakest_entry
                    )

                    weakest_score = (weakest_price - weakest_entry) / weakest_entry
                    new_score = edge

                    if new_score <= weakest_score:
                        continue

                    print(f"🔁 REPLACING {weakest_symbol} → {symbol}")

                    try:
                        api.submit_order(
                            symbol=weakest_symbol,
                            qty=int(weakest_pos.qty),
                            side="sell",
                            type="market",
                            time_in_force="day"
                        )
                    except Exception as e:
                        print(f"[REPLACE SELL ERROR] {weakest_symbol}: {e}")
                        continue

                print(f"🟢 BUY {symbol} prob={prob:.3f} edge={edge:.3f}")

                try:
                    api.submit_order(
                        symbol=symbol,
                        qty=target_qty,
                        side="buy",
                        type="market",
                        time_in_force="day"
                    )

                    portfolio_state["bought_at"][symbol] = now
                    LAST_TRADE_TS[symbol] = now

                except Exception as e:
                    print(f"[BUY ERROR] {symbol}: {e}")

            # =================================================
            # SELL LOGIC
            # =================================================
            elif qty > 0 and can_sell:

                if not entry_price:
                    continue

                pnl_pct = (live_price - entry_price) / entry_price

                # TAKE PROFIT ONLY (no immediate churn)
                if pnl_pct < TAKE_PROFIT:
                    continue

                print(f"💰 TAKE PROFIT {symbol} pnl={pnl_pct:.3f}")

                try:
                    api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="market",
                        time_in_force="day"
                    )

                    portfolio_state["entry_prices"].pop(symbol, None)
                    portfolio_state["bought_at"].pop(symbol, None)

                    LAST_TRADE_TS[symbol] = now

                except Exception as e:
                    print(f"[SELL ERROR] {symbol}: {e}")

        except Exception as e:
            print(f"[symbol loop error] {symbol}: {e}")

    portfolio_state.update({
        "positions": {s: int(p.qty) for s, p in alpaca_positions.items()},
        "last_probs": probs_snapshot,
        "last_prices": prices_snapshot,
        "equity": equity,
        "cash": cash
    })

    print(f"💼 Equity: {equity:.2f} | Cash: {cash:.2f}")

    # MOST RECENT
# def execute_portfolio(ranked):

#     try:
#         account = api.get_account()
#         equity = float(account.equity)
#         cash = float(account.cash)
#     except Exception as e:
#         print(f"[execute_portfolio] account error: {e}")
#         return

#     portfolio_state["equity"] = equity
#     portfolio_state["cash"] = cash

#     usable_equity = equity * MAX_EXPOSURE

#     # -----------------------------
#     # SINGLE SOURCE OF TRUTH
#     # -----------------------------
#     try:
#         alpaca_positions = {p.symbol: p for p in api.list_positions()}
#     except Exception as e:
#         print(f"[execute_portfolio] position fetch error: {e}")
#         alpaca_positions = {}

#     now = time.time()

#     probs_snapshot = {}
#     prices_snapshot = {}

#     # portfolio helpers
#     current_symbols = set(alpaca_positions.keys())
#     current_count = len(alpaca_positions)

#     # rank incoming signals
#     ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

#     # strongest candidates
#     candidates = ranked[:MAX_POSITIONS]

#     # weakest current holdings (for replacement logic)
#     def position_score(pos, price):
#         try:
#             entry = float(pos.avg_entry_price)
#             return (price - entry) / entry
#         except:
#             return 0.0

#     weakest_positions = sorted(
#         alpaca_positions.items(),
#         key=lambda x: position_score(x[1], prices_snapshot.get(x[0], 0.0))
#     )

#     # allocation model (dynamic)
#     remaining_slots = max(MAX_POSITIONS - current_count, 0)
#     allocation_base = usable_equity / MAX_POSITIONS

#     # -----------------------------
#     # MAIN LOOP
#     # -----------------------------
#     for symbol, prob, df in candidates:

#         try:
#             trade = api.get_latest_trade(symbol)
#             live_price = float(trade.price)

#             probs_snapshot[symbol] = prob
#             prices_snapshot[symbol] = live_price

#             pos = alpaca_positions.get(symbol)
#             qty = int(pos.qty) if pos else 0
#             entry_price = float(pos.avg_entry_price) if pos else None

#             bought_at = portfolio_state["bought_at"].get(symbol)
#             allow_buy = False
#             allow_sell = False
#             if qty > 0 and bought_at:
#                 held_time = now - bought_at
            
#                 if held_time < MIN_HOLD_TIME:
#                     # Allow BUY logic if you want, but block SELL
#                     allow_buy = True
#                     allow_sell = False
#                 else:
#                     allow_buy = False
#                     allow_sell = True
#             else:
#                 allow_buy = True
#                 allow_sell = True

#             # =========================================================
#             # DYNAMIC EDGE (core improvement)
#             # =========================================================
#             edge = prob - 0.5
#             if edge <= 0:
#                 continue

#             size_factor = min(max(edge * 2.5, 0.0), 1.0)

#             target_value = allocation_base * size_factor
#             target_qty = int(target_value / live_price)

#             if target_qty < 1:
#                 continue

#             # =========================================================
#             # STOP LOSS
#             # =========================================================
#             if qty > 0 and entry_price:
#                 pnl_pct = (live_price - entry_price) / entry_price

#                 if pnl_pct <= STOP_LOSS:
#                     print(f"🛑 STOP LOSS SELL {symbol} pnl={pnl_pct:.3f}")

#                     try:
#                         api.submit_order(
#                             symbol=symbol,
#                             qty=qty,
#                             side="sell",
#                             type="market",
#                             time_in_force="day"
#                         )

#                         portfolio_state["entry_prices"].pop(symbol, None)
#                         portfolio_state["bought_at"].pop(symbol, None)

#                     except Exception as e:
#                         print(f"[STOP LOSS ERROR] {symbol}: {e}")

#                     continue

#             # =========================================================
#             # DRIFT FILTER
#             # =========================================================
#             model_price = float(df["close"].iloc[-1])
#             drift = abs(live_price - model_price) / model_price

#             if drift > 0.005:
#                 continue

#             # =========================================================
#             # BUY / SCALE-IN
#             # =========================================================
#             if qty == 0 and allow_buy:

#                 last_trade = LAST_TRADE_TS.get(symbol, 0)
#                 if now - last_trade < TRADE_COOLDOWN:
#                     continue

#                 # REPLACEMENT LOGIC (IMPORTANT)
#                 if current_count >= MAX_POSITIONS:

#                     weakest_symbol, weakest_pos = weakest_positions[0]
#                     weakest_entry = float(weakest_pos.avg_entry_price)
#                     weakest_price = prices_snapshot.get(weakest_symbol, weakest_entry)

#                     weakest_score = (weakest_price - weakest_entry) / weakest_entry
#                     new_score = edge

#                     # replace only if clearly better
#                     if new_score <= weakest_score:
#                         continue

#                     print(f"🔁 REPLACING {weakest_symbol} → {symbol}")

#                     try:
#                         api.submit_order(
#                             symbol=weakest_symbol,
#                             qty=int(weakest_pos.qty),
#                             side="sell",
#                             type="market",
#                             time_in_force="day"
#                         )
#                     except Exception as e:
#                         print(f"[REPLACE SELL ERROR] {weakest_symbol}: {e}")
#                         continue

#                 print(f"🟢 BUY {symbol} prob={prob:.3f} edge={edge:.3f}")

#                 try:
#                     api.submit_order(
#                         symbol=symbol,
#                         qty=target_qty,
#                         side="buy",
#                         type="market",
#                         time_in_force="day"
#                     )

#                     portfolio_state["bought_at"][symbol] = now
#                     LAST_TRADE_TS[symbol] = now

#                 except Exception as e:
#                     print(f"[BUY ERROR] {symbol}: {e}")

#             # =========================================================
#             # SELL LOGIC
#             # =========================================================
#             elif qty > 0 and allow_sell:

#                 # model exit
#                 # if prob < 0.48:
#                 #     print(f"🔴 SELL {symbol} (low prob)")

#                 # take profit
#                 if entry_price and (live_price - entry_price) / entry_price >= TAKE_PROFIT:
#                     print(f"💰 TAKE PROFIT {symbol}")

#                 else:
#                     continue

#                 bought_at = portfolio_state["bought_at"].get(symbol)
#                 if bought_at:
#                     held_time = now - bought_at
                
#                     if held_time < MIN_HOLD_TIME:
#                         print(f"Not held long enough, no sell")
#                         continue
#                 try:
#                     api.submit_order(
#                         symbol=symbol,
#                         qty=qty,
#                         side="sell",
#                         type="market",
#                         time_in_force="day"
#                     )

#                     portfolio_state["entry_prices"].pop(symbol, None)
#                     portfolio_state["bought_at"].pop(symbol, None)

#                     LAST_TRADE_TS[symbol] = now

#                 except Exception as e:
#                     print(f"[SELL ERROR] {symbol}: {e}")

#         except Exception as e:
#             print(f"[symbol loop error] {symbol}: {e}")

#     # -----------------------------
#     # FINAL STATE UPDATE
#     # -----------------------------
#     portfolio_state.update({
#         "positions": {s: int(p.qty) for s, p in alpaca_positions.items()},
#         "last_probs": probs_snapshot,
#         "last_prices": prices_snapshot,
#         "equity": equity,
#         "cash": cash
#     })

#     print(f"💼 Equity: {equity:.2f} | Cash: {cash:.2f}")

# # =========================================================
# # EXECUTION
# # =========================================================
# def execute_portfolio(ranked):

#     try:
#         account = api.get_account()
#         equity = float(account.equity)
#         cash = float(account.cash)

#     except Exception as e:
#         print(f"[execute_portfolio] account error: {e}")
#         return

#     portfolio_state["equity"] = equity
#     portfolio_state["cash"] = cash

#     # =====================================================
#     # ONLY USE PART OF EQUITY
#     # =====================================================
#     usable_equity = equity * MAX_EXPOSURE

#     positions_snapshot = {}
#     probs_snapshot = {}
#     prices_snapshot = {}

#     # =====================================================
#     # COUNT CURRENT OPEN POSITIONS
#     # =====================================================
#     current_positions = api.list_positions()
#     num_open = len(current_positions)

#     remaining_slots = max(1, MAX_POSITIONS - num_open)

#     allocation_per = usable_equity / MAX_POSITIONS

#     for symbol, prob, df in ranked:

#         try:

#             # =============================================
#             # LIVE PRICE
#             # =============================================
#             trade = api.get_latest_trade(symbol)
#             live_price = float(trade.price)

#             probs_snapshot[symbol] = prob
#             prices_snapshot[symbol] = live_price

#             # =============================================
#             # CURRENT POSITION
#             # =============================================
#             try:
#                 pos = api.get_position(symbol)
#                 qty = int(pos.qty)

#             except Exception:
#                 qty = 0

#             positions_snapshot[symbol] = qty

#             # =============================================
#             # ENTRY PRICE
#             # =============================================
#             entry_price = portfolio_state["entry_prices"].get(symbol)
#             pnl_pct = 0.0
#             # =============================================
#             # STOP LOSS
#             # =============================================
#             if qty > 0 and entry_price:

#                 pnl_pct = (live_price - entry_price) / entry_price

#                 if pnl_pct <= STOP_LOSS:

#                     print(f"🛑 STOP LOSS SELL {symbol} pnl={pnl_pct:.3f}")

#                     try:

#                         api.submit_order(
#                             symbol=symbol,
#                             qty=qty,
#                             side="sell",
#                             type="market",
#                             time_in_force="day"
#                         )

#                         portfolio_state["entry_prices"].pop(symbol, None)
#                         portfolio_state["bought_at"].pop(symbol, None)

#                     except Exception as e:
#                         print(f"[STOP SELL ERROR] {symbol}: {e}")

#                     continue

#             # =============================================
#             # PRICE DRIFT CHECK
#             # =============================================
#             model_price = float(df["close"].iloc[-1])

#             drift = abs(live_price - model_price) / model_price

#             if drift > 0.005:
#                 print(f"⚠️ SKIP {symbol} drift={drift:.3f}")
#                 continue

#             # =============================================
#             # POSITION SIZE
#             # =============================================
#             if live_price <= 0:
#                 continue

#             target_qty = int(allocation_per / live_price)

#             if target_qty < 1:
#                 continue

#             # =============================================
#             # STALE POSITION CHECK
#             # =============================================
#             bought_at = portfolio_state["bought_at"].get(symbol)
            
#             if qty > 0 and bought_at and entry_price:
            
#                 held_time = time.time() - bought_at
            
#                 pnl_pct = (live_price - entry_price) / entry_price
            
#                 if (
#                     held_time > STALE_TIME
#                     and -STALE_THRESHOLD < pnl_pct < STALE_THRESHOLD
#                 ):
            
#                     print(f"🔴 STALE SELL {symbol} pnl={pnl_pct:.3f}")
            
#                     try:
            
#                         api.submit_order(
#                             symbol=symbol,
#                             qty=qty,
#                             side="sell",
#                             type="market",
#                             time_in_force="day"
#                         )
            
#                         portfolio_state["entry_prices"].pop(symbol, None)
#                         portfolio_state["bought_at"].pop(symbol, None)
            
#                         LAST_TRADE_TS[symbol] = time.time()
            
#                         num_open -= 1
            
#                     except Exception as e:
#                         print(f"[STALE SELL ERROR] {symbol}: {e}")
            
#                     continue

#             # =============================================
#             # BUY
#             # =============================================
#             if qty == 0 and prob >= THRESHOLD:

#                 # cooldown
#                 last_trade = LAST_TRADE_TS.get(symbol, 0)

#                 if time.time() - last_trade < TRADE_COOLDOWN:
#                     print(f"⏳ Cooldown {symbol}")
#                     continue

#                 # don't exceed max positions
#                 if num_open >= MAX_POSITIONS:
#                     continue

#                 # require stronger confidence
#                 if prob < 0.60:
#                     continue

#                 print(f"🟢 BUY {symbol} prob={prob:.3f}")

#                 try:

#                     api.submit_order(
#                         symbol=symbol,
#                         qty=target_qty,
#                         side="buy",
#                         type="market",
#                         time_in_force="day"
#                     )

#                     portfolio_state["entry_prices"][symbol] = live_price
#                     portfolio_state["bought_at"][symbol] = time.time()

#                     LAST_TRADE_TS[symbol] = time.time()

#                     num_open += 1

#                 except Exception as e:
#                     print(f"[BUY ERROR] {symbol}: {e}")

#             # =============================================
#             # SELL
#             # =============================================
#             elif qty > 0 and (
#                 prob < 0.48
#                 or pnl_pct >= TAKE_PROFIT
#             ):

#                 print(f"🔴 SELL {symbol} prob={prob:.3f}")

#                 try:

#                     api.submit_order(
#                         symbol=symbol,
#                         qty=qty,
#                         side="sell",
#                         type="market",
#                         time_in_force="day"
#                     )

#                     portfolio_state["entry_prices"].pop(symbol, None)
#                     portfolio_state["bought_at"].pop(symbol, None)

#                     LAST_TRADE_TS[symbol] = time.time()

#                     num_open -= 1

#                 except Exception as e:
#                     print(f"[SELL ERROR] {symbol}: {e}")

#         except Exception as e:

#             print(f"[symbol loop error] {symbol}: {e}")

#     portfolio_state.update({
#         "positions": positions_snapshot,
#         "last_probs": probs_snapshot,
#         "last_prices": prices_snapshot,
#         "equity": equity,
#         "cash": cash
#     })

#     print(f"💼 Equity: {equity:.2f} | Cash: {cash:.2f}")

# # =========================================================
# # ENGINE
# # =========================================================
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
            print("Checking market")
            if not market_is_open():
                print("🌙 Market closed — sleeping")
                time.sleep(60)
                continue
            print("Checking market done")
            print(f"Is trained: {is_trained}") 
            if not is_trained:
                print("⚠️ Model not trained yet")
                time.sleep(5)
                continue
            print("ranking...")
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

            print("TRAINED:", is_trained)
            print("MODEL:", model)
            print("SCALER:", scaler)

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
    reload_portfolio_state()
    model = None
    if model is None or scaler is None:
        print("🧠 No weights found → training fresh model")
        train()
        save_weights_to_supabase()
    if model is not None:
        is_trained = True

    # Start engine thread (TRADING LOOP)
    engine_thread = threading.Thread(target=run_engine, daemon=True)
    engine_thread.start()
    print("⚙️ Trading engine thread started")

    # Start Flask in main thread
    run_flask()


if __name__ == "__main__":
    start()
