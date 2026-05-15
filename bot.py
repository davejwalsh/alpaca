import os
import time
import threading
from collections import deque, defaultdict

import numpy as np
import alpaca_trade_api as tradeapi
from flask import Flask, jsonify

print("🔥 QUANT SYSTEM v6.1 (STABLE WRAPPER FIX)")

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
# CACHE (IMPORTANT FIX)
# =========================================================
cached_account = None
cached_positions = []
cached_equity = None
lock = threading.Lock()

# =========================================================
# SAFE WRAPPERS
# =========================================================
def safe_call(fn, default=None, retries=3):
    for _ in range(retries):
        try:
            return fn()
        except Exception as e:
            print("API retry:", e)
            time.sleep(1)
    return default

# =========================================================
# ACCOUNT THREAD
# =========================================================
def account_updater():
    global cached_account, cached_equity, cached_positions

    while True:
        acc = safe_call(api.get_account)

        if acc:
            with lock:
                cached_account = acc
                cached_equity = float(acc.equity)

        pos = safe_call(api.list_positions, default=[])
        if pos is not None:
            with lock:
                cached_positions = pos

        time.sleep(10)

# =========================================================
# HELPERS
# =========================================================
def equity():
    with lock:
        return cached_equity

def cash():
    with lock:
        return float(cached_account.cash) if cached_account else 0.0

def positions():
    with lock:
        return cached_positions

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
    return safe_call(
        lambda: api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=120).df,
        default=pd.DataFrame()
    )

def returns(series, n):
    if len(series) < n:
        return 0
    return series.iloc[-1] / series.iloc[-n] - 1

# =========================================================
# REGIME
# =========================================================
def market_regime():
    df = get_data("SPY")
    if df is None or df.empty:
        return "neutral"

    close = df["close"]
    vol = close.pct_change().rolling(30).std().iloc[-1]
    trend = returns(close, 30)

    if vol > 0.035:
        return "stress"
    if vol > 0.02 and trend < 0:
        return "risk_off"
    if trend > 0.01:
        return "risk_on"
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
# EXPOSURE (CACHED SAFE)
# =========================================================
def exposure():
    pos = positions()
    eq = equity()
    if not eq:
        return 0
    return sum(float(p.market_value) for p in pos) / eq

# =========================================================
# EXECUTION (UNCHANGED LOGIC)
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
# ENGINE (ONLY REDUCED API LOAD)
# =========================================================
def engine():
    global paused

    while True:
        print("\n🔁 CYCLE")

        if paused:
            time.sleep(5)
            continue

        eq = equity()
        if not eq:
            time.sleep(CHECK_INTERVAL)
            continue

        reg = market_regime()
        reg_mult = regime_multiplier(reg)
        current_exp = exposure()

        scored = []

        for s in SYMBOLS:
            df = get_data(s)
            if df is None or len(df) < 60:
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

        equity_curve.append(eq)
        cash_curve.append(float(cash()))

        print(f"💼 EQ={eq:.2f} EXP={current_exp:.2f}")

        time.sleep(CHECK_INTERVAL)

# =========================================================
# API (NO DIRECT ALPACA CALLS)
# =========================================================
@app.route("/")
def home():
    return {"status": "running"}

@app.route("/status")
def status():
    return jsonify({
        "equity": equity(),
        "cash": cash(),
        "regime": market_regime(),
        "exposure": exposure()
    })

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
    threading.Thread(target=account_updater, daemon=True).start()
    threading.Thread(target=engine, daemon=True).start()

if __name__ == "__main__":
    print("🚀 QUANT SYSTEM v6.1 START")
    start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
