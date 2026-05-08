import os
import time
import threading
from collections import deque, defaultdict
import numpy as np
import alpaca_trade_api as tradeapi
from flask import Flask, jsonify

print("🔥 HEDGE FUND v3 STARTING")

# =========================================================
# CONFIG
# =========================================================
CHECK_INTERVAL = 30
COOLDOWN = 300
MAX_POSITIONS = 12

BASE_RISK = 0.01
MAX_EXPOSURE = 0.80

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

strategy_pnl = {"rules": deque(maxlen=200), "ml": deque(maxlen=200)}
strategy_weight = {"rules": 0.5, "ml": 0.5}

ml_weights = defaultdict(lambda: np.random.uniform(-0.3, 0.3))

# =========================================================
# UNIVERSE
# =========================================================
SYMBOLS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","V","UNH",
    "HD","PG","MA","DIS","BAC","XOM","AVGO","LLY","ADBE","COST",
    "PEP","KO","CRM","MRK","ABT","CVX","TMO","WMT","CSCO","MCD",
    "ACN","DHR","AMD","TXN","NEE","LIN","PM","UPS","ORCL","BMY",
    "QCOM","LOW","INTC","SPGI","CAT","GS","MS","BLK"
]

# crude sector mapping (proxy)
SECTOR = {
    "AAPL":"tech","MSFT":"tech","GOOGL":"tech","META":"tech","NVDA":"tech",
    "TSLA":"auto","JPM":"fin","BAC":"fin","GS":"fin","MS":"fin",
    "XOM":"energy","CVX":"energy",
    "WMT":"consumer","COST":"consumer","MCD":"consumer"
}

# =========================================================
# HELPERS
# =========================================================
def equity():
    return float(api.get_account().equity)

def cash():
    return float(api.get_account().cash)

def positions():
    return api.list_positions()

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
    return api.get_bars(symbol, tradeapi.TimeFrame.Minute, limit=80).df

def returns(s, n):
    return s.iloc[-1] / s.iloc[-n] - 1

# =========================================================
# REGIME DETECTION
# =========================================================
def market_regime():
    # broad index proxy = SPY if available via AAPL basket proxy
    try:
        df = get_data("AAPL")
        vol = df["close"].pct_change().rolling(20).std().iloc[-1]
        trend = returns(df["close"], 20)

        if vol > 0.04:
            return "crash"
        if vol > 0.025:
            return "risk_off"
        if trend > 0.02:
            return "risk_on"
        return "chop"
    except:
        return "chop"

# =========================================================
# SIGNALS
# =========================================================
def rules_signal(df):
    return returns(df["close"], 5) * 0.6 + returns(df["close"], 20) * 0.4

def ml_features(df):
    return {
        "r5": returns(df["close"], 5),
        "r10": returns(df["close"], 10),
        "r20": returns(df["close"], 20)
    }

def ml_signal(symbol, df):
    f = ml_features(df)
    return sum(f[k] * ml_weights[f"{symbol}_{k}"] for k in f)

# =========================================================
# CORRELATION FILTER (simple proxy)
# =========================================================
def correlation_penalty(symbol, scored_symbols):
    sector = SECTOR.get(symbol, "other")
    return sum(1 for s in scored_symbols if SECTOR.get(s, "other") == sector) * 0.15

# =========================================================
# RISK MODEL
# =========================================================
def volatility(df):
    return max(df["close"].pct_change().rolling(20).std().iloc[-1], 1e-6)

def regime_risk_multiplier(regime):
    return {
        "risk_on": 1.2,
        "chop": 1.0,
        "risk_off": 0.6,
        "crash": 0.3
    }.get(regime, 1.0)

# =========================================================
# ENGINE
# =========================================================
def engine():
    global paused

    while True:
        print("\n🔁 HEDGE FUND CYCLE v3")

        if paused:
            time.sleep(5)
            continue

        eq = equity()
        reg = market_regime()
        risk_mult = regime_risk_multiplier(reg)

        current_positions = {p.symbol: p for p in positions()}
        exposure = sum(float(p.market_value) for p in current_positions.values()) / eq

        scored = []

        # =====================================================
        # SCAN
        # =====================================================
        for s in SYMBOLS:
            try:
                df = get_data(s)
                if len(df) < 40:
                    continue

                r = rules_signal(df)
                m = ml_signal(s, df)

                signal = (strategy_weight["rules"] * r +
                          strategy_weight["ml"] * m)

                signal += returns(df["close"], 1) * 0.2

                vol = volatility(df)

                # regime scaling
                signal = signal * risk_mult / vol

                if abs(signal) < 0.02:
                    continue

                scored.append((s, signal, df, vol))

            except:
                continue

        if not scored:
            time.sleep(CHECK_INTERVAL)
            continue

        # =====================================================
        # RANK
        # =====================================================
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        top = scored[:MAX_POSITIONS]

        symbols_only = [x[0] for x in top]

        print(f"📊 REGIME: {reg} | TOP: {symbols_only}")

        available = max(0, (MAX_EXPOSURE - exposure)) * eq

        total_strength = sum(abs(x[1]) for x in top) + 1e-9

        # =====================================================
        # TRADE
        # =====================================================
        for symbol, signal, df, vol in top:

            price = df["close"].iloc[-1]
            entry, qty = position(symbol)

            sector_penalty = sum(
                1 for s in symbols_only
                if SECTOR.get(s) == SECTOR.get(symbol)
            )

            adjusted_signal = signal * (1 - sector_penalty * 0.1)

            weight = abs(adjusted_signal) / total_strength
            allocation = available * weight

            size = int(allocation / (price * vol))
            size = max(0, min(size, 10))

            now = time.time()

            # ENTRY
            if qty == 0 and adjusted_signal > 0 and size > 0:

                if symbol in last_trade_time:
                    if now - last_trade_time[symbol] < COOLDOWN:
                        continue

                print(f"🟢 BUY {symbol} size={size}")
                api.submit_order(symbol, size, "buy", "market", "gtc")
                last_trade_time[symbol] = now

            # EXIT
            if qty > 0:
                pnl = (price - float(entry)) / float(entry)

                if pnl < -0.02 * risk_mult:
                    api.submit_order(symbol, qty, "sell", "market", "gtc")

                elif pnl > 0.10:
                    api.submit_order(symbol, max(1, qty // 2), "sell", "market", "gtc")

                elif pnl > 0.20:
                    api.submit_order(symbol, qty, "sell", "market", "gtc")

        equity_curve.append(eq)
        cash_curve.append(cash())

        print(f"💼 EQ={eq:.2f} EXP={exposure:.2f}")

        time.sleep(CHECK_INTERVAL)

# =========================================================
# API ENDPOINTS
# =========================================================
@app.route("/")
def home():
    return {"status": "running"}

@app.route("/status")
def status():
    acc = api.get_account()
    return jsonify({
        "equity": float(acc.equity),
        "cash": float(acc.cash),
        "regime": market_regime(),
        "paused": paused
    })

@app.route("/portfolio")
def portfolio():
    pos = api.list_positions()
    return jsonify([{
        "symbol": p.symbol,
        "qty": int(p.qty),
        "value": float(p.market_value)
    } for p in pos])

@app.route("/equity")
def equity_api():
    return {"equity": list(equity_curve)}

@app.route("/cash")
def cash_api():
    return {"cash": list(cash_curve)}

@app.route("/diagnostics")
def diagnostics():
    return {
        "regime": market_regime(),
        "strategy_weights": strategy_weight,
        "num_positions": len(positions())
    }

# =========================================================
# START
# =========================================================
def start():
    time.sleep(5)
    threading.Thread(target=engine, daemon=True).start()

if __name__ == "__main__":
    print("🚀 HEDGE FUND v3 LIVE")
    threading.Thread(target=start, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
