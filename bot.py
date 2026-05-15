import os
import time
import threading
import numpy as np
import pandas as pd
from flask import Flask, jsonify
import alpaca_trade_api as tradeapi

# ================= CONFIG =================

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

SYMBOLS = ["AAPL", "TSLA", "NVDA", "AMD", "SPY"]

TIMEFRAME = "1Min"
LOOKBACK = 200

RISK_PER_TRADE = 0.01
STOP_LOSS_PCT = 0.005
TAKE_PROFIT_PCT = 0.01

MIN_MOVE = 0.0015
MAX_SPREAD = 0.002
COOLDOWN_SECONDS = 300
DAILY_LOSS_LIMIT = -0.03

# ==========================================

app = Flask(__name__)
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)

positions = {}
last_trade_time = {}
daily_start_equity = None

# ============== HELPERS ===================

def get_account():
    return api.get_account()

def get_equity():
    return float(get_account().equity)

def get_daily_pnl():
    global daily_start_equity
    equity = get_equity()
    if daily_start_equity is None:
        daily_start_equity = equity
        return 0
    return (equity - daily_start_equity) / daily_start_equity

def get_data(symbol):
    bars = api.get_bars(symbol, TIMEFRAME, limit=LOOKBACK).df

    if bars is None or bars.empty:
        return pd.DataFrame()

    if 'symbol' in bars.columns:
        return bars[bars['symbol'] == symbol].copy()

    return bars.copy()

def compute_indicators(df):
    df['rsi'] = compute_rsi(df['close'])
    df['ma_fast'] = df['close'].rolling(10).mean()
    df['ma_slow'] = df['close'].rolling(50).mean()
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['volume_avg'] = df['volume'].rolling(20).mean()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ============== SIGNAL ENGINE ===================

def generate_signal(symbol, df):



    if len(df) < 60:
        return None

    df = compute_indicators(df)
    latest = df.iloc[-1]

    price = latest['close']

    spread = abs(df['high'].iloc[-1] - df['low'].iloc[-1]) / price
    if spread > MAX_SPREAD:
        return None

    # ---------- MIN MOVE FILTER ----------
    recent_move = abs(df['close'].pct_change().iloc[-1])
    if recent_move < MIN_MOVE:
        return None

    # ---------- VOLUME FILTER ----------
    if latest['volume'] < latest['volume_avg']:
        return None

    # ---------- TREND ----------
    trend_up = latest['ma_fast'] > latest['ma_slow']
    trend_down = latest['ma_fast'] < latest['ma_slow']

    # ---------- RSI ----------
    rsi = latest['rsi']

    # ---------- COMPRESSION ----------
    volatility = df['volatility'].iloc[-1]
    compression = volatility < 0.01

    # ---------- DROP DETECTION ----------
    lookback_drop = (df['close'].iloc[-20] - price) / df['close'].iloc[-20]

    # ---------- BREAKOUT ----------
    breakout_up = (
        compression and
        lookback_drop > 0.02 and
        price > df['close'].rolling(10).mean().iloc[-1] and
        df['close'].iloc[-1] > df['close'].iloc[-2]
    )

    breakout_down = (
        compression and
        lookback_drop < -0.02 and
        price < df['close'].rolling(10).mean().iloc[-1]
    )

    # ---------- SIGNAL LOGIC ----------
    signal = None

    if breakout_up:
        signal = "BUY"

    elif breakout_down:
        signal = "SELL"

    elif trend_up and rsi < 40:
        signal = "BUY"

    elif trend_down and rsi > 60:
        signal = "SELL"

    # ---------- DEBUG ----------
    print(f"""
    SYMBOL: {symbol}
    PRICE: {price}
    SIGNAL: {signal}
    RSI: {rsi}
    VOL: {volatility}
    DROP: {lookback_drop}
    """)

    return signal

# ============== EXECUTION ===================

def place_trade(symbol, signal, price):

    open_positions = [p.symbol for p in api.list_positions()]

    if symbol in open_positions:
        return
    
    # ---------- DAILY LOSS LIMIT ----------
    if get_daily_pnl() < DAILY_LOSS_LIMIT:
        print("Daily loss limit hit. Stopping trading.")
        return

    # ---------- COOLDOWN ----------
    now = time.time()
    if symbol in last_trade_time and now - last_trade_time[symbol] < COOLDOWN_SECONDS:
        return

    # ---------- POSITION SIZING ----------
    equity = get_equity()
    risk = equity * RISK_PER_TRADE
    stop_distance = price * STOP_LOSS_PCT
    qty = int(risk / stop_distance)

    if qty <= 0:
        return

    try:
        if signal == "BUY":
            api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
        elif signal == "SELL":
            api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')

        last_trade_time[symbol] = now

        print(f"TRADE: {signal} {symbol} QTY: {qty}")

    except Exception as e:
        print(f"Order failed: {e}")

# ============== MAIN LOOP ===================

def trading_loop():
    while True:
        try:
            for symbol in SYMBOLS:
                df = get_data(symbol)
                
                if df is None or df.empty or len(df) < 60:
                    continue
                signal = generate_signal(symbol, df)

                if signal:
                    price = df['close'].iloc[-1]
                    place_trade(symbol, signal, price)

        except Exception as e:
            print(f"Loop error: {e}")

        time.sleep(10)

# ============== API ===================

@app.route("/")
def home():
    return jsonify({"status": "running"})

@app.route("/pnl")
def pnl():
    return jsonify({"daily_pnl": get_daily_pnl()})

@app.route("/account")
def account():
    try:
        acc = api.get_account()

        return jsonify({
            "equity": float(acc.equity),
            "cash": float(acc.cash),
            "buying_power": float(acc.buying_power),
            "portfolio_value": float(acc.portfolio_value),
            "status": acc.status
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/equity")
def equity():
    try:
        current_equity = get_equity()

        return jsonify({
            "current_equity": current_equity,
            "start_equity": daily_start_equity,
            "daily_pnl_pct": get_daily_pnl()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/portfolio")
def portfolio():
    try:
        positions = api.list_positions()

        data = []
        for p in positions:
            data.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": p.side,
                "market_value": float(p.market_value),
                "avg_entry_price": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc)
            })

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)})
# ============== START ===================

if __name__ == "__main__":
    threading.Thread(target=trading_loop).start()
    app.run(host="0.0.0.0", port=5000)
