import os
import time
import threading
from datetime import datetime, timedelta

import pandas as pd
import alpaca_trade_api as tradeapi
from flask import Flask, jsonify

API_KEY = os.getenv("APCA_API_KEY_ID")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

SYMBOL = "AAPL"
TRADE_QTY = 1

import os
print("APCA_API_KEY_ID exists:", bool(os.getenv("APCA_API_KEY_ID")))
print("APCA_API_SECRET_KEY exists:", bool(os.getenv("APCA_API_SECRET_KEY")))
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
app = Flask(__name__)

# -------------------------
# Trading logic
# -------------------------

def get_price_history(symbol):
    now = datetime.utcnow()
    past = now - timedelta(minutes=10)

    bars = api.get_bars(
        symbol,
        tradeapi.TimeFrame.Minute,
        past.isoformat(),
        now.isoformat()
    ).df

    return bars

def get_position():
    try:
        position = api.get_position(SYMBOL)
        return float(position.avg_entry_price), int(position.qty)
    except:
        return None, 0

def buy():
    print("📈 BUY")
    api.submit_order(
        symbol=SYMBOL,
        qty=TRADE_QTY,
        side="buy",
        type="market",
        time_in_force="gtc"
    )

def sell():
    print("📉 SELL")
    api.submit_order(
        symbol=SYMBOL,
        qty=TRADE_QTY,
        side="sell",
        type="market",
        time_in_force="gtc"
    )

def run_strategy():
    bars = get_price_history(SYMBOL)

    if len(bars) < 5:
        return

    recent = bars.tail(5)
    old_price = recent["close"].iloc[0]
    current_price = recent["close"].iloc[-1]

    change_pct = ((current_price - old_price) / old_price) * 100

    entry_price, qty = get_position()

    print(f"{datetime.utcnow()} | {current_price:.2f} | {change_pct:.2f}%")

    if qty == 0:
        if change_pct <= -1.0:
            buy()
    else:
        profit_pct = ((current_price - entry_price) / entry_price) * 100

        if profit_pct >= 1.5 or profit_pct <= -2.0:
            sell()

def trading_loop():
    while True:
        try:
            run_strategy()
        except Exception as e:
            print(f"Trading error: {e}")

        time.sleep(60)

# -------------------------
# Reporting API
# -------------------------

@app.route("/status")
def status():
    account = api.get_account()
    equity = float(account.equity)
    cash = float(account.cash)

    return jsonify({
        "equity": equity,
        "cash": cash,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/report")
def report():
    account = api.get_account()
    positions = api.list_positions()
    activities = api.get_activities()

    pos_data = []
    for p in positions:
        pos_data.append({
            "symbol": p.symbol,
            "qty": int(p.qty),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "unrealized_pl": float(p.unrealized_pl)
        })

    trades = []
    for a in activities[:20]:  # last 20 trades
        trades.append({
            "symbol": a.symbol,
            "side": a.side,
            "qty": a.qty,
            "price": a.price,
            "date": a.transaction_time
        })

    return jsonify({
        "equity": float(account.equity),
        "cash": float(account.cash),
        "positions": pos_data,
        "recent_trades": trades
    })

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    print("🚀 Bot + API running")

    t = threading.Thread(target=trading_loop)
    t.start()

    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
