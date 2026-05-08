import os
import time
import threading
from datetime import datetime, timedelta

import pandas as pd
from flask import Flask, jsonify
from ib_insync import *

app = Flask(__name__)

# -------------------------
# IBKR Setup
# -------------------------

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=1)

SYMBOL = "AAPL"
TRADE_QTY = 1

contract = Stock(SYMBOL, "SMART", "USD")

# -------------------------
# Helpers
# -------------------------

def get_price_history():
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr='10 M',
        barSizeSetting='1 min',
        whatToShow='TRADES',
        useRTH=True
    )

    df = util.df(bars)
    return df

def get_position():
    positions = ib.positions()
    for p in positions:
        if p.contract.symbol == SYMBOL:
            return p.avgCost, p.position
    return None, 0

def get_current_price():
    ticker = ib.reqMktData(contract, "", False, False)
    ib.sleep(2)
    return ticker.last

def buy():
    print("📈 BUY")
    order = MarketOrder("BUY", TRADE_QTY)
    ib.placeOrder(contract, order)

def sell():
    print("📉 SELL")
    order = MarketOrder("SELL", TRADE_QTY)
    ib.placeOrder(contract, order)

# -------------------------
# Strategy
# -------------------------

def run_strategy():
    df = get_price_history()

    if len(df) < 5:
        return

    recent = df.tail(5)
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
    account_values = ib.accountSummary()

    equity = next((float(v.value) for v in account_values if v.tag == "NetLiquidation"), 0)
    cash = next((float(v.value) for v in account_values if v.tag == "AvailableFunds"), 0)

    return jsonify({
        "equity": equity,
        "cash": cash,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/report")
def report():
    positions = ib.positions()
    trades = ib.trades()

    pos_data = []
    for p in positions:
        pos_data.append({
            "symbol": p.contract.symbol,
            "qty": p.position,
            "avg_cost": p.avgCost
        })

    trade_data = []
    for t in trades[-20:]:
        trade_data.append({
            "symbol": t.contract.symbol,
            "action": t.order.action,
            "qty": t.order.totalQuantity,
            "status": t.orderStatus.status
        })

    return jsonify({
        "positions": pos_data,
        "recent_trades": trade_data
    })

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    print("🚀 IBKR Bot running")

    t = threading.Thread(target=trading_loop)
    t.start()

    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
