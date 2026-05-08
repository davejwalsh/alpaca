import alpaca_trade_api as tradeapi

API_KEY = "YOUR_KEY"
SECRET_KEY = "YOUR_SECRET"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

account = api.get_account()
print(account)

# Buy $50 of Apple
api.submit_order(
    symbol="AAPL",
    qty=1,
    side="buy",
    type="market",
    time_in_force="gtc"
)
