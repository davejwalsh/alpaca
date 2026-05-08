FROM python:3.11

WORKDIR /app
COPY . .

RUN pip install alpaca-trade-api

CMD ["python", "bot.py"]
