"""
Fetch the investor-fear data set used in human_factors.py and strategy_tournament.py.
  data/sentiment.csv : investor-fear indices from Yahoo Finance (India VIX, CBOE VIX), daily close.
Run: python code/fetch_human.py
"""
import pandas as pd, yfinance as yf

v = yf.download(["^INDIAVIX", "^VIX"], start="2010-01-01", end="2026-10-31", auto_adjust=True, progress=False)["Close"]
v.columns = ["INDIAVIX" if c == "^INDIAVIX" else "VIX" for c in v.columns]
v.to_csv("data/sentiment.csv"); print("sentiment", v.shape, v.index[0].date(), v.index[-1].date(), v.notna().sum().to_dict())
