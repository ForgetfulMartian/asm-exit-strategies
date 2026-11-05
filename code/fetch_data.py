"""Download daily adjusted closes for the six series (Yahoo Finance via yfinance) -> data/basket_close.csv"""
import yfinance as yf
tk = ["^NSEI", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "AAPL", "TSLA"]
d = yf.download(tk, start="2010-01-01", end="2026-10-31", progress=False, auto_adjust=True)["Close"]
d.to_csv("data/basket_close.csv"); print(d.shape)
