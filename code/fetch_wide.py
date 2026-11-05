"""Wider, more varied universe for the robustness study: the original six + ~30 more series (Indian large/mid caps incl. collapsed names, US blue chips and decliners,
international indices, gold and bond ETFs) -> data/wide_close.csv.  Yahoo serves only tickers that still exist; fully delisted failures cannot be included (survivorship remains)."""
import yfinance as yf, pandas as pd
CORE = ["^NSEI", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "AAPL", "TSLA"]
NEW = ["INFY.NS", "ITC.NS", "SBIN.NS", "ONGC.NS", "HINDUNILVR.NS", "ADANIENT.NS", "YESBANK.NS", "IDEA.NS", "SUZLON.NS", "TATAMOTORS.NS",
       "MSFT", "JPM", "XOM", "JNJ", "KO", "INTC", "GE", "F", "PFE", "GME", "CCL", "BAC", "WMT",
       "^GSPC", "^N225", "^FTSE", "^GDAXI", "GLD", "TLT", "BABA", "TM"]
d = yf.download(CORE + NEW, start="2010-01-01", end="2026-10-31", progress=False, auto_adjust=True)["Close"]
n = d.notna().sum(); good = n[n >= 2500].index.tolist(); bad = n[n < 2500]
print("kept", len(good), "dropped", dict(bad)); print(n[good].to_string())
d[good].to_csv("data/wide_close.csv")
