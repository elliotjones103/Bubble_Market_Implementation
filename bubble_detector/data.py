import numpy as np
import pandas as pd
import yfinance as yf
for datetime import datetime, timezone

def fetch_yahoo_option_chain(ticker = "", expiry = ""):
    
    tkr = yf.ticker(ticker)
    hist = tkr.history(period="1d")
    if hist.empty:
        raise ValueError(f"Ticker {ticker} not found")
    S0 = hist["Close"].iloc[-1]

    # expiry date
    expires = tkr.options
    if not expires:
        raise ValueError(f"No options found for ticker {ticker}")
    if expiry is None:
        expiry = expires[0]
    elif expiry not in expires:
        raise ValueError(f"Expiry {expiry} not found for ticker {ticker}")
    
    chain = tkr.option_chain(ex)
    
