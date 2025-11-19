import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

def fetch_yahoo_option_chain(ticker = "", expiry = ""):
    tkr = yf.Ticker(ticker)
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
    
    chain = tkr.option_chain(expiry)
    calls = chain.calls
    calls['mid'] = (calls['bid'] + calls['ask']) / 2
    mask = calls['mid'] <= 0
    calls.loc[mask, "mid"] = calls.loc[mask, "lastPrice"]
    calls = calls[calls['mid'] > 0].copy()

    ### time to expiry in years ###
    now = datetime.now(timezone.utc)
    exp_date = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dt_days = (exp_date - now).total_seconds() / (365 * 24 * 60 * 60)
    T = max(dt_days, 1/365)
    K = calls['strike'].values
    C = calls['mid'].values
    r = 0 

    return S0, T, r, K, C, calls

def choose_ticker_and_expiry(default_ticker = ""):
    raw_ticker = input(f"Enter ticker symbol (default: {default_ticker}): ").strip()
    ticker = raw_ticker if raw_ticker else default_ticker
    tkr = yf.Ticker(ticker)
    expires = tkr.options
    
    if not expires:
        raise ValueError(f"No options found for ticker {ticker}")
        return ticker, None 
    print("Available expiry dates:")
    
    for i, j in enumerate(expires):
        print(f" {i}: {j}")
    choice = input("Choose expiry by index (default: 0): ").strip()
    if choice == "":
        idx = 0
    
    else:
        try:
            idx = int(choice)
            if idx < 0 or idx >= len(expires):
                raise ValueError()
            expiry = expires[idx]
        except ValueError:
            raise ValueError("Invalid expiry index")
    expiry = expires[idx]
    print(f"Selected ticker: {ticker}, expiry: {expiry}")
    return ticker, expiry

