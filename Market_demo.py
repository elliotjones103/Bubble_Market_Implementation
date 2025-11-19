import numpy as np
from bubble_detector.data import fetch_yahoo_option_chain
from bubble_detector.Delta_band import delta_band_from_calls, plot_call_and_tail_delta_band
from bubble_detector.data import choose_ticker_and_expiry

if __name__ == "__main__":
    ticker, expiry = choose_ticker_and_expiry(default_ticker="AAPL")

    s0, T, r, K, C, calls = fetch_yahoo_option_chain(ticker=ticker, expiry=expiry)
    print(f" S0 = {s0:.2f},   T = {T:.4f} years,   r = {r:.4f},  call quotes = {len(C)}") 

    mask = (K >= 0.5*s0) & (K <= 2.0*s0)
    K = K[mask]
    C = C[mask]

    est = delta_band_from_calls(K, C, S0=s0, T=T, r=r, d=0.0,
                            band=(0.25,0.9), left_mode="linear")
    
    print(f"E_Q[S_T]_hat = {est['E_hat']:.3f},   m_hat = {est['m_hat']:.3f} (S0 - E_hat)")

    plot_call_and_tail_delta_band(est, title=f"Delta-band reconstruction for {ticker} exp {expiry}")
    plot_call_and_tail_delta_band(est, show="observed", title=f"Delta-band reconstruction for {ticker} exp {expiry}")
