import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from bubble_detector.data import fetch_yahoo_option_chain, choose_ticker_and_expiry
from bubble_detector.Delta_band import delta_band_from_calls, plot_call_and_tail_delta_band

def make_delta_band_fig(est, show="fundamental", title="Delta-band reconstruction"):
    K, C = est["K"], est["C"]
    K_all  = est["curves"]["K_all"]
    Cstar  = est["curves"]["Cstar"]        
    m_hat  = est["m_hat"]
    
    if show == "observed":
        yline = Cstar + m_hat             
        label = r"reconstructed $C(K,T)=C^*(K,T)+\hat{m}$"
    else:  
        yline = Cstar
        label = r"reconstructed fundamental $C^*(K,T)$"
        
    ### focus on the useful range ###
    j = np.argmax(Cstar < 1e-4) if np.any(Cstar < 1e-4) else len(Cstar)-1
    xmax = K_all[max(10, j)]
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(K, C, s=14, label="calls (input)", color = "teal")
    ax.plot(K_all, yline, label=label, color = "peru")
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Call price C(K,T)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

### app ###

st.set_page_config(
    page_title="Delta-band Bubble Detector - NYSE Market",
    layout="wide",
)
st.title("Delta-band Bubble Detector")
st.markdown(
    "Estimate the martingale defect $m(T)$ from option prices using the "
"delta-band estimator developed in the Masters Thesis - Modelling and Quantifying the Martingale Defect in Asset Price Bubbles")

### Sidebar controls ###
st.sidebar.header("Inputs")
default_ticker = "AAPL"
ticker = st.sidebar.text_input("Ticker symbol", value=default_ticker).upper()

### Fetch expiries for that ticker ###
expiries = []
if ticker:
    try:
        tkr = yf.Ticker(ticker)
        expiries = tkr.options
    except Exception as e:
        st.sidebar.error(f"Error fetching expiries for {ticker}: {e}")

expiry = st.sidebar.selectbox(
    "Expiry date (nearest if first)",
    options=expiries if expiries else [None],
    index=0,
)

band_low = st.sidebar.slider("Delta band lower bound", 0.0, 0.9, 0.25, 0.05)
band_high = st.sidebar.slider("Delta band upper bound", band_low + 0.05, 1.0, 0.70, 0.05)
left_mode = st.sidebar.selectbox("Left extrapolation mode", ["linear", "min", "max"])
run_button = st.sidebar.button("Run Delta-band Estimation")

### Main logic ###
if run_button:
    try:
        with st.spinner("Fetching option chain and running estimator..."):
            S0, T, r, K, C, calls_df = fetch_yahoo_option_chain(
                ticker=ticker,
                expiry=None if expiry in (None, "None") else expiry,
            )

            ### sensible strike region around the money ###
            mask = (K > 0.7 * S0) & (K < 1.5 * S0)
            K_use = K[mask]
            C_use = C[mask]

            est = delta_band_from_calls(
                K_use,
                C_use,
                S0=S0,
                T=T,
                r=r,
                d=0.0,
                band=(band_low, band_high),
                left_mode=left_mode,
            )

        st.subheader(f"Results for {ticker} (exp {expiry})")

        ### metrics row ###
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Spot price $S_0$ (USD)", f"{S0:.2f}")
        col2.metric("Time to expiry $T$ (years)", f"{T:.4f}")
        col3.metric(r"$E_Q[S_T]$ (estimate)", f"{est['E_hat']:.3f}")
        col4.metric(r"$\hat m(T)$ (size of bubble gap)", f"{est['m_hat']:.3f}")

        st.write(
            f"Risk-free rate $r$ assumed: **{r:.4f}**. "
            "Results are sensitive to delta band, tail assumptions, and short-maturity noise."
        )
        st.divider()

        ### plots in a vertical stack ###
        fig1 = make_delta_band_fig(
            est,
            show="fundamental",
            title=f"Delta-band reconstruction for {ticker} exp {expiry} (fundamental)",
        )
        fig2 = make_delta_band_fig(
            est,
            show="observed",
            title=f"Delta-band reconstruction for {ticker} exp {expiry} (observed)",
        )

        col_left, col_right = st.columns(2)
        with col_left:
            st.pyplot(fig1, use_container_width=False)
        with col_right:
            st.pyplot(fig2, use_container_width=False)

        ### raw data ###
        with st.expander("Show raw option quotes used in estimation"):
            st.dataframe(
                calls_df[["strike", "bid", "ask", "lastPrice"]],
                use_container_width=True,
            )

        st.caption(
            "Interpret $\\hat m(T)$ as a diagnostic signal. "
            "A value near zero is consistent with a true-martingale model; "
            "larger positive values suggest a potential bubble component under the chosen assumptions."
        )
    except Exception as e:
        st.error(f"Something went wrong: {e}")