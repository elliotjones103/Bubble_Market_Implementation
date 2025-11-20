import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from bubble_detector.data import fetch_yahoo_option_chain, choose_ticker_and_expiry
from bubble_detector.Delta_band import delta_band_from_calls, plot_call_and_tail_delta_band
from bubble_detector.models import sim_cev_paths, sim_bs_paths, sim_sabr_beta1_paths



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

st.sidebar.header("Inputs")
mode = st.sidebar.radio("Data source",["Market options", "Black-Scholes model", "CEV strict local model", "SABR model"],)

band_low = st.sidebar.slider("Delta band lower bound", 0.0, 0.9, 0.25, 0.05)
band_high = st.sidebar.slider("Delta band upper bound", band_low + 0.05, 1.0, 0.95, 0.05)
left_mode = st.sidebar.selectbox("Left extrapolation mode", ["linear", "min", "max"])

ticker = None
expiry = None
S0_input = None
T_input = None
sigma_bs = None
sigma_cev = None
beta_cev = None
sigma_sabr = None
rho_sabr = None
nu_sabr = None


### Market Mode ###
if mode == "Market options":
    default_ticker = "AAPL"
    ticker = st.sidebar.text_input("Ticker symbol", value=default_ticker).upper()
    # Fetch expiries for that ticker
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

### Model Mode ### 

elif mode == "Black-Scholes model":
    S0_input = st.sidebar.number_input(r"Initial spot price $S_0$", value=100.0, min_value=0.01, max_value=200.00)
    T_input = st.sidebar.number_input(r"Time to maturity $T$ (years)", value=1.0, min_value=0.01, max_value=5.00)
    r_input = st.sidebar.number_input(r"Risk-free rate $r$ (annualized)", value=0.0, min_value=0.0, max_value=1.00)
    sigma_bs = st.sidebar.number_input(r"Volatility $\sigma$ (annualized)", value=0.20, min_value=0.01, max_value=5.00)

elif mode == "CEV strict local model":
    S0_input = st.sidebar.number_input(r"Initial spot price $S_0$", value=100.0, min_value=0.01, max_value=200.00)
    T_input = st.sidebar.number_input(r"Time to maturity $T$ (years)", value=1.0, min_value=0.01, max_value=5.00)
    r_input = st.sidebar.number_input(r"Risk-free rate $r$ (annualized)", value=0.0, min_value=0.0, max_value=1.00)
    sigma_cev = st.sidebar.number_input(r"Volatility $\sigma$ (annualized)", value=0.16, min_value=0.01, max_value=5.00)
    beta_cev = st.sidebar.number_input(r"Elasticity $\beta$ (>1)", value=1.5, min_value=0.00, max_value=5.00)

elif mode == "SABR model":
    S0_input = st.sidebar.number_input(r"Initial spot price $S_0$", value=100.0, min_value=0.01, max_value=200.00)
    T_input = st.sidebar.number_input(r"Time to maturity $T$ (years)", value=1.0, min_value=0.01, max_value=5.00)
    r_input = st.sidebar.number_input(r"Risk-free rate $r$ (annualized)", value=0.0, min_value=0.0, max_value=1.00)
    sigma_sabr = st.sidebar.number_input(r"Volatility $\sigma$ (annualized)", value=0.25, min_value=0.01, max_value=5.00)
    rho_sabr = st.sidebar.number_input(r"Correlation $\rho$ (-1 to 1)", value=0.70, min_value=-1.00, max_value=1.00)
    nu_sabr = st.sidebar.number_input(r"Volatility of volatility $\nu$ (annualized)", value=2.00, min_value=0.01, max_value=5.00)

run_button = st.sidebar.button("Run Delta-band Estimation")

tab_app, tab_math = st.tabs(["Estimator", "Mathematical background"])

with tab_app:
    if run_button:
        try:    
            rng = np.random.default_rng(1)
            
            
            if mode == "Market options":
                with st.spinner("Fetching option chain and running estimator..."):
                    S0, T, r, K, C, calls_df = fetch_yahoo_option_chain(
                        ticker=ticker,
                        expiry=None if expiry in (None, "None") else expiry,
                    )

                    ### sensible strike region around the money ###
                    mask = (K > 0.7 * S0) & (K < 1.5 * S0)
                    K_use = K[mask]
                    C_use = C[mask]

            elif mode == "Black-Scholes model":
                S0 = float(S0_input)
                T = float(T_input)
                r = float(r_input)
                calls_df = None
                label = r"Black-Scholes model (\sigma={sigma_bs:.2f})"
                
                ST = sim_bs_paths(
                    N=30000, M=1500, T=T, rng=rng, sigma=sigma_bs, S0=S0
                )
                K = np.linspace(0.1 * S0, 3.0 * S0, 60)
                C = np.maximum(ST[:, None] - K[None, :], 0.0).mean(axis=0)
                K_use, C_use = K, C

            elif mode == "CEV strict local model":
                S0 = float(S0_input)
                T = float(T_input)
                r = float(r_input)
                calls_df = None
                label = r"CEV model (\sigma={sigma_cev:.2f}, \beta={beta_cev:.2f})"

                ST = sim_cev_paths(
                    N=30000, M=1500, T=T, rng=rng,
                    sigma=sigma_cev, beta=beta_cev, S0=S0
                )
                K = np.linspace(0.1 * S0, 3.0 * S0, 60)
                C = np.maximum(ST[:, None] - K[None, :], 0.0).mean(axis=0)
                K_use, C_use = K, C
                                

            elif mode == "SABR model":
                S0 = float(S0_input)
                T = float(T_input)
                r = float(r_input)
                calls_df = None
                label = (
                    r"SABR $\beta=1$ model ($\sigma$0={sigma0_sabr:.2f}, "
                    f"ν={nu_sabr:.2f}, ρ={rho_sabr:.2f})"
                )

                ST = sim_sabr_beta1_paths(
                    N=30000, M=1500, T=T, rng=rng,
                    sigma0=sigma_sabr, nu=nu_sabr, rho=rho_sabr, S0=S0
                )
                K = np.linspace(0.1 * S0, 3.0 * S0, 60)
                C = np.maximum(ST[:, None] - K[None, :], 0.0).mean(axis=0)
                K_use, C_use = K, C
            
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


            ### metrics row ###
            
            st.markdown("### Estimation Results ###")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(r"Spot price $S_0$ (USD)", f"{S0:.2f}")
            col2.metric(r"Time to expiry $T$ (years)", f"{T:.4f}")
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
            if mode == "Market options":
                with st.expander("Show raw option quotes used in estimation"):
                    st.dataframe(
                        calls_df[["strike", "bid", "ask", "lastPrice"]],
                        use_container_width=True,
                    )

            st.caption(
                r"Note: Interpret $\hat{m}(T)$ as a diagnostic signal. "
                "A value near zero is consistent with a true-martingale model; "
                "larger positive values suggest a potential bubble component under the chosen assumptions."
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.info("Choose inputs from the sidebar and click 'Run Delta-band Estimation'.")

with tab_math:
    st.header("Mathematical idea behind the delta-band bubble detector")

    st.markdown(r"""
Note: We work under a **risk–neutral measure** $Q$ and assume zero rates and dividends
($r = d = 0$) so that the discounted stock price $(S_t)_{t \le T}$ is a non-negative
local $Q$–martingale.

---

### Fundamental price and bubble gap

The **fundamental price** of the stock at time $t$ is
    """)
    st.latex(r"""
    S_t^* = \mathbb E^Q[S_T \mid \mathcal F_t].
    """)

    st.markdown(r"""
When $S$ is a true martingale, $S_t = S_t^*$ and there is no bubble.  
When $S$ is a **strict local martingale** it is possible for a bubble to exsist, it is a **supermartingale** and therefore

    """)
    st.latex(r"""
\mathbb E^Q[S_T \mid \mathcal F_t] \le S_t,
    """)
    st.markdown(r"""     
At $t=0$ we define the **martingale defect (bubble gap) as**
""")
    st.latex(r"""
                    
    m(T) := S_0 - \mathbb E^Q[S_T] \;\ge 0.
    """)

    st.markdown(r"""
---

### How options see the bubble

The **fundamental** call and put prices at maturity $T$ are
    """)

    st.latex(
        r"C^*(K,T) = \mathbb{E}^Q[(S_T - K)^+], \qquad "
        r"P^*(K,T) = \mathbb{E}^Q[(K - S_T)^+]."
    )

    st.markdown(r"""
Under mild no-arbitrage and no-dominance assumptions one can show that, in discounted
units,

    """)
    st.latex(r"""
    C(K,T) = C^*(K,T) + m(T), \qquad
    P(K,T) = P^*(K,T),
    """)

    st.markdown(r"""
and the deep-OTM limits are
                
    """)
    st.latex(r"""

\lim_{K \downarrow 0} C(K,T) = S_0, \qquad
\lim_{K \uparrow \infty} C(K,T) = m(T).
\newline


    """)
    st.markdown(r"""
So **far OTM calls form a flat plateau whose height equals the bubble gap** $m(T)$.

---

### Delta-band estimator from a finite call sheet

In practice we only observe calls for a finite grid of strikes. The delta-band estimator:

1. **Selects a liquid delta band:** $(\delta_{\min},\delta_{\max})$ and keeps calls
   whose Black-Scholes deltas fall in that range.
2. **Fits an arbitrage-free call curve:** $K \mapsto C(K,T)$ (decreasing and convex)
   using isotonic / convex regression, and differentiates it in $K$ to obtain an
   estimate of the risk-neutral tail probability
   """)
    st.latex(r"""
   q(K) \approx \mathbb Q(S_T \ge K) = -\partial_K C^*(K,T).
   
    """)
    st.markdown(r"""
3. **Reconstructs $\mathbb E^Q[S_T]$:** via
    """)
    st.latex(r"""
   \mathbb E^Q[S_T] = \int_0^\infty \mathbb Q(S_T \ge K)\,\mathrm dK
   \approx \int_0^\infty q(K)\,\mathrm dK
    """)
    st.markdown(r"""
   using robust extrapolation on the left and right tails.

Finally the **estimated martingale defect** is
$
\hat m(T) = S_0 - \widehat{\mathbb E^Q[S_T]}.
$

We apply the same estimator both to **simulated models** (Black–Scholes, CEV, SABR)
and to **live market options**, so you can compare the theory against real market data.
    """)

