import numpy as np
from numpy.random import default_rng

from bubble_detector.models import sim_cev_paths, sim_bs_paths
from bubble_detector.Delta_band import delta_band_from_calls, plot_call_and_tail_delta_band


if __name__ == "__main__":
    S0, T = 100.0, 1.0
    rng = np.random.default_rng(2)

    ### CEV ###
    ST = sim_cev_paths(N=30000, M=1500, T=T, rng=rng, sigma=0.16, beta=1.5, S0=S0)

    ### price calls on a grid from the MC ST r=d=0 ###
    K = np.linspace(0.1*S0, 3.0*S0, 60)
    C = np.maximum(ST[:,None] - K[None,:], 0.0).mean(axis=0)

    est = delta_band_from_calls(K, C, S0=S0, T=T, r=0.0, d=0.0,
                            band=(0,1), left_mode="linear")

    ### plot ###
    print(f"E_Q[S_T]_hat = {est['E_hat']:.3f},   m_hat = {est['m_hat']:.3f} (S0 - E_hat)")
    plot_call_and_tail_delta_band(est, title="Delta-band reconstruction")
    plot_call_and_tail_delta_band(est, show="observed", title="Delta-band reconstruction")

    ### Black-Scholes ###
    ST = sim_bs_paths(N=30000, M=1500, T=T, rng=rng, sigma=0.20, S0=S0)

    ### price calls on a grid from the MC ST r=d=0 ###
    K = np.linspace(0.1*S0, 3.0*S0, 60)
    C = np.maximum(ST[:,None] - K[None,:], 0.0).mean(axis=0)

    est = delta_band_from_calls(K, C, S0=S0, T=T, r=0.0, d=0.0,
                            band=(0,1), left_mode="linear")

    ### plot ###
    print(f"E_Q[S_T]_hat = {est['E_hat']:.3f},   m_hat = {est['m_hat']:.3f} (S0 - E_hat)")
    plot_call_and_tail_delta_band(est, title="Delta-band reconstruction")
    plot_call_and_tail_delta_band(est, show="observed", title="Delta-band reconstruction")