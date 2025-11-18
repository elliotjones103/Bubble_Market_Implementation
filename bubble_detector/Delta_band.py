import math
import numpy as np
import matplotlib.pyplot as plt


### Black-Scholes ###
def _N(x):
    x = np.asarray(x)
    return 0.5*(1.0 + np.vectorize(math.erf)(x/np.sqrt(2.0)))

def bs_call_delta(S0, K, T, r, d, sigma):
    if T <= 0 or sigma <= 0:
        return float(np.exp(-d*T) * (1.0 if S0>K else (0.0 if S0<K else 0.5)))
    srt = sigma*np.sqrt(T)
    d1 = (np.log(S0/K) + (r - d + 0.5*sigma*sigma)*T)/srt
    return float(np.exp(-d*T)*_N(d1))

def implied_vol_call(price, S0, K, T, r, d, tol=1e-8, maxit=120):
    def _bs_price(S0, K, T, r, d, sigma):
        if T <= 0 or sigma <= 0: return float(max(S0 - K, 0.0))
        srt = sigma*np.sqrt(T)
        d1 = (np.log(S0/K) + (r - d + 0.5*sigma*sigma)*T)/srt
        d2 = d1 - srt
        return float(S0*np.exp(-d*T)*_N(d1) - K*np.exp(-r*T)*_N(d2))
    intrinsic = max(S0 - K, 0.0)
    p = min(max(price, intrinsic), S0)
    lo, hi = 1e-8, 5.0
    f_lo = _bs_price(S0, K, T, r, d, lo) - p
    f_hi = _bs_price(S0, K, T, r, d, hi) - p
    if f_lo*f_hi <= 0:
        for _ in range(maxit):
            mid = 0.5*(lo+hi)
            f_mid = _bs_price(S0, K, T, r, d, mid) - p
            if abs(f_mid) < tol or (hi-lo) < 1e-8: return max(mid, 1e-8)
            if f_lo*f_mid <= 0: hi, f_hi = mid, f_mid
            else:               lo, f_lo = mid, f_mid
    sigma = 0.4
    for _ in range(maxit):
        diff = _bs_price(S0, K, T, r, d, sigma) - p
        if abs(diff) < tol: return max(sigma, 1e-8)
        h = 1e-4
        up = _bs_price(S0, K, T, r, d, sigma+h)
        dn = _bs_price(S0, K, T, r, d, sigma-h)
        vega = (up - dn)/(2*h)
        if vega <= 1e-10: break
        sigma = max(1e-8, sigma - diff/vega)
    return max(sigma, 1e-8)
    
def monotone_decreasing_projection(q):
    q = np.clip(np.asarray(q,float), 0.0, 1.0)
    out = np.empty_like(q); m = 1.0
    for i, val in enumerate(q):
        m = min(m, val); out[i] = m
    return out

def isotonic_increasing(y):
    y = np.asarray(y, float).copy(); n = len(y)
    lvl = y.copy(); w = np.ones(n); i = 0
    while i < n-1:
        if lvl[i] > lvl[i+1]:
            tot = lvl[i]*w[i] + lvl[i+1]*w[i+1]
            w[i+1] += w[i]; lvl[i+1] = tot / w[i+1]
            j = i
            while j > 0 and lvl[j-1] > lvl[j]:
                tot = lvl[j-1]*w[j-1] + lvl[j]*w[j]
                w[j] += w[j-1]; lvl[j] = tot / w[j]; j -= 1
            lvl = np.delete(lvl, i); w = np.delete(w, i); n -= 1; i = max(j, 0)
        else:
            i += 1
    return np.repeat(lvl, w.astype(int))

def convex_decreasing_fit_on_band(K, C):
    K = np.asarray(K, float); C = np.asarray(C, float)
    idx = np.argsort(K); K, C = K[idx], C[idx]
    dK = np.diff(K); s = np.diff(C)/dK
    s_iso = isotonic_increasing(s)
    s_iso = np.minimum(s_iso, 0.0)
    cum = np.concatenate(([0.0], np.cumsum(s_iso*dK)))
    y1  = np.mean(C - cum)
    C_fit = np.maximum(y1 + cum, 0.0)
    dC = np.empty_like(C_fit)
    dC[0]  = (C_fit[1]-C_fit[0])/(K[1]-K[0])
    dC[-1] = (C_fit[-1]-C_fit[-2])/(K[-1]-K[-2])
    dC[1:-1] = (C_fit[2:]-C_fit[:-2])/(K[2:]-K[:-2])
    return K, C_fit, dC

### delta-band estimator ###
def delta_band_from_calls(K, C, S0, T, r=0.0, d=0.0,
                          band=(0.45,0.55), left_mode="linear", left_alpha=None):
    K = np.asarray(K, float); C = np.asarray(C, float)
    Ctilde = C * np.exp(r*T); F0T = S0 * np.exp((r - d)*T)
    ivs  = [implied_vol_call(c, S0, k, T, r, d) for k,c in zip(K,C)]
    dels = [bs_call_delta(S0, k, T, r, d, s)   for k,s in zip(K,ivs)]
    ivs, dels = np.array(ivs), np.array(dels)
    msk = (dels >= band[0]) & (dels <= band[1])
    if msk.sum() < 3: raise ValueError("Not enough quotes in chosen delta band")
    Kb, Cb = K[msk], Ctilde[msk]
    K_mid, C_fit_mid, dC_mid = convex_decreasing_fit_on_band(Kb, Cb)
    q_mid = monotone_decreasing_projection(-dC_mid)
    K0, q0 = K_mid[0], float(np.clip(q_mid[0], 0.0, 1.0))
    if left_alpha is not None:
        gamma = left_alpha/(1.0-left_alpha) if left_alpha < 1 else 1e6
        K_left = np.linspace(0.0, K0, 256)
        q_left = 1.0 - (1.0-q0)*(K_left/K0)**gamma
        E_left = float(np.trapz(q_left, K_left))
    else:
        if left_mode == "min":
            K_left = np.linspace(0.0, K0, 64); q_left = np.full_like(K_left, q0); E_left = K0*q0
        elif left_mode == "max":
            K_left = np.linspace(0.0, K0, 64); q_left = np.ones_like(K_left);     E_left = K0
        else:
            K_left = np.linspace(0.0, K0, 64); q_left = np.linspace(1.0, q0, 64); E_left = 0.5*(1+q0)*K0

    KN, CN = K_mid[-1], float(C_fit_mid[-1])
    qN = max(min(q_mid[-1], 1.0), 1e-8)
    lam = qN / max(CN, 1e-12)
    K_right = np.linspace(KN, KN + 20.0/lam, 256)
    q_right = qN * np.exp(-lam*(K_right-KN))
    E_mid   = float(np.trapz(q_mid, K_mid))
    E_right = float(CN)  # exact integral under our match
    E_hat   = float(E_left + E_mid + E_right)
    m_hat   = float(F0T - E_hat)

    K_all  = np.concatenate([K_left[:-1], K_mid, K_right])
    q_all  = np.concatenate([q_left[:-1], q_mid, q_right])
    idx = np.argsort(K_all); K_all, q_all = K_all[idx], q_all[idx]
    tail = np.zeros_like(K_all); acc=0.0
    for i in range(len(K_all)-2, -1, -1):
        dK = K_all[i+1] - K_all[i]
        acc += 0.5*(q_all[i] + q_all[i+1])*dK
        tail[i] = acc
    Cstar = tail
    if K_all[0] > 0: K_all = np.insert(K_all, 0, 0.0); Cstar = np.insert(Cstar, 0, E_hat)
    Cobs = Cstar + m_hat
    return {"F0T":F0T, "E_hat":E_hat, "m_hat":m_hat, "K":K, "C":C,
            "curves":{"K_all":K_all, "Cstar":Cstar, "Cobs":Cobs}}

def plot_call_and_tail_delta_band(est, show="fundamental", title="Delta-band reconstruction"):
    K, C = est["K"], est["C"]
    K_all  = est["curves"]["K_all"]
    Cstar  = est["curves"]["Cstar"]        
    m_hat  = est["m_hat"]
    
    if show == "observed":
        yline = Cstar + m_hat             
        label = r"reconstructed $C(K,T)=C^*(K,T)+\hat{m}$"
    else:  # "fundamental"
        yline = Cstar
        label = r"reconstructed fundamental $C^*(K,T)$"
        
    ### focus on the useful range ###
    j = np.argmax(Cstar < 1e-4) if np.any(Cstar < 1e-4) else len(Cstar)-1
    xmax = K_all[max(10, j)]
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,3.6))
    plt.scatter(K, C, s=14, label="calls (input)", color = "teal")
    plt.plot(K_all, yline, label=label, color = "peru")
    plt.axhline(m_hat, ls="--", label=r"$\hat{m}(T)=$"f"{m_hat:.2f}", color = "black")
    plt.xlim(0, xmax)
    plt.xlabel("Strike K"); plt.ylabel("Call price")
    plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()