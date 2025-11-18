import numpy as np

def bootstrap_mean_ci(x, B=800, alpha=0.05, seed=7):
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        means[b] = x[idx].mean()
    mu_hat = x.mean()
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return mu_hat, lo, hi, means

def one_sided_bubble_test(ST, S0=100.0, alpha=0.05, B=800, seed=7):
    rng = np.random.default_rng(seed)
    n = len(ST)
    means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        means[b] = ST[idx].mean()

    mu_hat = ST.mean()
    U = np.quantile(means, 1 - alpha)  
    m_lower = S0 - U                    
    reject = (m_lower > 0)
    pval = (means >= S0).mean()        

    # CI
    lo2, hi2 = np.quantile(means, [alpha/2, 1 - alpha/2])
    return {
        "E[ST]_hat": mu_hat,
        "m_hat": S0 - mu_hat,
        "m_lower_95": m_lower,          
        "reject_H0": bool(reject),
        "p_value": pval,
        "E[ST]_CI_two_sided": (lo2, hi2),   
        "m_CI_two_sided": (S0 - hi2, S0 - lo2)
    }


def isotonic_increasing(y):
    # classic PAVA (unit weights)
    y = np.asarray(y, float).copy()
    n = len(y)
    lvl = y.copy()
    w = np.ones(n)
    i = 0
    while i < n-1:
        if lvl[i] > lvl[i+1]:
            tot = lvl[i]*w[i] + lvl[i+1]*w[i+1]
            w[i+1] += w[i]
            lvl[i+1] = tot / w[i+1]
            # collapse block
            j = i
            while j > 0 and lvl[j-1] > lvl[j]:
                tot = lvl[j-1]*w[j-1] + lvl[j]*w[j]
                w[j] += w[j-1]
                lvl[j] = tot / w[j]
                j -= 1
            # shift left
            lvl = np.delete(lvl, i)
            w   = np.delete(w, i)
            n -= 1
            i = max(j, 0)
        else:
            i += 1
    return np.repeat(lvl, w.astype(int))

def fit_call_curve_monotone_convex(K, C):
    K = np.asarray(K, float)
    C = np.asarray(C, float)
    idx = np.argsort(K)
    K, C = K[idx], C[idx]
    dK = np.diff(K)
    # initial slopes between points
    s = np.diff(C) / dK
    s_iso = isotonic_increasing(s)
    s_iso = np.minimum(s_iso, 0.0)
    for i in range(len(s_iso)-2, -1, -1):
        if s_iso[i] > s_iso[i+1]:
            s_iso[i] = s_iso[i+1]
    cum = np.concatenate(([0.0], np.cumsum(s_iso * dK)))
    y1 = np.mean(C - cum)
    C_fit = y1 + cum
    # non-negativity (tiny floor), preserve shape
    C_fit = np.maximum(C_fit, 0.0)
    return K, C_fit

def call_prices_from_ST(ST, K_grid):
    ST = np.asarray(ST, float)
    K = np.asarray(K_grid, float)
    C = np.maximum(ST[:, None] - K[None, :], 0.0).mean(axis=0)
    return C

def m_from_call_tail(K, C_fit, tail_frac=0.2):
    n = len(K)
    m = max(1, int(np.ceil(tail_frac * n)))
    return float(C_fit[-m:].mean())

def sim_bs_paths(N, M, T, rng, sigma=0.20, S0=100.0):
    dt = T / M
    Z = rng.standard_normal((N, M))
    logS = np.full(N, np.log(S0))
    drift = -0.5 * sigma**2 * dt
    diff  = sigma * np.sqrt(dt)
    for k in range(M):
        logS += drift + diff * Z[:, k]
    return np.exp(logS)

def sim_cev_paths(N, M, T, rng, sigma=0.16, beta=1.5, S0=100.0):
    assert beta > 1.0
    dt = T / M
    sdt = np.sqrt(dt)
    Z = rng.standard_normal((N, M))
    a = (beta-1.0)*sigma
    b = 0.5*(beta-1.0)*beta*sigma**2
    Y = np.full(N, S0**(-(beta-1.0)))
    eps = 1e-12
    for k in range(M):
        dW = sdt * Z[:, k]
        Y = Y - a*dW + b*dt/np.maximum(Y, eps)
        Y = np.maximum(Y, eps)
    S = np.power(Y, -1.0/(beta-1.0))
    return np.minimum(S, 1e12)

def sim_sabr_beta1_paths(N, M, T, rng, sigma0=0.25, nu=1.0, rho=0.7, S0=100.0):
    dt = T / M
    sdt = np.sqrt(dt)
    Z2 = rng.standard_normal((N, M))
    sig = np.full(N, sigma0)
    A   = np.zeros(N)
    for k in range(M):
        sig_next = sig * np.exp(-0.5*nu*nu*dt + nu*sdt*Z2[:, k])
        A += 0.5 * (sig*sig + sig_next*sig_next) * dt
        sig = sig_next
    sigma_T = sig
    B = (sigma_T - sigma0)/nu
    Zperp = rng.standard_normal(N)
    J = np.sqrt(np.maximum(A, 0.0)) * Zperp
    logS = rho * B + np.sqrt(max(0.0, 1.0 - rho*rho)) * J - 0.5 * A
    return S0 * np.exp(logS)