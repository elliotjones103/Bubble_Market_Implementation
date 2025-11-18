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

def experiment_rejection_rate(sim_fn, params_list, T, S0=100.0,
                              Npaths=3000, Msteps=400, alpha=0.05,
                              B_boot=200, R=6, seed=1):
    rng = np.random.default_rng(seed)
    rates = []
    for p in params_list:
        rej = 0
        for r in range(R):
            ST = sim_fn(Npaths, Msteps, T, rng, **p)
            out = one_sided_bubble_test(ST, S0=S0, alpha=alpha, B=B_boot, seed=rng.integers(1e9))
            rej += int(out["reject_H0"])
        rates.append(rej / R)
    return np.array(rates)