import numpy as np
from gbm_simulator import simulate_gbm

def calculate_var_cvar(S0, r, sigma, T, steps, n_sim, confidence=0.95):
    """
    Monte Carlo VaR and CVaR for a single asset position
    """
    paths = simulate_gbm(S0, r, sigma, T, steps, n_sim)
    ST = paths[-1]
    returns = (ST - S0) / S0
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar, returns

def stress_test(S0, K, r, sigma, T, steps, n_sim, shocks=[-0.10, -0.20, -0.30, -0.40]):
    """
    Stress test: apply price shocks and reprice options
    shocks: list of percentage drops (e.g., -0.10 = -10%)
    """
    results = {}
    from option_pricer import price_european
    base_price, _ = price_european(S0, K, r, sigma, T, steps, n_sim)
    for shock in shocks:
        S_shocked = S0 * (1 + shock)
        shocked_price, _ = price_european(S_shocked, K, r, sigma, T, steps, n_sim)
        pnl_change = shocked_price - base_price
        results[f"{int(shock*100)}% shock"] = {
            "Shocked Spot": round(S_shocked, 2),
            "Option Price": round(shocked_price, 4),
            "P&L Change": round(pnl_change, 4),
            "P&L Change (%)": round((pnl_change / base_price) * 100, 2)
        }
    return base_price, results
