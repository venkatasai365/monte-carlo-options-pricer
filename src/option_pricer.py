import numpy as np
from gbm_simulator import simulate_gbm

def price_european(S0, K, r, sigma, T, steps, n_sim, option_type='call'):
    paths = simulate_gbm(S0, r, sigma, T, steps, n_sim)
    ST = paths[-1]
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    std_error = np.std(payoff) / np.sqrt(n_sim)
    return price, std_error

def price_asian(S0, K, r, sigma, T, steps, n_sim, option_type='call'):
    """Asian option: payoff based on average price over life of option"""
    paths = simulate_gbm(S0, r, sigma, T, steps, n_sim)
    avg_price = np.mean(paths[1:], axis=0)  # exclude S0
    if option_type == 'call':
        payoff = np.maximum(avg_price - K, 0)
    else:
        payoff = np.maximum(K - avg_price, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    std_error = np.std(payoff) / np.sqrt(n_sim)
    return price, std_error

def price_barrier(S0, K, B, r, sigma, T, steps, n_sim, barrier_type='down-and-out'):
    """
    Barrier option: knocked out/in when price crosses barrier B
    barrier_type: 'down-and-out', 'down-and-in', 'up-and-out', 'up-and-in'
    """
    paths = simulate_gbm(S0, r, sigma, T, steps, n_sim)
    ST = paths[-1]
    if barrier_type == 'down-and-out':
        barrier_hit = np.any(paths <= B, axis=0)
        payoff = np.where(barrier_hit, 0, np.maximum(ST - K, 0))
    elif barrier_type == 'down-and-in':
        barrier_hit = np.any(paths <= B, axis=0)
        payoff = np.where(barrier_hit, np.maximum(ST - K, 0), 0)
    elif barrier_type == 'up-and-out':
        barrier_hit = np.any(paths >= B, axis=0)
        payoff = np.where(barrier_hit, 0, np.maximum(ST - K, 0))
    elif barrier_type == 'up-and-in':
        barrier_hit = np.any(paths >= B, axis=0)
        payoff = np.where(barrier_hit, np.maximum(ST - K, 0), 0)
    price = np.exp(-r * T) * np.mean(payoff)
    std_error = np.std(payoff) / np.sqrt(n_sim)
    return price, std_error
