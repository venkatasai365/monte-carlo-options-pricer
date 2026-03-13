import numpy as np

def simulate_gbm(S0, r, sigma, T, steps, n_simulations, seed=42):
    """
    Simulate Geometric Brownian Motion price paths.
    S0: initial stock price
    r: risk-free rate
    sigma: volatility
    T: time to maturity (years)
    steps: number of time steps
    n_simulations: number of paths
    """
    np.random.seed(seed)
    dt = T / steps
    # GBM formula: S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    Z = np.random.standard_normal((steps, n_simulations))
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    daily_returns = np.exp(drift + diffusion)
    price_paths = np.zeros((steps + 1, n_simulations))
    price_paths[0] = S0
    for t in range(1, steps + 1):
        price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]
    return price_paths
