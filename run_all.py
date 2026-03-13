# ============================================================
# SECTION 1: IMPORTS & PARAMETERS
# What: Load libraries and define our financial parameters
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # saves charts to file instead of popup
import matplotlib.pyplot as plt
import os

os.makedirs(r'C:\mcop\outputs', exist_ok=True)   # create outputs folder

# --- Financial Parameters (you can change these!) ---
S0    = 100     # Current stock price = $100
K     = 105     # Strike price = $105 (slightly out of the money)
B     = 85      # Barrier level = $85 (knocked out if price falls here)
r     = 0.05    # Risk-free rate = 5% per year
sigma = 0.20    # Volatility = 20% (how much stock moves)
T     = 1.0     # Time to expiry = 1 year
steps = 252     # 252 trading days in a year
n_sim = 50000   # Run 50,000 simulations

print("✅ SECTION 1 DONE: Parameters loaded")
print(f"   Stock=${S0} | Strike=${K} | Vol={sigma*100}% | Time={T}yr")


# ============================================================
# SECTION 2: GBM PRICE PATH SIMULATOR
# What: Simulates how a stock price moves over time
# Formula: S(t) = S(t-1) * exp((r - 0.5*sigma²)*dt + sigma*sqrt(dt)*Z)
# Z = random shock (from normal distribution)
# ============================================================
def simulate_gbm(S0, r, sigma, T, steps, n_simulations=50000, seed=42, n_sim=None):
    if n_sim is not None:
        n_simulations = n_sim
    np.random.seed(seed)               # seed = same random numbers every run
    dt = T / steps                     # dt = 1/252 = one trading day
    Z  = np.random.standard_normal((steps, n_simulations))  # random shocks
    price_paths = np.zeros((steps + 1, n_simulations))
    price_paths[0] = S0                # all paths start at S0=$100
    for t in range(1, steps + 1):
        price_paths[t] = price_paths[t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1]
        )
    return price_paths                 # shape: (253, 50000)

# --- Run & Plot ---
paths = simulate_gbm(S0, r, sigma, T, steps, n_sim=200)
print(f"\n✅ SECTION 2 DONE: Simulated price paths")
print(f"   Path shape: {paths.shape}  (253 days × 200 paths shown)")
print(f"   Final prices range: ${paths[-1].min():.2f} to ${paths[-1].max():.2f}")

plt.figure(figsize=(13, 5))
plt.plot(paths[:, :50], alpha=0.25, linewidth=0.8, color='steelblue')
plt.axhline(y=K, color='red',    linestyle='--', linewidth=2, label=f'Strike K=${K}')
plt.axhline(y=B, color='orange', linestyle='--', linewidth=2, label=f'Barrier B=${B}')
plt.title('Monte Carlo GBM — 50 Simulated Stock Price Paths over 1 Year', fontsize=13, fontweight='bold')
plt.xlabel('Trading Days (0 = Today, 252 = 1 Year Later)')
plt.ylabel('Stock Price ($)')
plt.legend(); plt.tight_layout()
plt.savefig(r'C:\mcop\outputs\gbm_paths.png', dpi=150)
plt.close()
print("   Chart saved: gbm_paths.png")

# ============================================================
# SECTION 3: OPTION PRICING
# European: payoff at expiry only (standard option)
# Asian:    payoff based on AVERAGE price (cheaper, path-dependent)
# Barrier:  knocked out if price hits barrier (even cheaper)
# ============================================================
def price_european(S0, K, r, sigma, T, steps, n_sim, option_type='call'):
    paths  = simulate_gbm(S0, r, sigma, T, steps, n_simulations=n_sim)
    ST     = paths[-1]                 # only final price matters
    payoff = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    return np.exp(-r*T) * np.mean(payoff), np.std(payoff)/np.sqrt(n_sim)

def price_asian(S0, K, r, sigma, T, steps, n_sim, option_type='call'):
    paths     = simulate_gbm(S0, r, sigma, T, steps, n_simulations=n_sim)
    avg_price = np.mean(paths[1:], axis=0)   # average price over whole year
    payoff    = np.maximum(avg_price - K, 0) if option_type=='call' else np.maximum(K - avg_price, 0)
    return np.exp(-r*T) * np.mean(payoff), np.std(payoff)/np.sqrt(n_sim)

def price_barrier(S0, K, B, r, sigma, T, steps, n_sim, barrier_type='down-and-out'):
    paths = simulate_gbm(S0, r, sigma, T, steps, n_simulations=n_sim)
    ST    = paths[-1]
    if   barrier_type == 'down-and-out': hit = np.any(paths <= B, axis=0); payoff = np.where(hit, 0, np.maximum(ST-K, 0))
    elif barrier_type == 'down-and-in':  hit = np.any(paths <= B, axis=0); payoff = np.where(hit, np.maximum(ST-K,0), 0)
    elif barrier_type == 'up-and-out':   hit = np.any(paths >= B, axis=0); payoff = np.where(hit, 0, np.maximum(ST-K, 0))
    else:                                hit = np.any(paths >= B, axis=0); payoff = np.where(hit, np.maximum(ST-K,0), 0)
    return np.exp(-r*T) * np.mean(payoff), np.std(payoff)/np.sqrt(n_sim)

# --- Run & Print ---
eu_call, se = price_european(S0,K,r,sigma,T,steps,n_sim,'call')
eu_put,  _  = price_european(S0,K,r,sigma,T,steps,n_sim,'put')
asian,   _  = price_asian(S0,K,r,sigma,T,steps,n_sim,'call')
bar_do,  _  = price_barrier(S0,K,B,r,sigma,T,steps,n_sim,'down-and-out')
bar_di,  _  = price_barrier(S0,K,B,r,sigma,T,steps,n_sim,'down-and-in')

print(f"\n✅ SECTION 3 DONE: Option Prices")
print(f"   European Call  : ${eu_call:.4f}  (standard call option)")
print(f"   European Put   : ${eu_put:.4f}  (standard put option)")
print(f"   Asian Call     : ${asian:.4f}  (cheaper — based on avg price)")
print(f"   Barrier D-Out  : ${bar_do:.4f}  (dies if price hits ${B})")
print(f"   Barrier D-In   : ${bar_di:.4f}  (lives only if price hits ${B})")
print(f"   Std Error (MC) : ${se:.6f}  (simulation accuracy)")

# ============================================================
# SECTION 4: VALUE AT RISK (VaR) & CONDITIONAL VaR (CVaR)
# VaR 95%  = "We lose MORE than X% only 5% of the time"
# CVaR 95% = "When we DO breach VaR, average loss is X%"
# CVaR is more important — used in Basel III banking regulation
# ============================================================
def calculate_var_cvar(S0, r, sigma, T, steps, n_sim, confidence=0.95):
    paths   = simulate_gbm(S0, r, sigma, T, steps, n_simulations=n_sim)
    returns = (paths[-1] - S0) / S0    # percentage returns
    var     = np.percentile(returns, (1-confidence)*100)
    cvar    = returns[returns <= var].mean()
    return var, cvar, returns

var95, cvar95, returns = calculate_var_cvar(S0, r, sigma, T, steps, n_sim, 0.95)
var99, cvar99, _       = calculate_var_cvar(S0, r, sigma, T, steps, n_sim, 0.99)

print(f"\n✅ SECTION 4 DONE: Risk Metrics")
print(f"   95% VaR  = {var95:.2%}  → 95% of time losses stay above this")
print(f"   95% CVaR = {cvar95:.2%}  → worst 5% average loss")
print(f"   99% VaR  = {var99:.2%}  → 99% of time losses stay above this")
print(f"   99% CVaR = {cvar99:.2%}  → worst 1% average loss")

plt.figure(figsize=(12, 5))
plt.hist(returns, bins=120, alpha=0.7, color='steelblue', edgecolor='white')
plt.axvline(var95,  color='orange', linestyle='--', lw=2, label=f'VaR  95%: {var95:.2%}')
plt.axvline(cvar95, color='red',    linestyle='--', lw=2, label=f'CVaR 95%: {cvar95:.2%}')
plt.axvline(var99,  color='purple', linestyle='--', lw=2, label=f'VaR  99%: {var99:.2%}')
plt.axvline(cvar99, color='darkred',linestyle=':',  lw=2, label=f'CVaR 99%: {cvar99:.2%}')
plt.title('Return Distribution — VaR & CVaR (50,000 Simulations)', fontsize=13, fontweight='bold')
plt.xlabel('1-Year Portfolio Return'); plt.ylabel('Number of Simulations')
plt.legend(); plt.tight_layout()
plt.savefig(r'C:\mcop\outputs\var_cvar.png', dpi=150)
plt.close()
print("   Chart saved: var_cvar.png")

# ============================================================
# SECTION 5: STRESS TEST
# What: Simulates market crash scenarios
# Asks: "If stock drops 10%/20%/30%/40%, how much do we lose?"
# Used by risk managers daily in hedge funds & banks
# ============================================================
def stress_test(S0, K, r, sigma, T, steps, n_sim):
    base, _ = price_european(S0, K, r, sigma, T, steps, n_sim)
    results = {}
    for shock in [-0.10, -0.20, -0.30, -0.40]:
        sp, _ = price_european(S0*(1+shock), K, r, sigma, T, steps, n_sim)
        pnl   = sp - base
        results[f"{int(shock*100)}% shock"] = {
            "Shocked Spot":   round(S0*(1+shock), 2),
            "Option Price":   round(sp, 4),
            "P&L Change":     round(pnl, 4),
            "P&L Change (%)": round((pnl/base)*100, 2)
        }
    return base, results

base, stress = stress_test(S0, K, r, sigma, T, steps, n_sim)
stress_df    = pd.DataFrame(stress).T.reset_index().rename(columns={'index':'Scenario'})

print(f"\n✅ SECTION 5 DONE: Stress Test")
print(f"   Base Option Price = ${base:.4f}")
print(stress_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['#e74c3c','#c0392b','#922b21','#641e16']
axes[0].bar(stress_df['Scenario'], stress_df['P&L Change (%)'], color=colors, width=0.5)
axes[0].set_title('P&L Change (%) Under Market Shocks', fontweight='bold')
axes[0].set_ylabel('P&L Change (%)'); axes[0].set_ylim(-110, 5)
for i,v in enumerate(stress_df['P&L Change (%)']):
    axes[0].text(i, v-4, f'{v:.1f}%', ha='center', color='white', fontweight='bold')

axes[1].bar(stress_df['Scenario'], stress_df['Option Price'],
            color=['#3498db','#2980b9','#1a6fa8','#0e4d78'], width=0.5)
axes[1].axhline(y=base, color='red', linestyle='--', lw=2, label=f'Base=${base:.2f}')
axes[1].set_title('Option Price After Shock', fontweight='bold')
axes[1].set_ylabel('Option Price ($)'); axes[1].legend()
for i,v in enumerate(stress_df['Option Price']):
    axes[1].text(i, v+0.1, f'${v:.2f}', ha='center', fontweight='bold')

fig.suptitle('Monte Carlo Stress Test — Market Crash Scenarios', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig(r'C:\mcop\outputs\stress_test.png', dpi=150)
plt.close()
print("   Chart saved: stress_test.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*55)
print("🎉  PROJECT COMPLETE — MONTE CARLO OPTIONS PRICER")
print("="*55)
print(f"  European Call  : ${eu_call:.4f}")
print(f"  Asian Call     : ${asian:.4f}")
print(f"  Barrier D-Out  : ${bar_do:.4f}")
print(f"  95% VaR        : {var95:.2%}")
print(f"  95% CVaR       : {cvar95:.2%}")
print(f"  -40% Shock Loss: {stress_df['P&L Change (%)'].iloc[-1]:.1f}%")
print("="*55)
print("  Charts saved to: C:\\mcop\\outputs\\")
print("="*55)
