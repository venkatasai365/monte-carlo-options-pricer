# 📈 Monte Carlo Options Pricer & Stress Tester

A quantitative finance project implementing Monte Carlo simulation for pricing
European, Asian, and Barrier options with VaR, CVaR, and stress testing.

## Features
- GBM price path simulation (50,000 paths)
- European, Asian, Barrier (Down-Out/In, Up-Out/In) option pricing
- Value at Risk (VaR) at 95% and 99% confidence levels
- Conditional VaR (CVaR / Expected Shortfall)
- Portfolio stress testing under -10% to -40% price shocks

## Tech Stack
Python | NumPy | Pandas | Matplotlib | Plotly | SciPy

## Results Summary
| Metric | Value |
|--------|-------|
| European Call Price | ~$8.02 |
| 95% VaR | ~-26% |
| 95% CVaR | ~-33% |

## How to Run
pip install -r requirements.txt
jupyter notebook notebooks/monte_carlo_analysis.ipynb

## Author
Venkatasai Thaduri | NISM Certified | Quant Researcher
