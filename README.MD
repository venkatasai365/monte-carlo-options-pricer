# 📈 Monte Carlo Options Pricer & Stress Tester

A quantitative finance project implementing **Monte Carlo simulation** with
**Geometric Brownian Motion (GBM)** to price European, Asian, and Barrier options,
with full VaR, CVaR, and stress testing.

---

## 📊 Results

### Option Prices (50,000 Simulations)
| Option Type | MC Price |
|-------------|----------|
| European Call | $7.99 |
| European Put | $7.93 |
| Asian Call | $3.50 |
| Barrier Down-Out | $7.74 |
| Barrier Down-In | $0.25 |

### Risk Metrics
| Metric | Value |
|--------|-------|
| 95% VaR | -26.08% |
| 95% CVaR | -31.68% |
| 99% VaR | -35.28% |
| 99% CVaR | -39.21% |

### Stress Test (Market Shocks)
| Scenario | Option Price | P&L Change |
|----------|-------------|------------|
| -10% shock | $3.59 | -55.08% |
| -20% shock | $1.18 | -85.22% |
| -30% shock | $0.25 | -96.87% |
| -40% shock | $0.03 | -99.64% |

---

## 📉 Charts
# 📈 Monte Carlo Options Pricer & Stress Tester

A quantitative finance project implementing **Monte Carlo simulation** with
**Geometric Brownian Motion (GBM)** to price European, Asian, and Barrier options,
with full VaR, CVaR, and stress testing.

---

## 📊 Results

### Option Prices (50,000 Simulations)
| Option Type | MC Price |
|-------------|----------|
| European Call | $7.99 |
| European Put | $7.93 |
| Asian Call | $3.50 |
| Barrier Down-Out | $7.74 |
| Barrier Down-In | $0.25 |

### Risk Metrics
| Metric | Value |
|--------|-------|
| 95% VaR | -26.08% |
| 95% CVaR | -31.68% |
| 99% VaR | -35.28% |
| 99% CVaR | -39.21% |

### Stress Test (Market Shocks)
| Scenario | Option Price | P&L Change |
|----------|-------------|------------|
| -10% shock | $3.59 | -55.08% |
| -20% shock | $1.18 | -85.22% |
| -30% shock | $0.25 | -96.87% |
| -40% shock | $0.03 | -99.64% |

---

## 📉 Charts

### GBM Simulated Price Paths
![GBM Paths](outputs/gbm_paths.png)

### VaR & CVaR Return Distribution
![VaR CVaR](outputs/var_cvar.png)

### Stress Test Results
![Stress Test](outputs/stress_test.png)

---

## 🧠 Concepts Covered

- **Geometric Brownian Motion (GBM)** — stochastic price path simulation
- **European Options** — standard call/put pricing via Monte Carlo
- **Asian Options** — path-dependent pricing using average price
- **Barrier Options** — knock-out/knock-in exotic option pricing
- **Value at Risk (VaR)** — 95% and 99% confidence levels
- **Conditional VaR (CVaR)** — Expected Shortfall (Basel III standard)
- **Stress Testing** — P&L impact under -10% to -40% market crashes

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.26-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-red)

---

## 🚀 How to Run

```bash
git clone https://github.com/venkatasai365/monte-carlo-options-pricer.git
cd monte-carlo-options-pricer
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python run_all.py


### GBM Simulated Price Paths
![GBM Paths](outputs/gbm_paths.png)

### VaR & CVaR Return Distribution
![VaR CVaR](outputs/var_cvar.png)

### Stress Test Results
![Stress Test](outputs/stress_test.png)

---

## 🧠 Concepts Covered

- **Geometric Brownian Motion (GBM)** — stochastic price path simulation
- **European Options** — standard call/put pricing via Monte Carlo
- **Asian Options** — path-dependent pricing using average price
- **Barrier Options** — knock-out/knock-in exotic option pricing
- **Value at Risk (VaR)** — 95% and 99% confidence levels
- **Conditional VaR (CVaR)** — Expected Shortfall (Basel III standard)
- **Stress Testing** — P&L impact under -10% to -40% market crashes

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.26-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-red)

---

## 🚀 How to Run

```bash
git clone https://github.com/venkatasai365/monte-carlo-options-pricer.git
cd monte-carlo-options-pricer
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python run_all.py
