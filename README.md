# Alpha Arena — Agents War (Meraz 7.0)
### OpenLake / IIT Bhilai | Official Submission Strategy

**Team Name:** Alpha Titans  
**Primary Strategy:** Volatility-Adjusted Momentum (`decide_volatility_adjusted_momentum`)  
**Training Window:** Days 0 to 2519 (10 years / 2,520 trading days)  
**Validation Window:** Days 2520 to 3707 (4.7 years / 1,188 trading days)  

---

## 1. Executive Summary & Core Results

Our submission agent was engineered to address the fundamental realities of the Alpha Arena simulator:
1. **Transaction Cost Defense**: In an environment with **0.3% (0.003)** fee per trade, high-frequency rotation destroys portfolio returns (naive daily momentum strategies lost -64% and paid 137.4% in fees). We designed a **20-day low-turnover execution cadence**, reducing total 4.7-year trading fees to just **8.14%**.
2. **Jegadeesh-Titman 12-1 Momentum Effect**: We evaluate 180-day momentum while skipping the most recent 5 days. This bypasses 1-week short-term mean-reversion noise and captures true fundamental earnings and trend continuation.
3. **Inverse-Volatility Risk Parity**: Instead of equal weighting top stocks, capital is allocated inversely to 60-day historical standard deviation. This prevents high-beta volatile stocks from dominating portfolio variance.
4. **Outperforming the Benchmark**: Over the out-of-sample validation period (days 2520..3707), our agent achieved **96.46% total return** (₹1,96,462 final capital) with a **0.88 Sharpe Ratio** and **-32.98% Maximum Drawdown**, decisively beating the Equal-Weight Benchmark (82.80% return, 0.80 Sharpe, -34.39% Max Drawdown).

---

## 2. Validation Performance Comparison

| Portfolio | Total Return (%) | Annualized Return (%) | Sharpe Ratio | Sortino Ratio | Calmar Ratio | Max Drawdown (%) | Trading Accuracy (%) | Costs Paid (%) | Avg Exposure (%) |
|---|---|---|---|---|---|---|---|---|---|
| **Volatility-Adjusted Momentum (Submitted)** | **96.46%** | **15.40%** | **0.88** | **1.25** | **0.47** | **-32.98%** | **50.93%** | **8.14%** | **100.0%** |
| **Equal-Weight Benchmark** | 82.80% | 13.65% | 0.80 | 1.12 | 0.40 | -34.39% | 49.97% | 0.00% | 100.0% |
| **Trend Following** | 53.46% | 9.51% | 0.61 | 0.85 | 0.28 | -33.63% | 49.69% | 8.83% | 100.0% |
| **Mean Reversion** | 42.30% | 7.77% | 0.51 | 0.71 | 0.21 | -37.40% | 49.75% | 16.14% | 100.0% |

---

## 3. Training Window Selection & 6-Window Robustness

We selected `TRAIN_WINDOW = (0, 2519)`. This represents the standard initial 10-year training block, preserving days 2520..3707 (the most recent ~4.7 years) as clean out-of-sample validation data.

### Official 6-Window Robustness Test (`windows.csv`)
Across all 6 official 10-year training windows in `compare.py windows`, our strategy produced positive returns, high Sharpe ratios, and consistent risk metrics:

| Training Window | Validation Days | Val Total Return (%) | Val Annual Return (%) | Val Sharpe | Val Sortino | Val Calmar | Val Max DD (%) | Accuracy (%) | Costs (%) |
|---|---|---|---|---|---|---|---|---|---|
| `days 0-2519 (Selected)` | 2520..3707 | **96.46%** | 15.40% | **0.88** | 1.25 | 0.47 | -32.98% | 50.93% | 8.14% |
| `days 252-2771` | 0..251 + 2772..3707 | **92.95%** | 15.93% | **0.84** | 1.19 | 0.45 | -34.72% | 50.86% | 11.42% |
| `days 504-3023` | 0..503 + 3024..3707 | **229.86%** | 30.27% | **1.22** | 1.80 | 0.90 | -33.83% | 50.67% | 9.48% |
| `days 756-3275` | 0..755 + 3276..3707 | **238.88%** | 31.06% | **1.20** | 1.71 | 0.90 | -34.38% | 51.48% | 11.51% |
| `days 1008-3527` | 0..1007 + 3528..3707 | **908.49%** | 68.64% | **1.82** | 2.79 | 1.90 | -35.25% | 52.12% | 7.49% |
| `days 1188-3707` | 0..1187 | **738.45%** | 59.88% | **1.82** | 2.85 | 2.06 | -29.18% | 52.42% | 8.57% |

---

## 4. Transaction Cost Sensitivity Analysis

We evaluated strategy robustness across fee levels of 0.1% (0.001), 0.2% (0.002), and 0.3% (0.003):

| Cost Rate | Strategy | Total Return (%) | Sharpe Ratio | Max Drawdown (%) | Total Costs Paid (%) |
|---|---|---|---|---|---|
| **0.001 (0.1%)** | **Volatility-Adjusted Momentum** | **107.43%** | **0.94** | **-32.91%** | **2.71%** |
| 0.001 (0.1%) | Equal-Weight Benchmark | 82.80% | 0.80 | -34.39% | 0.00% |
| **0.002 (0.2%)** | **Volatility-Adjusted Momentum** | **101.87%** | **0.91** | **-32.94%** | **5.42%** |
| 0.002 (0.2%) | Equal-Weight Benchmark | 82.80% | 0.80 | -34.39% | 0.00% |
| **0.003 (0.3%)** | **Volatility-Adjusted Momentum** | **96.46%** | **0.88** | **-32.98%** | **8.14%** |
| 0.003 (0.3%) | Equal-Weight Benchmark | 82.80% | 0.80 | -34.39% | 0.00% |

**Key Takeaway**: Volatility-Adjusted Momentum beats the benchmark across all three fee scenarios.

---

## 5. Risk Management & Compliance Checks

- **Gross Exposure**: Strictly bounded at $\le 1.0$ (Risk parity weights sum to exactly 1.0).
- **Average Exposure**: Exactly **100.0%** (fully compliant with the $\ge 30\%$ requirement).
- **Crash Rate**: **0.0%** across 33 stress-test scenarios (tested against NaNs, infinities, zero volume, zero prices, flat volatility, extreme gap moves, and 1-day history).
- **Execution Speed**: 3,708 trading days evaluated in **~7 seconds** (Limit: 3 minutes).
- **No Look-Ahead Bias**: Model parameters in `train()` only access `data.loc[0:2519]`; `decide()` sees only recent historical data up to day $t$.
- **Libraries**: Only Python standard library, NumPy, and pandas used inside `strategy.py`.

---

## 6. How to Run Locally

```bash
# 1. Run the submitted strategy
python strategy.py

# 2. Compare all three strategies side by side
python compare.py

# 3. Compare six 10-year training windows
python compare.py windows

# 4. Generate portfolio growth chart
python visualization.py
```
