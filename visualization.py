"""
visualization.py — Portfolio Growth Graph Deliverable
ALPHA ARENA — AGENTS WAR (MERAZ 7.0)

Generates cumulative portfolio equity curve of the submitted strategy:
- X-axis: Day
- Y-axis: Portfolio value in Rs starting from Rs 1,00,000
- Benchmark line: Equal-weight benchmark (1/12 in every stock)
- Saves: portfolio_growth.png
"""

import matplotlib.pyplot as plt
from pathlib import Path
import engine
import strategy

HERE = Path(__file__).resolve().parent

def main():
    data = engine.load(HERE / "data" / "train.csv")
    window = engine.train_agent(strategy, data)
    
    # Backtest strategy and benchmark
    res_strat = engine.backtest(strategy.decide, data, verbose=False)
    res_bm = engine.backtest(engine.equal_weight, data, verbose=False)
    
    # We display the validation days (outside training window)
    val_start = window[1] + 1
    log_strat = res_strat["log"].loc[val_start:]
    log_bm = res_bm["log"].loc[val_start:]
    
    # Re-base to Rs 1,00,000
    eq_strat = engine.STARTING_CAPITAL * (1.0 + log_strat["return"]).cumprod()
    eq_strat.loc[val_start - 1] = engine.STARTING_CAPITAL
    eq_strat = eq_strat.sort_index()
    
    eq_bm = engine.STARTING_CAPITAL * (1.0 + log_bm["return"]).cumprod()
    eq_bm.loc[val_start - 1] = engine.STARTING_CAPITAL
    eq_bm = eq_bm.sort_index()
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    
    ax.plot(eq_strat.index, eq_strat.values, label=f"Alpha Titans (Volatility-Adjusted Momentum) — Final: ₹{eq_strat.values[-1]:,.0f}", color="#1f77b4", linewidth=2.5)
    ax.plot(eq_bm.index, eq_bm.values, label=f"Equal-Weight Benchmark — Final: ₹{eq_bm.values[-1]:,.0f}", color="#7f7f7f", linestyle="--", linewidth=2.0)
    
    ax.set_title("Alpha Arena (Meraz 7.0) — Portfolio Growth vs Benchmark\nOut-of-Sample Validation Period (Days 2520 to 3707)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Trading Day", fontsize=11, fontweight="bold")
    ax.set_ylabel("Portfolio Value (₹)", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"₹{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=10)
    
    out_path = HERE / "portfolio_growth.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Graph saved to {out_path}")

if __name__ == "__main__":
    main()
