"""
ALPHA ARENA — AGENTS WAR (MERAZ 7.0)
Official Submission Agent

Team Name: Alpha Titans
Primary Strategy: Volatility-Adjusted Momentum (decide_volatility_adjusted_momentum)
Benchmark: Equal-Weight Portfolio (1/12 in every stock)

Architecture & Key Quantitative Decisions:
1. Low-Turnover Execution: Rebalances every 20 trading days (~monthly) to avoid
   destructive transaction costs (0.3% per trade).
2. Volatility-Adjusted Momentum: Uses 180-day momentum skipping the last 5 days
   (Jegadeesh-Titman 12-1 effect to avoid short-term mean reversion pullbacks),
   normalized by 60-day realized volatility.
3. Inverse-Volatility Risk Parity: Allocates inversely proportional to volatility across
   the top 5-6 ranked stocks, preventing volatile stocks from dominating portfolio variance.
4. Robust Defensive Fallbacks: Handles zero history, missing data, flat prices,
   and extreme shocks cleanly without crashing or violating exposure rules (maintaining 100% exposure).
"""

import math
import numpy as np
import pandas as pd

TEAM_NAME = "Alpha Titans"
TRAIN_WINDOW = (0, 2519)      # 10 years (2520 trading days), leaves days 2520..3707 for out-of-sample validation
MODEL = {}                    # Learned parameters stored here by train()

# Internal state memory to manage low-turnover rebalancing across consecutive decision calls
_MOM_STATE = {}
_TREND_STATE = {}
_MR_STATE = {}


def train(data):
    """
    Called ONCE with ONLY the historical data within TRAIN_WINDOW.
    Discovers optimal hyperparameters strictly from training data without lookahead bias:
      1. Moving average combinations for Trend Following.
      2. Z-score lookback for Mean Reversion.
      3. Momentum lookback and portfolio basket size (k) for Volatility-Adjusted Momentum.
    """
    close = data["close"]
    stocks = list(close.columns)
    
    # 1. Optimize Trend Following MA lengths
    best_sh_trend, best_fast, best_slow = -np.inf, 30, 150
    for fast in [20, 30, 50]:
        for slow in [100, 150, 200]:
            ma_f = close.rolling(fast).mean()
            ma_s = close.rolling(slow).mean()
            trend = (close > ma_s) & (ma_f > ma_s)
            daily_ret = close.pct_change().shift(-1)
            port = (daily_ret * trend).sum(axis=1) / trend.sum(axis=1).replace(0, np.nan)
            port_clean = port.dropna()
            if len(port_clean) > 250:
                sh = float(port_clean.mean() / (port_clean.std() + 1e-6))
                if sh > best_sh_trend:
                    best_sh_trend, best_fast, best_slow = sh, fast, slow
    MODEL["trend_fast"] = best_fast
    MODEL["trend_slow"] = best_slow
    
    # 2. Optimize Mean Reversion Z-score lookback
    best_sh_mr, best_z = -np.inf, 15
    for z in [10, 15, 20]:
        mean_z = close.rolling(z).mean()
        std_z = close.rolling(z).std().replace(0, np.nan)
        z_score = (close - mean_z) / std_z
        oversold = z_score < -0.5
        daily_ret = close.pct_change().shift(-1)
        port = (daily_ret * oversold).sum(axis=1) / oversold.sum(axis=1).replace(0, np.nan)
        port_clean = port.dropna()
        if len(port_clean) > 250:
            sh = float(port_clean.mean() / (port_clean.std() + 1e-6))
            if sh > best_sh_mr:
                best_sh_mr, best_z = sh, z
    MODEL["mr_z"] = best_z
    
    # 3. Optimize Volatility-Adjusted Momentum parameters
    best_sh_mom, best_lb, best_k = -np.inf, 180, 5
    for lb in [90, 120, 150, 180]:
        for k in [4, 5, 6]:
            mom = close.shift(5) / close.shift(lb + 5) - 1.0
            vol = close.pct_change().rolling(60).std().clip(lower=1e-4)
            score = mom / vol
            ranks = score.rank(axis=1, ascending=False)
            held = ranks <= k
            daily_ret = close.pct_change().shift(-1)
            port = (daily_ret * held).sum(axis=1) / k
            port_clean = port.dropna()
            if len(port_clean) > 250:
                sh = float(port_clean.mean() / (port_clean.std() + 1e-6))
                if sh > best_sh_mom:
                    best_sh_mom, best_lb, best_k = sh, lb, k
    MODEL["mom_lb"] = best_lb
    MODEL["mom_k"] = best_k
    MODEL["trained"] = True


def decide_trend_following(history):
    """
    Strategy 1: Trend Following with Dual Moving Average & Inverse Volatility Weighting.
    - Captures established medium/long-term upward trends.
    - Rebalances every 15 trading days to suppress churn and transaction cost drag.
    - If fewer than 4 stocks are in an uptrend, picks the top 4 relative-strength stocks
      to guarantee robust compliance with the minimum 30% exposure rule.
    """
    close = history["close"].ffill().bfill()
    stocks = list(close.columns)
    today = close.index[-1]
    
    # Reset internal memory on new backtest runs or out-of-order execution
    if "last_day" not in _TREND_STATE or today <= _TREND_STATE["last_day"]:
        _TREND_STATE.clear()
        _TREND_STATE["last_rebal"] = -9999
        _TREND_STATE["weights"] = {s: 1.0 / len(stocks) for s in stocks}
    _TREND_STATE["last_day"] = today
    
    # Low-turnover gate: preserve position on non-rebalance days
    if (today - _TREND_STATE["last_rebal"]) < 15:
        return _TREND_STATE["weights"]
    _TREND_STATE["last_rebal"] = today
    
    fast = MODEL.get("trend_fast", 30)
    slow = MODEL.get("trend_slow", 150)
    
    if len(close) < slow + 5 or (close <= 0).all().all():
        return _TREND_STATE["weights"]
        
    ma_fast = close.iloc[-fast:].mean()
    ma_slow = close.iloc[-slow:].mean()
    vol = close.iloc[-60:].pct_change().std().fillna(0.02).clip(lower=1e-4)
    inv_vol = 1.0 / vol
    
    bullish = (close.iloc[-1] > ma_slow) & (ma_fast > ma_slow)
    picks = list(bullish[bullish].index)
    
    if len(picks) < 4:
        trend_ratio = (close.iloc[-1] / ma_slow.replace(0, np.nan) - 1.0).fillna(0.0)
        picks = list(trend_ratio.nlargest(4).index)
        
    sub_inv = inv_vol[picks]
    total_inv = sub_inv.sum()
    if total_inv <= 0 or not np.isfinite(total_inv):
        w = {s: 1.0 / len(picks) for s in picks}
    else:
        w = (sub_inv / total_inv).to_dict()
        
    _TREND_STATE["weights"] = w
    return _TREND_STATE["weights"]


def decide_mean_reversion(history):
    """
    Strategy 2: Mean Reversion with Trend Filter Protection.
    - Identifies quality stocks experiencing temporary short-term pullbacks (oversold Z-score).
    - Long-term trend filter (120-day MA) prevents buying falling knives in secular downtrends.
    - Rebalances every 15 trading days with Inverse Volatility weighting.
    """
    close = history["close"].ffill().bfill()
    stocks = list(close.columns)
    today = close.index[-1]
    
    if "last_day" not in _MR_STATE or today <= _MR_STATE["last_day"]:
        _MR_STATE.clear()
        _MR_STATE["last_rebal"] = -9999
        _MR_STATE["weights"] = {s: 1.0 / len(stocks) for s in stocks}
    _MR_STATE["last_day"] = today
    
    if (today - _MR_STATE["last_rebal"]) < 15:
        return _MR_STATE["weights"]
    _MR_STATE["last_rebal"] = today
    
    z_win = MODEL.get("mr_z", 15)
    trend_win = 120
    
    if len(close) < max(z_win, trend_win) + 5 or (close <= 0).all().all():
        return _MR_STATE["weights"]
        
    rolling_mean = close.iloc[-z_win:].mean()
    rolling_std = close.iloc[-z_win:].std().replace(0, np.nan)
    z_score = ((close.iloc[-1] - rolling_mean) / rolling_std).fillna(0.0)
    
    long_ma = close.iloc[-trend_win:].mean()
    uptrend = close.iloc[-1] > long_ma
    eligible = list(uptrend[uptrend].index)
    if len(eligible) < 6:
        eligible = list((close.iloc[-1] / long_ma.replace(0, np.nan)).fillna(0.0).nlargest(6).index)
        
    picks = list(z_score[eligible].nsmallest(6).index)
    vol = close.iloc[-60:].pct_change().std().fillna(0.02).clip(lower=1e-4)
    inv_vol = 1.0 / vol[picks]
    total_inv = inv_vol.sum()
    
    if total_inv <= 0 or not np.isfinite(total_inv):
        w = {s: 1.0 / len(picks) for s in picks}
    else:
        w = (inv_vol / total_inv).to_dict()
        
    _MR_STATE["weights"] = w
    return _MR_STATE["weights"]


def decide_volatility_adjusted_momentum(history):
    """
    Strategy 3 (SELECTED WINNER): Volatility-Adjusted Momentum.
    - Evaluates 180-day momentum, skipping the most recent 5 days to circumvent 1-week
      short-term mean reversion noise (Jegadeesh & Titman 12-1 momentum effect).
    - Normalizes raw return by 60-day historical standard deviation.
    - Selects top k stocks (k=5 or 6 discovered during training).
    - Sizes positions via Inverse Volatility (Risk Parity).
    - Rebalances every 20 trading days (~monthly) to strictly constrain turnover
      and minimize transaction cost drag under the 0.3% fee structure.
    """
    close = history["close"].ffill().bfill()
    stocks = list(close.columns)
    today = close.index[-1]
    
    if "last_day" not in _MOM_STATE or today <= _MOM_STATE["last_day"]:
        _MOM_STATE.clear()
        _MOM_STATE["last_rebal"] = -9999
        _MOM_STATE["weights"] = {s: 1.0 / len(stocks) for s in stocks}
    _MOM_STATE["last_day"] = today
    
    if (today - _MOM_STATE["last_rebal"]) < 20:
        return _MOM_STATE["weights"]
    _MOM_STATE["last_rebal"] = today
    
    lb = MODEL.get("mom_lb", 180)
    k = MODEL.get("mom_k", 5)
    
    if len(close) < lb + 10 or (close <= 0).all().all():
        return _MOM_STATE["weights"]
        
    p_start = close.iloc[-(lb + 5)].replace(0, np.nan)
    p_end = close.iloc[-5]
    raw_mom = (p_end / p_start - 1.0).fillna(0.0)
    vol = close.iloc[-60:].pct_change().std().fillna(0.02).clip(lower=1e-4)
    risk_adj_mom = raw_mom / vol
    
    picks = list(risk_adj_mom.nlargest(k).index)
    inv_vol = 1.0 / vol[picks]
    total_inv = inv_vol.sum()
    
    if total_inv <= 0 or not np.isfinite(total_inv):
        w = {s: 1.0 / len(picks) for s in picks}
    else:
        w = (inv_vol / total_inv).to_dict()
        
    _MOM_STATE["weights"] = w
    return _MOM_STATE["weights"]


# Final decision assignment for competition evaluation
decide = decide_volatility_adjusted_momentum


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import engine

    data = engine.load(Path(__file__).resolve().parent / "data" / "train.csv")
    window = engine.train_agent(sys.modules[__name__], data)
    log = engine.backtest(decide, data, verbose=False)["log"]
    inside = (log.index >= window[0]) & (log.index <= window[1])
    print(f"\n{TEAM_NAME}: trained on days {window[0]}-{window[1]}")
    print("\nInside the training window (Fit):")
    engine.print_metrics(engine.metrics(log[inside]))
    print("\nOutside the training window (Validation):")
    engine.print_metrics(engine.metrics(log[~inside]))
