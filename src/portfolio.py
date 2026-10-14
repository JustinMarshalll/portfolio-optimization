"""Mean-variance portfolio tools used by analysis.ipynb.

Everything here works on annualised figures. Daily simple returns are
annualised by multiplying the mean by 252 and the covariance by 252
(about 252 trading days a year).

Portfolios are long-only and fully invested: every weight is between 0 and 1
and the weights add up to 1.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_prices(tickers, start, end, cache_path=None):
    """Return a DataFrame of daily adjusted close prices (one column per ticker).

    If ``cache_path`` exists the prices are read from it, so the notebook gives
    the same numbers every time it is run. Otherwise they are downloaded from
    Yahoo Finance and saved to ``cache_path``.
    """
    if cache_path is not None and Path(cache_path).exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(
                f"{cache_path} has no column for {missing}. Delete the file to re-download."
            )
        prices = prices.loc[start:end, list(tickers)]
        # end is exclusive, matching yfinance
        return prices[prices.index < pd.Timestamp(end)]

    import yfinance as yf

    # yfinance's price repair (fixes 100x errors, common with JSE prices quoted
    # in cents) needs scikit-learn. Without it every download fails, so fall
    # back to unrepaired prices with a warning.
    try:
        import sklearn  # noqa: F401
        repair = True
    except ImportError:
        repair = False
        warnings.warn(
            "scikit-learn is not installed, so yfinance cannot repair bad prices. "
            "Run `pip install scikit-learn`, delete data/prices.csv and re-run for repaired data."
        )

    raw = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,  # Close is adjusted for dividends and splits
        repair=repair,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            "Yahoo Finance returned no data. The 'Failed downloads' message printed above "
            "gives the reason for each ticker."
        )

    prices = raw["Close"]
    if isinstance(prices, pd.Series):  # a single ticker comes back as a Series
        prices = prices.to_frame(tickers[0])

    empty = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if empty:
        raise RuntimeError(f"No price data for {empty}. Check the ticker symbols on finance.yahoo.com.")

    prices = prices[list(tickers)]
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)
    return prices


def clean_prices(prices, max_gap=5):
    """Drop days where every price is missing, then forward-fill short gaps.

    A gap is a day where one share has no price but others do (a suspension or
    a missing data point). Gaps of up to ``max_gap`` days are filled with the
    last known price, which means a return of zero on those days. Rows that are
    still incomplete after that are dropped.
    """
    prices = prices.dropna(how="all")
    prices = prices.ffill(limit=max_gap)
    return prices.dropna()


def daily_returns(prices):
    """Simple daily returns: P_t / P_{t-1} - 1."""
    return prices.pct_change().dropna()


def annualised_stats(returns):
    """Annualised mean return vector and covariance matrix."""
    mean = returns.mean() * TRADING_DAYS
    cov = returns.cov() * TRADING_DAYS
    return mean, cov


# ---------------------------------------------------------------------------
# Portfolio maths
# ---------------------------------------------------------------------------

def portfolio_return(w, mean):
    return float(np.asarray(w) @ np.asarray(mean))


def portfolio_volatility(w, cov):
    w = np.asarray(w)
    return float(np.sqrt(w @ np.asarray(cov) @ w))


def sharpe_ratio(w, mean, cov, rf):
    return (portfolio_return(w, mean) - rf) / portfolio_volatility(w, cov)


def performance(w, mean, cov, rf):
    """Expected return, volatility and Sharpe ratio of one portfolio."""
    ret = portfolio_return(w, mean)
    vol = portfolio_volatility(w, cov)
    return {"return": ret, "volatility": vol, "sharpe": (ret - rf) / vol}


def simulate_portfolios(mean, cov, rf, n=20_000, seed=42):
    """Evaluate ``n`` random long-only portfolios.

    Weights are drawn from a flat Dirichlet distribution, which spreads them
    evenly over every possible way of splitting 100% between the assets.
    (Drawing uniform numbers and dividing by their sum, the other common
    approach, crowds the weights towards equal-weight.)

    Returns a DataFrame with columns return, volatility, sharpe, and the weights
    as an (n, k) array.
    """
    rng = np.random.default_rng(seed)
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    weights = rng.dirichlet(np.ones(len(mean)), size=n)

    rets = weights @ mean
    # row-wise w_i' C w_i without building an n x n matrix
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
    results = pd.DataFrame({"return": rets, "volatility": vols, "sharpe": (rets - rf) / vols})
    return results, weights


def _solve(objective, k, extra_constraints=()):
    """Minimise ``objective`` over long-only, fully invested weights."""
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *extra_constraints]
    result = minimize(
        objective,
        x0=np.full(k, 1.0 / k),  # start from equal weight
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Optimiser failed: {result.message}")
    w = np.clip(result.x, 0.0, None)  # remove tiny negative rounding errors
    return w / w.sum()


def max_sharpe_portfolio(mean, cov, rf):
    """Weights of the long-only portfolio with the highest Sharpe ratio.

    If no asset beats the risk-free rate, every long-only portfolio has a
    negative Sharpe ratio and "maximum Sharpe" stops meaning anything useful
    (the optimiser would pick the riskiest mix to make the ratio less negative),
    so this raises an error instead.
    """
    mean, cov = np.asarray(mean), np.asarray(cov)
    if mean.max() <= rf:
        raise ValueError(
            f"No asset has an expected return above the risk-free rate ({rf:.2%}); "
            "the maximum-Sharpe portfolio is not defined. Lower the risk-free rate "
            "or change the assets or the period."
        )
    return _solve(lambda w: -sharpe_ratio(w, mean, cov, rf), len(mean))


def min_variance_portfolio(cov):
    """Weights of the long-only portfolio with the lowest volatility."""
    cov = np.asarray(cov)
    return _solve(lambda w: w @ cov @ w, cov.shape[0])


def efficient_frontier(mean, cov, n_points=60):
    """Trace the long-only efficient frontier.

    For each target return between the minimum-variance portfolio's return and
    the highest single-asset return, find the portfolio with the lowest
    volatility that earns exactly that return.

    Returns a DataFrame with columns return and volatility.
    """
    mean, cov = np.asarray(mean), np.asarray(cov)
    w_min = min_variance_portfolio(cov)
    targets = np.linspace(w_min @ mean, mean.max(), n_points)

    rows = []
    for target in targets:
        target_constraint = {"type": "eq", "fun": lambda w, t=target: w @ mean - t}
        w = _solve(lambda w: w @ cov @ w, len(mean), [target_constraint])
        rows.append({"return": w @ mean, "volatility": np.sqrt(w @ cov @ w)})
    return pd.DataFrame(rows)


def risk_contributions(w, cov):
    """Share of portfolio variance that comes from each asset.

    Asset i contributes w_i * (C w)_i to the variance w'Cw, so the shares add
    up to 1. A share can be larger or smaller than the asset's weight: a
    volatile asset that moves with the rest contributes more risk than its
    weight, and one that moves against the rest contributes less.
    """
    w, cov = np.asarray(w), np.asarray(cov)
    return w * (cov @ w) / (w @ cov @ w)


# ---------------------------------------------------------------------------
# Estimation error
# ---------------------------------------------------------------------------

def block_bootstrap(returns, block=21, rng=None):
    """One moving-block bootstrap resample of a returns DataFrame.

    Days are drawn in consecutive blocks (21 trading days is about a month)
    rather than one at a time, so calm and volatile spells stay together the
    way they do in real markets. The resample has the same length as the input.
    """
    rng = np.random.default_rng(rng)
    n = len(returns)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)).ravel()[:n]
    return returns.iloc[idx]


def bootstrap_max_sharpe(returns, rf, n_boot=500, block=21, seed=0):
    """Maximum-Sharpe weights re-estimated on ``n_boot`` bootstrap resamples.

    Shows how much the "optimal" weights depend on the particular five years of
    data that happened to occur. Resamples in which no asset beats the
    risk-free rate have no maximum-Sharpe portfolio and are skipped; the number
    skipped is returned so it can be reported.

    Returns (DataFrame of weights with one row per usable resample, number skipped).
    """
    rng = np.random.default_rng(seed)
    rows, skipped = [], 0
    for _ in range(n_boot):
        sample = block_bootstrap(returns, block=block, rng=rng)
        mean, cov = annualised_stats(sample)
        try:
            rows.append(max_sharpe_portfolio(mean, cov, rf))
        except (ValueError, RuntimeError):
            skipped += 1
    return pd.DataFrame(rows, columns=returns.columns), skipped


# ---------------------------------------------------------------------------
# Out-of-sample backtest
# ---------------------------------------------------------------------------

def buy_and_hold(returns, weights):
    """Daily value of R1 invested at the start with ``weights`` and never rebalanced.

    Each holding grows with its own returns, so the weights drift over time,
    which is what would happen to a real portfolio left alone.
    """
    growth = (1 + returns).cumprod()            # value of R1 in each asset
    return growth @ np.asarray(weights)


def max_drawdown(values):
    """Largest fall from a previous peak, as a negative fraction."""
    return float((values / values.cummax() - 1).min())


def backtest_summary(values, rf):
    """Annualised return, volatility, Sharpe and max drawdown of a value series."""
    daily = values.pct_change().dropna()
    years = len(daily) / TRADING_DAYS
    total = values.iloc[-1] / values.iloc[0]
    cagr = total ** (1 / years) - 1
    vol = daily.std() * np.sqrt(TRADING_DAYS)
    ann_mean = daily.mean() * TRADING_DAYS     # same convention as the in-sample Sharpe
    return {
        "total return": total - 1,
        "annualised return (CAGR)": cagr,
        "volatility": vol,
        "sharpe": (ann_mean - rf) / vol,
        "max drawdown": max_drawdown(values),
    }


def resampled_max_sharpe(boot_weights):
    """Average of the bootstrap max-Sharpe weights (Michaud-style resampling).

    Averaging over resamples keeps the assets the optimiser picks again and
    again and shrinks the ones it only picks in a few samples, so the result is
    usually more spread out and less sensitive to one particular history.
    """
    w = np.asarray(boot_weights).mean(axis=0)
    return w / w.sum()


def sharpe_standard_error(sharpe, n_days):
    """Approximate standard error of an annualised Sharpe ratio from daily data.

    Lo (2002), "The Statistics of Sharpe Ratios", gives the standard error of a
    Sharpe ratio estimated from T independent, identically distributed returns
    as sqrt((1 + SR^2 / 2) / T), with SR in the same per-period units. Here SR
    is converted to a daily figure, the formula applied with T = n_days, and the
    result scaled back to annual units. Over two years this is about 0.7, so
    Sharpe ratios closer together than that are not clearly different.
    """
    daily = sharpe / np.sqrt(TRADING_DAYS)
    return float(np.sqrt((1 + 0.5 * daily**2) / n_days) * np.sqrt(TRADING_DAYS))


def walk_forward(returns, rebalance_dates, lookback, weight_fn, rf):
    """Walk-forward backtest with periodic re-optimisation.

    On each rebalance date the weights are re-estimated with ``weight_fn`` using
    only the ``lookback`` trading days before that date, then held (and allowed
    to drift) until the next rebalance date. If ``weight_fn`` returns None, the
    money sits in cash earning ``rf`` for that period.

    Returns (value series starting at 1 on the day before the first rebalance,
    DataFrame of the weights chosen on each rebalance date, with NaN for cash).
    """
    rebalance_dates = [pd.Timestamp(d) for d in rebalance_dates]
    cash_daily = (1 + rf) ** (1 / TRADING_DAYS) - 1
    first = returns.index[returns.index >= rebalance_dates[0]][0]
    value = 1.0
    values = [pd.Series([1.0], index=[returns.index[returns.index.get_loc(first) - 1]])]
    chosen = {}

    for i, date in enumerate(rebalance_dates):
        history = returns[returns.index < date].iloc[-lookback:]
        end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        period = returns[returns.index >= date]
        if end is not None:
            period = period[period.index < end]
        if period.empty:
            continue

        w = weight_fn(history)
        if w is None:
            chosen[date] = pd.Series(np.nan, index=returns.columns)
            path = value * (1 + cash_daily) ** np.arange(1, len(period) + 1)
            path = pd.Series(path, index=period.index)
        else:
            chosen[date] = pd.Series(w, index=returns.columns)
            path = value * buy_and_hold(period, w)
        values.append(path)
        value = float(path.iloc[-1])

    return pd.concat(values), pd.DataFrame(chosen).T
