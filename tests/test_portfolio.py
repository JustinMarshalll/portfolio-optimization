"""Checks the optimiser against cases with known answers.

Run from the repo root with:  python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from src import portfolio as pf

RF = 0.05


@pytest.fixture
def market():
    mean = np.array([0.10, 0.14, 0.18, 0.12])
    vols = np.array([0.15, 0.22, 0.30, 0.18])
    corr = np.array([
        [1.0, 0.3, 0.2, 0.4],
        [0.3, 1.0, 0.5, 0.3],
        [0.2, 0.5, 1.0, 0.2],
        [0.4, 0.3, 0.2, 1.0],
    ])
    cov = np.outer(vols, vols) * corr
    return mean, cov


def test_min_variance_two_uncorrelated_assets():
    # Closed form for two uncorrelated assets: w1 = s2^2 / (s1^2 + s2^2)
    cov = np.diag([0.2**2, 0.3**2])
    w = pf.min_variance_portfolio(cov)
    expected = 0.3**2 / (0.2**2 + 0.3**2)
    assert w[0] == pytest.approx(expected, abs=1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_max_sharpe_matches_closed_form(market):
    # Without the long-only bound the tangency portfolio is proportional to
    # inv(C) (mu - rf). For this market every weight is positive, so the
    # long-only optimiser must find the same answer.
    mean, cov = market
    raw = np.linalg.solve(cov, mean - RF)
    expected = raw / raw.sum()
    assert (expected > 0).all()
    w = pf.max_sharpe_portfolio(mean, cov, RF)
    np.testing.assert_allclose(w, expected, atol=1e-4)


def test_optimisers_beat_every_random_portfolio(market):
    mean, cov = market
    sims, weights = pf.simulate_portfolios(mean, cov, RF, n=5000, seed=0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert (weights >= 0).all()

    w_sharpe = pf.max_sharpe_portfolio(mean, cov, RF)
    w_minvar = pf.min_variance_portfolio(cov)
    assert pf.sharpe_ratio(w_sharpe, mean, cov, RF) >= sims["sharpe"].max() - 1e-9
    assert pf.portfolio_volatility(w_minvar, cov) <= sims["volatility"].min() + 1e-9


def test_long_only_bounds_hold_when_an_asset_is_poor():
    # Asset 2 has a low return and high risk; unconstrained it would be shorted.
    mean = np.array([0.15, 0.02, 0.12])
    cov = np.diag([0.2, 0.3, 0.18]) ** 2
    w = pf.max_sharpe_portfolio(mean, cov, RF)
    assert (w >= 0).all()
    assert w[1] == pytest.approx(0.0, abs=1e-6)


def test_frontier_is_increasing(market):
    mean, cov = market
    frontier = pf.efficient_frontier(mean, cov, n_points=20)
    assert frontier["return"].is_monotonic_increasing
    assert frontier["volatility"].diff().dropna().ge(-1e-9).all()
    # its first point is the minimum-variance portfolio
    w_min = pf.min_variance_portfolio(cov)
    assert frontier["volatility"].iloc[0] == pytest.approx(pf.portfolio_volatility(w_min, cov), rel=1e-4)


def test_buy_and_hold_and_drawdown():
    returns = pd.DataFrame({"A": [0.10, -0.50, 0.0], "B": [0.0, 0.0, 0.0]})
    values = pf.buy_and_hold(returns, [0.5, 0.5])
    # A: 1.1, 0.55, 0.55 ; B: 1, 1, 1
    np.testing.assert_allclose(values.values, [1.05, 0.775, 0.775])
    assert pf.max_drawdown(values) == pytest.approx(0.775 / 1.05 - 1)


def test_annualised_stats_scaling():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0.001, 0.01, size=(1000, 2)), columns=["A", "B"])
    mean, cov = pf.annualised_stats(returns)
    np.testing.assert_allclose(mean, returns.mean() * 252)
    np.testing.assert_allclose(cov, returns.cov() * 252)


def test_max_sharpe_refuses_when_nothing_beats_rf(market):
    mean, cov = market
    with pytest.raises(ValueError):
        pf.max_sharpe_portfolio(mean, cov, rf=0.25)


def test_risk_contributions():
    # Identical, uncorrelated assets held equally contribute equally
    cov = np.eye(4) * 0.04
    np.testing.assert_allclose(pf.risk_contributions(np.full(4, 0.25), cov), 0.25)
    # In general the shares add up to 1
    rng = np.random.default_rng(2)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T
    w = rng.dirichlet(np.ones(5))
    assert pf.risk_contributions(w, cov).sum() == pytest.approx(1.0)


def test_block_bootstrap_keeps_blocks_and_length():
    returns = pd.DataFrame({"A": np.arange(100.0), "B": np.arange(100.0)})
    sample = pf.block_bootstrap(returns, block=10, rng=0)
    assert len(sample) == 100
    # inside each block of 10 the original days stay consecutive
    a = sample["A"].to_numpy().reshape(10, 10)
    assert (np.diff(a, axis=1) == 1).all()


def test_bootstrap_max_sharpe_is_reproducible(market):
    mean, cov = market
    rng = np.random.default_rng(5)
    daily = pd.DataFrame(rng.multivariate_normal(mean / 252, cov / 252, size=750), columns=list("ABCD"))
    w1, skipped1 = pf.bootstrap_max_sharpe(daily, RF, n_boot=30, seed=1)
    w2, _ = pf.bootstrap_max_sharpe(daily, RF, n_boot=30, seed=1)
    pd.testing.assert_frame_equal(w1, w2)
    assert len(w1) + skipped1 == 30
    np.testing.assert_allclose(w1.sum(axis=1), 1.0)
    assert (w1.to_numpy() >= 0).all()


def test_resampled_weights_average():
    boot = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    np.testing.assert_allclose(pf.resampled_max_sharpe(boot), [0.5, 0.5])


def test_sharpe_standard_error():
    # SR = 0 over 4 years of daily data: sqrt(252 / 1008) = 0.5
    assert pf.sharpe_standard_error(0.0, 4 * 252) == pytest.approx(0.5)
    # annual SR of 1 is a daily SR of 1/sqrt(252)
    expected = np.sqrt((1 + 0.5 / 252) / 504) * np.sqrt(252)
    assert pf.sharpe_standard_error(1.0, 504) == pytest.approx(expected)


def test_walk_forward_matches_buy_and_hold_with_one_rebalance():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2021-01-01", periods=300)
    returns = pd.DataFrame(rng.normal(0.0005, 0.01, size=(300, 2)), index=idx, columns=["A", "B"])
    w = np.array([0.3, 0.7])
    values, chosen = pf.walk_forward(returns, [idx[100]], lookback=50, weight_fn=lambda h: w, rf=0.05)
    expected = pf.buy_and_hold(returns.iloc[100:], w)
    np.testing.assert_allclose(values.iloc[1:].values, expected.values)
    assert values.iloc[0] == 1.0 and values.index[0] == idx[99]
    np.testing.assert_allclose(chosen.iloc[0].values, w)


def test_walk_forward_cash_and_chaining():
    idx = pd.bdate_range("2021-01-01", periods=30)
    returns = pd.DataFrame({"A": np.full(30, 0.01)}, index=idx)
    rf = 0.10
    # first period in cash, second fully invested in A
    fns = iter([None, np.array([1.0])])
    values, chosen = pf.walk_forward(returns, [idx[10], idx[20]], lookback=5,
                                     weight_fn=lambda h: next(fns), rf=rf)
    daily_cash = (1 + rf) ** (1 / 252) - 1
    expected_end = (1 + daily_cash) ** 10 * 1.01 ** 10
    assert values.iloc[-1] == pytest.approx(expected_end)
    assert chosen.iloc[0].isna().all()


def test_walk_forward_uses_only_past_data():
    idx = pd.bdate_range("2021-01-01", periods=40)
    returns = pd.DataFrame({"A": np.arange(40.0) / 1000}, index=idx)
    seen = []
    def fn(history):
        seen.append(history.index.max())
        return np.array([1.0])
    pf.walk_forward(returns, [idx[20], idx[30]], lookback=10, weight_fn=fn, rf=0.0)
    assert seen == [idx[19], idx[29]]
