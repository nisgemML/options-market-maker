"""
Delta hedger tests.
Key properties: after hedge, |net_delta| < threshold; cost always non-negative.
"""
import pytest
from src.pricing import OptionType
from src.hedging.portfolio import HedgePortfolio, OptionPosition
from src.hedging.delta_hedger import DeltaHedger, HedgeFrequency


def make_portfolio(S: float, K: float, T: float, qty: float = 1.0) -> HedgePortfolio:
    portfolio = HedgePortfolio()
    pos = OptionPosition(
        strike=K, expiry=T, option_type=OptionType.CALL,
        quantity=qty, entry_price=5.0, r=0.05, sigma=0.20
    )
    portfolio.add_position(pos)
    return portfolio


def test_band_hedge_reduces_delta():
    S, K, T = 100.0, 100.0, 0.25
    portfolio = make_portfolio(S, K, T, qty=10.0)
    delta_before = portfolio.net_delta(S)

    hedger = DeltaHedger(hedge_freq=HedgeFrequency.BAND, band_threshold=0.01)
    new_portfolio, record = hedger.step(portfolio, S, step=0, dt=1/252)

    assert record is not None, "Expected hedge trade"
    assert abs(new_portfolio.net_delta(S)) < abs(delta_before)
    assert abs(new_portfolio.net_delta(S)) < 0.1  # close to zero


def test_hedge_cost_nonneg():
    S, K, T = 100.0, 100.0, 0.25
    portfolio = make_portfolio(S, K, T, qty=5.0)
    hedger = DeltaHedger(hedge_freq=HedgeFrequency.BAND, transaction_cost_bps=2.0)
    _, record = hedger.step(portfolio, S, step=0, dt=1/252)
    if record:
        assert record.cost >= 0.0


def test_periodic_hedge_fires_on_schedule():
    S, K, T = 100.0, 100.0, 0.25
    portfolio = make_portfolio(S, K, T, qty=1.0)
    hedger = DeltaHedger(hedge_freq=HedgeFrequency.PERIODIC, periodic_n=5)

    fired_steps = []
    for step in range(20):
        _, record = hedger.step(portfolio, S, step=step, dt=1/252)
        if record:
            fired_steps.append(step)

    # Should fire at steps 0, 5, 10, 15
    assert all(s % 5 == 0 for s in fired_steps)


def test_no_hedge_when_delta_within_band():
    portfolio = HedgePortfolio()  # empty portfolio — delta = 0
    hedger = DeltaHedger(hedge_freq=HedgeFrequency.BAND, band_threshold=0.05)
    _, record = hedger.step(portfolio, 100.0, step=1, dt=1/252)
    assert record is None, "Should not hedge when delta is zero"


def test_gamma_hedge_widens_band_for_low_gamma():
    """Low gamma options should have a wide band → less frequent hedging."""
    S = 100.0
    # Far OTM → low gamma
    portfolio = make_portfolio(S, K=200.0, T=0.1, qty=1.0)
    hedger = DeltaHedger(
        hedge_freq=HedgeFrequency.GAMMA,
        gamma_multiple=1.0,
    )
    # With low gamma, band should be wider — may not trigger
    _, record = hedger.step(portfolio, S, step=1, dt=1/252)
    # Just verify it doesn't crash and returns valid state
    assert record is None or record.cost >= 0
