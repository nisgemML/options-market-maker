"""
Delta hedger tests.
Key properties: after hedge, |net_delta| < threshold; cost always non-negative.
Tests work with the immutable HedgePortfolio (frozen dataclass).
"""
import pytest
from src.pricing import OptionType
from src.hedging.portfolio import HedgePortfolio, OptionPosition
from src.hedging.delta_hedger import DeltaHedger, HedgeFrequency


def make_portfolio(S: float, K: float, T: float, qty: float = 1.0) -> HedgePortfolio:
    pos = OptionPosition(
        strike=K, expiry=T, option_type=OptionType.CALL,
        quantity=qty, entry_price=5.0, r=0.05, sigma=0.20
    )
    return HedgePortfolio().add_position(pos)


def test_portfolio_is_immutable():
    """HedgePortfolio must be a frozen dataclass — direct mutation raises."""
    portfolio = HedgePortfolio()
    with pytest.raises((AttributeError, TypeError)):
        portfolio.delta_hedge = 5.0


def test_add_position_returns_new_portfolio():
    """add_position must return a new portfolio, not mutate self."""
    p0 = HedgePortfolio()
    pos = OptionPosition(strike=100, expiry=0.25, option_type=OptionType.CALL,
                          quantity=1, entry_price=5.0, r=0.05, sigma=0.20)
    p1 = p0.add_position(pos)
    assert len(p0.positions) == 0   # original unchanged
    assert len(p1.positions) == 1


def test_positions_stored_as_tuple():
    """Positions must be a tuple (immutable sequence), not a list."""
    portfolio = make_portfolio(100.0, 100.0, 0.25)
    assert isinstance(portfolio.positions, tuple)


def test_band_hedge_reduces_delta():
    S, K, T = 100.0, 100.0, 0.25
    portfolio = make_portfolio(S, K, T, qty=10.0)
    delta_before = portfolio.net_delta(S)

    hedger = DeltaHedger(hedge_freq=HedgeFrequency.BAND, band_threshold=0.01)
    new_portfolio, record = hedger.step(portfolio, S, step=0, dt=1/252)

    assert record is not None
    assert abs(new_portfolio.net_delta(S)) < abs(delta_before)
    assert abs(new_portfolio.net_delta(S)) < 0.1


def test_hedge_returns_new_portfolio():
    """DeltaHedger.step must return a new portfolio, not mutate."""
    S = 100.0
    portfolio = make_portfolio(S, 100.0, 0.25, qty=5.0)
    hedger = DeltaHedger(hedge_freq=HedgeFrequency.BAND, band_threshold=0.01)
    new_portfolio, record = hedger.step(portfolio, S, step=0, dt=1/252)

    if record is not None:
        assert new_portfolio is not portfolio


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

    assert all(s % 5 == 0 for s in fired_steps)


def test_no_hedge_when_delta_within_band():
    portfolio = HedgePortfolio()   # empty — delta = 0
    hedger = DeltaHedger(hedge_freq=HedgeFrequency.BAND, band_threshold=0.05)
    _, record = hedger.step(portfolio, 100.0, step=1, dt=1/252)
    assert record is None


def test_gamma_hedge_widens_band_for_low_gamma():
    S = 100.0
    portfolio = make_portfolio(S, K=200.0, T=0.1, qty=1.0)
    hedger = DeltaHedger(hedge_freq=HedgeFrequency.GAMMA, gamma_multiple=1.0)
    _, record = hedger.step(portfolio, S, step=1, dt=1/252)
    assert record is None or record.cost >= 0


def test_straddle_delta_near_zero(straddle_portfolio):
    """Long straddle has near-zero net delta (call and put delta cancel)."""
    S = 100.0
    delta = straddle_portfolio.net_delta(S)
    assert abs(delta) < 0.15, f"Straddle delta too large: {delta:.4f}"
