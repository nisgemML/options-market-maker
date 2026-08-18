"""
Shared pytest fixtures and Hypothesis configuration.

Hypothesis profiles:
  ci        — 100 examples, used in CI (fast)
  dev       — 300 examples, used locally
  thorough  — 1000 examples, used before releases

Load a profile via:
    pytest --hypothesis-profile=thorough
or set in environment:
    HYPOTHESIS_PROFILE=thorough pytest
"""
import pytest
import numpy as np
from hypothesis import settings, HealthCheck, given
from hypothesis import strategies as st

from src.pricing import OptionType
from src.hedging.portfolio import HedgePortfolio, OptionPosition

# ── Hypothesis profiles ─────────────────────────────────────────────────────
settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    deadline=None,
)
settings.register_profile(
    "dev",
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
settings.register_profile(
    "thorough",
    max_examples=1000,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
settings.load_profile("ci")


# ── Hypothesis strategies ────────────────────────────────────────────────────
# Composable building blocks for property-based tests.

spot_strategy = st.floats(min_value=20.0, max_value=500.0,
                           allow_nan=False, allow_infinity=False)
strike_strategy = st.floats(min_value=20.0, max_value=500.0,
                              allow_nan=False, allow_infinity=False)
expiry_strategy = st.floats(min_value=0.01, max_value=3.0,
                              allow_nan=False, allow_infinity=False)
vol_strategy = st.floats(min_value=0.02, max_value=2.0,
                          allow_nan=False, allow_infinity=False)
rate_strategy = st.floats(min_value=0.0, max_value=0.15,
                           allow_nan=False, allow_infinity=False)
option_type_strategy = st.sampled_from([OptionType.CALL, OptionType.PUT])

quantity_strategy = st.floats(min_value=-10.0, max_value=10.0,
                               allow_nan=False, allow_infinity=False).filter(lambda x: x != 0)


@st.composite
def option_position_strategy(draw) -> OptionPosition:
    """Hypothesis strategy that draws a valid OptionPosition."""
    return OptionPosition(
        strike=draw(strike_strategy),
        expiry=draw(expiry_strategy),
        option_type=draw(option_type_strategy),
        quantity=draw(quantity_strategy),
        entry_price=draw(st.floats(min_value=0.01, max_value=50.0,
                                    allow_nan=False, allow_infinity=False)),
        r=draw(rate_strategy),
        sigma=draw(vol_strategy),
        q=0.0,
    )


@st.composite
def portfolio_strategy(draw, max_legs: int = 4) -> HedgePortfolio:
    """Hypothesis strategy that draws a small HedgePortfolio."""
    n = draw(st.integers(min_value=0, max_value=max_legs))
    portfolio = HedgePortfolio()
    for _ in range(n):
        pos = draw(option_position_strategy())
        portfolio = portfolio.add_position(pos)
    return portfolio


# ── Pytest fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def atm_call_portfolio() -> HedgePortfolio:
    """Single long ATM call position."""
    pos = OptionPosition(
        strike=100.0, expiry=0.25, option_type=OptionType.CALL,
        quantity=1.0, entry_price=5.0, r=0.05, sigma=0.20
    )
    return HedgePortfolio().add_position(pos)


@pytest.fixture
def straddle_portfolio() -> HedgePortfolio:
    """Long ATM straddle (call + put, same strike/expiry)."""
    portfolio = HedgePortfolio()
    for ot in [OptionType.CALL, OptionType.PUT]:
        pos = OptionPosition(
            strike=100.0, expiry=0.25, option_type=ot,
            quantity=1.0, entry_price=5.0, r=0.05, sigma=0.20
        )
        portfolio = portfolio.add_position(pos)
    return portfolio


@pytest.fixture
def flat_price_path() -> np.ndarray:
    return np.full(253, 100.0)
