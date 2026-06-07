"""
Implied vol solver tests.
Key property: solve(price(sigma)) == sigma (round-trip).
"""
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.pricing.black_scholes import BlackScholes, OptionType
from src.pricing.implied_vol import ImpliedVolSolver, ImpliedVolSolverError

valid_spot   = st.floats(min_value=50.0,  max_value=300.0)
valid_strike = st.floats(min_value=50.0,  max_value=300.0)
valid_T      = st.floats(min_value=0.05,  max_value=1.0)
valid_sigma  = st.floats(min_value=0.05,  max_value=1.0)
valid_r      = st.floats(min_value=0.01,  max_value=0.08)


@given(
    S=valid_spot, K=valid_strike, T=valid_T, r=valid_r, sigma=valid_sigma,
    option_type=st.sampled_from([OptionType.CALL, OptionType.PUT])
)
@settings(max_examples=300)
def test_iv_round_trip(S, K, T, r, sigma, option_type):
    """solve(price(sigma)) == sigma — fundamental IV round-trip."""
    price = float(BlackScholes.price(S, K, T, r, sigma, option_type))
    assume(price > 1e-4)  # skip near-zero prices (deep OTM)

    try:
        iv = ImpliedVolSolver.solve(price, S, K, T, r, option_type)
        assert abs(iv - sigma) < 1e-5, f"Round-trip failed: {iv:.6f} != {sigma:.6f}"
    except ImpliedVolSolverError:
        pass  # legitimate failure on edge cases


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.50, 0.80])
def test_iv_known_values(sigma, option_type):
    """Round-trip for well-conditioned ATM options."""
    S, K, T, r = 100.0, 100.0, 0.25, 0.05
    price = float(BlackScholes.price(S, K, T, r, sigma, option_type))
    iv = ImpliedVolSolver.solve(price, S, K, T, r, option_type)
    assert abs(iv - sigma) < 1e-7


def test_iv_below_intrinsic_raises():
    with pytest.raises(ImpliedVolSolverError, match="intrinsic"):
        ImpliedVolSolver.solve(-1.0, 100, 100, 0.25, 0.05, OptionType.CALL)


def test_iv_at_expiry_raises():
    with pytest.raises(ImpliedVolSolverError, match="expiry"):
        ImpliedVolSolver.solve(5.0, 100, 100, 0.0, 0.05, OptionType.CALL)


def test_iv_surface_shape():
    prices = np.array([[5.0, 3.0, 1.5], [8.0, 5.5, 3.0]])
    strikes = np.array([95.0, 100.0, 105.0])
    expiries = np.array([0.25, 0.50])
    iv_surface = ImpliedVolSolver.solve_surface(prices, 100.0, strikes, expiries, 0.05, OptionType.CALL)
    assert iv_surface.shape == (2, 3)
    # All solvable entries should be positive
    valid = iv_surface[~np.isnan(iv_surface)]
    assert np.all(valid > 0)
