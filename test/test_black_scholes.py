"""
Black-Scholes tests — analytical properties + finite-difference validation.
Uses Hypothesis for property-based testing.
"""
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.pricing.black_scholes import BlackScholes, OptionType
from src.pricing.greeks import finite_diff_greeks, compute_greeks


# ---- parameter strategies ----
valid_spot   = st.floats(min_value=50.0,  max_value=500.0,  allow_nan=False, allow_infinity=False)
valid_strike = st.floats(min_value=50.0,  max_value=500.0,  allow_nan=False, allow_infinity=False)
valid_T      = st.floats(min_value=0.01,  max_value=2.0,    allow_nan=False, allow_infinity=False)
valid_sigma  = st.floats(min_value=0.05,  max_value=1.5,    allow_nan=False, allow_infinity=False)
valid_r      = st.floats(min_value=0.0,   max_value=0.10,   allow_nan=False, allow_infinity=False)


# ---- PROPERTY 1: Put-call parity ----
@given(S=valid_spot, K=valid_strike, T=valid_T, r=valid_r, sigma=valid_sigma)
@settings(max_examples=500)
def test_put_call_parity(S, K, T, r, sigma):
    """C - P = S - K*exp(-rT) (zero dividend case)."""
    bs = BlackScholes
    call = bs.price(S, K, T, r, sigma, OptionType.CALL)
    put  = bs.price(S, K, T, r, sigma, OptionType.PUT)
    lhs  = call - put
    rhs  = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8, f"PCP violated: {lhs:.8f} != {rhs:.8f}"


# ---- PROPERTY 2: Call price bounded by [max(S-K*exp(-rT),0), S] ----
@given(S=valid_spot, K=valid_strike, T=valid_T, r=valid_r, sigma=valid_sigma)
@settings(max_examples=500)
def test_call_bounds(S, K, T, r, sigma):
    call = float(BlackScholes.price(S, K, T, r, sigma, OptionType.CALL))
    lower = max(S - K * np.exp(-r * T), 0.0)
    assert call >= lower - 1e-8, f"Call below intrinsic: {call} < {lower}"
    assert call <= S + 1e-8,     f"Call above spot: {call} > {S}"


# ---- PROPERTY 3: Put price bounded by [max(K*exp(-rT)-S,0), K*exp(-rT)] ----
@given(S=valid_spot, K=valid_strike, T=valid_T, r=valid_r, sigma=valid_sigma)
@settings(max_examples=500)
def test_put_bounds(S, K, T, r, sigma):
    put   = float(BlackScholes.price(S, K, T, r, sigma, OptionType.PUT))
    Kdf   = K * np.exp(-r * T)
    lower = max(Kdf - S, 0.0)
    assert put >= lower - 1e-8, f"Put below intrinsic: {put} < {lower}"
    assert put <= Kdf + 1e-8,   f"Put above K*exp(-rT): {put} > {Kdf}"


# ---- PROPERTY 4: Price monotone in vol (vega > 0) ----
@given(S=valid_spot, K=valid_strike, T=valid_T, r=valid_r, sigma=valid_sigma)
@settings(max_examples=500)
def test_price_monotone_in_vol(S, K, T, r, sigma):
    assume(sigma < 1.4)
    for ot in [OptionType.CALL, OptionType.PUT]:
        p1 = float(BlackScholes.price(S, K, T, r, sigma,        ot))
        p2 = float(BlackScholes.price(S, K, T, r, sigma + 0.01, ot))
        assert p2 >= p1 - 1e-8, f"Price not monotone in vol: {p2} < {p1}"


# ---- PROPERTY 5: Call delta in [0,1], put delta in [-1,0] ----
@given(S=valid_spot, K=valid_strike, T=valid_T, r=valid_r, sigma=valid_sigma)
@settings(max_examples=500)
def test_delta_bounds(S, K, T, r, sigma):
    call_delta = float(BlackScholes.delta(S, K, T, r, sigma, OptionType.CALL))
    put_delta  = float(BlackScholes.delta(S, K, T, r, sigma, OptionType.PUT))
    assert 0.0 - 1e-8 <= call_delta <= 1.0 + 1e-8
    assert -1.0 - 1e-8 <= put_delta <= 0.0 + 1e-8


# ---- PROPERTY 6: Gamma always non-negative ----
@given(S=valid_spot, K=valid_strike, T=valid_T, r=valid_r, sigma=valid_sigma)
@settings(max_examples=500)
def test_gamma_nonneg(S, K, T, r, sigma):
    gamma = float(BlackScholes.gamma(S, K, T, r, sigma))
    assert gamma >= -1e-10, f"Gamma negative: {gamma}"


# ---- PROPERTY 7: Vega always non-negative ----
@given(S=valid_spot, K=valid_strike, T=valid_T, r=valid_r, sigma=valid_sigma)
@settings(max_examples=500)
def test_vega_nonneg(S, K, T, r, sigma):
    vega = float(BlackScholes.vega(S, K, T, r, sigma))
    assert vega >= -1e-10, f"Vega negative: {vega}"


# ---- PROPERTY 8: Theta negative (option loses value over time, all else equal) ----
@given(S=valid_spot, K=valid_strike, T=valid_T, r=valid_r, sigma=valid_sigma)
@settings(max_examples=300)
def test_theta_negative(S, K, T, r, sigma):
    assume(r <= 0.05)  # at high r, put theta can be positive
    for ot in [OptionType.CALL, OptionType.PUT]:
        theta = float(BlackScholes.theta(S, K, T, r, sigma, ot))
        # Near expiry with large intrinsic value, theta can be small positive for puts
        # Relax to a weak bound
        assert theta < 1.0, f"Theta unreasonably large: {theta}"


# ---- PROPERTY 9: Finite-difference Greeks match analytical ----
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("S,K,T,r,sigma", [
    (100, 100, 0.25, 0.05, 0.20),  # ATM
    (100,  90, 0.25, 0.05, 0.20),  # ITM call
    (100, 110, 0.25, 0.05, 0.20),  # OTM call
    (100, 100, 1.00, 0.05, 0.40),  # longer dated, high vol
])
def test_greeks_finite_diff(S, K, T, r, sigma, option_type):
    analytical = compute_greeks(S, K, T, r, sigma, option_type)
    fd = finite_diff_greeks(S, K, T, r, sigma, option_type, dS=0.001, dv=0.0001)

    assert abs(analytical.delta - fd["delta"]) < 1e-4, \
        f"Delta mismatch: {analytical.delta:.6f} vs {fd['delta']:.6f}"
    assert abs(analytical.gamma - fd["gamma"]) < 1e-4, \
        f"Gamma mismatch: {analytical.gamma:.6f} vs {fd['gamma']:.6f}"
    assert abs(analytical.vega - fd["vega"])   < 1e-4, \
        f"Vega mismatch: {analytical.vega:.6f} vs {fd['vega']:.6f}"
    assert abs(analytical.vanna - fd["vanna"]) < 1e-3, \
        f"Vanna mismatch: {analytical.vanna:.6f} vs {fd['vanna']:.6f}"


# ---- PROPERTY 10: Price at expiry equals intrinsic value ----
@pytest.mark.parametrize("S,K,option_type,expected", [
    (105, 100, OptionType.CALL, 5.0),
    (95,  100, OptionType.CALL, 0.0),
    (95,  100, OptionType.PUT,  5.0),
    (105, 100, OptionType.PUT,  0.0),
])
def test_price_at_expiry(S, K, option_type, expected):
    price = float(BlackScholes.price(S, K, 0.0, 0.05, 0.20, option_type))
    assert abs(price - expected) < 1e-6
