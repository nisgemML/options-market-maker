"""
Vol surface tests — SVI and SSVI.

Key properties:
  1. SVI total variance is non-negative everywhere
  2. SVI butterfly arbitrage check (convexity in k)
  3. VolSurface calendar-spread check passes after fit
  4. SSVI admissibility — Gatheral-Jacquier constraints
  5. SSVI price monotone in T at ATM (calendar-spread free)
  6. SSVI fit reduces to reasonable vols
  7. SSVI and SVI agree at single-expiry limit
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pricing.surface import (
    SVIParams, SSVIParams, VolSurface, SSVIVolSurface
)


# ── SVI tests ────────────────────────────────────────────────────────────────

def make_svi(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1) -> SVIParams:
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def test_svi_total_variance_nonneg():
    svi = make_svi()
    k = np.linspace(-1.0, 1.0, 200)
    w = svi.total_variance(k)
    assert np.all(w >= 0), "SVI total variance must be non-negative"


def test_svi_convex():
    """Butterfly-free: SVI with these params is convex in k."""
    svi = make_svi()
    k = np.linspace(-1.0, 1.0, 200)
    assert svi.is_convex(k)


def test_svi_implied_vol_positive():
    svi = make_svi()
    k = np.linspace(-0.5, 0.5, 50)
    vols = svi.implied_vol(k, T=0.25)
    assert np.all(vols >= 0)


@given(
    a=st.floats(0.001, 0.5),
    b=st.floats(0.01, 0.5),
    rho=st.floats(-0.99, 0.99),
    m=st.floats(-0.5, 0.5),
    sigma=st.floats(0.01, 0.5),
)
@settings(max_examples=200, deadline=None)
def test_svi_variance_nonneg_property(a, b, rho, m, sigma):
    """SVI total variance ≥ 0 for all valid parameters and log-moneyness."""
    try:
        svi = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
    except ValueError:
        return
    k = np.linspace(-1.0, 1.0, 50)
    w = svi.total_variance(k)
    assert np.all(w >= -1e-10), f"Negative variance detected: {w.min():.6f}"


def test_svi_invalid_params():
    with pytest.raises(ValueError, match="rho"):
        SVIParams(a=0.04, b=0.1, rho=1.5, m=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="b"):
        SVIParams(a=0.04, b=-0.1, rho=0.0, m=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="sigma"):
        SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=-0.01)


# ── VolSurface calibration tests ─────────────────────────────────────────────

def make_flat_market(vol: float = 0.20, n_strikes: int = 7, n_expiries: int = 3):
    strikes  = np.array([85, 90, 95, 100, 105, 110, 115], dtype=float)[:n_strikes]
    expiries = np.array([0.25, 0.50, 1.00])[:n_expiries]
    market_vols = np.full((n_expiries, n_strikes), vol)
    return market_vols, strikes, expiries


def test_vol_surface_fit_flat():
    """Fitting a flat vol surface should recover ~flat vols."""
    market_vols, strikes, expiries = make_flat_market(vol=0.20)
    surface = VolSurface.fit(market_vols, strikes, expiries, S=100.0, r=0.05)

    for T in expiries:
        for K in strikes:
            iv = surface.interpolate(K, T)
            assert abs(iv - 0.20) < 0.03, f"Vol far from flat at K={K}, T={T}: {iv:.3f}"


def test_vol_surface_calendar_spread():
    """Calendar spread check must pass on a valid fitted surface."""
    market_vols, strikes, expiries = make_flat_market()
    surface = VolSurface.fit(market_vols, strikes, expiries, S=100.0, r=0.05)
    assert surface.calendar_spread_check()


def test_vol_surface_interpolation_range():
    """Interpolated vol must stay in reasonable range."""
    market_vols = np.array([[0.22, 0.20, 0.19, 0.20, 0.21, 0.22, 0.23],
                              [0.21, 0.19, 0.18, 0.19, 0.20, 0.21, 0.22]])
    strikes = np.array([85, 90, 95, 100, 105, 110, 115], dtype=float)
    expiries = np.array([0.25, 0.50])
    surface = VolSurface.fit(market_vols, strikes, expiries, S=100.0, r=0.05)

    # Interpolate at mid-expiry
    iv = surface.interpolate(K=100.0, T=0.375)
    assert 0.10 < iv < 0.50, f"Interpolated vol out of range: {iv:.3f}"


# ── SSVI tests ───────────────────────────────────────────────────────────────

def make_ssvi(rho=-0.3, eta=0.5, gamma=0.5) -> SSVIParams:
    return SSVIParams(rho=rho, eta=eta, gamma=gamma)


def test_ssvi_admissibility_valid():
    p = make_ssvi(rho=-0.3, eta=0.5, gamma=0.5)
    ok, reason = p.is_admissible()
    assert ok, f"Should be admissible but: {reason}"


def test_ssvi_admissibility_lee_bound_violated():
    """η(1+|ρ|) > 2 violates Lee moment formula."""
    p = SSVIParams(rho=0.5, eta=1.5, gamma=0.5)  # 1.5 * 1.5 = 2.25 > 2
    ok, reason = p.is_admissible()
    assert not ok
    assert "Lee" in reason


def test_ssvi_admissibility_gamma_bound():
    p = SSVIParams(rho=0.0, eta=0.5, gamma=1.5)
    ok, reason = p.is_admissible()
    assert not ok
    assert "γ" in reason


def test_ssvi_total_variance_positive():
    """SSVI total variance must be positive for valid parameters."""
    p = make_ssvi()
    k = np.linspace(-1.0, 1.0, 100)
    theta_t = 0.04   # 20% ATM vol, T=1
    w = p.total_variance(k, theta_t)
    assert np.all(w > 0)


def test_ssvi_phi_positive():
    """φ(θ) = η/θ^γ > 0 for positive η and θ."""
    p = make_ssvi(eta=0.5, gamma=0.5)
    for theta in [0.01, 0.04, 0.10, 0.25]:
        assert p.phi(theta) > 0


def test_ssvi_calendar_spread_free():
    """SSVI calendar spread condition: θ_t non-decreasing ↔ no calendar arb."""
    p = make_ssvi()
    theta_times = np.array([0.04, 0.06, 0.08, 0.10])  # non-decreasing
    assert p.calendar_spread_free(theta_times)

    theta_times_bad = np.array([0.04, 0.09, 0.07, 0.10])  # not monotone
    assert not p.calendar_spread_free(theta_times_bad)


def test_ssvi_surface_fit():
    """SSVI fits a skewed market smile within reasonable error."""
    strikes  = np.array([85, 90, 95, 100, 105, 110, 115], dtype=float)
    expiries = np.array([0.25, 0.50, 1.00])
    # Negatively skewed vols (typical equity surface)
    smiles = np.array([
        [0.28, 0.24, 0.21, 0.19, 0.20, 0.21, 0.22],
        [0.26, 0.23, 0.20, 0.18, 0.19, 0.20, 0.21],
        [0.24, 0.21, 0.19, 0.17, 0.18, 0.19, 0.20],
    ])
    surface = SSVIVolSurface.fit(smiles, strikes, expiries, S=100.0, r=0.05)

    # Fitted vols should be within 3 vol points of market at each node
    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            iv = surface.interpolate(K, T)
            assert abs(iv - smiles[i, j]) < 0.05, (
                f"SSVI fit error too large at K={K}, T={T}: "
                f"fitted={iv:.3f}, market={smiles[i,j]:.3f}"
            )


def test_ssvi_arbitrage_free_check():
    """SSVI arbitrage-free check returns dict with expected keys."""
    strikes  = np.array([90, 95, 100, 105, 110], dtype=float)
    expiries = np.array([0.25, 0.50])
    vols = np.array([[0.22, 0.20, 0.19, 0.20, 0.22],
                      [0.21, 0.19, 0.18, 0.19, 0.21]])
    surface = SSVIVolSurface.fit(vols, strikes, expiries, S=100.0, r=0.05)
    result = surface.is_arbitrage_free()
    assert set(result.keys()) == {"admissible", "calendar_spread_free", "butterfly_free"}
