"""
Heston model tests.

Key properties tested:
  1. BS limit: Heston price → BS price as ξ → 0 (vol-of-vol vanishes)
  2. Put-call parity holds under Heston
  3. Feller condition warning when 2κθ ≤ ξ²
  4. Smile shape: Heston generates skew (non-flat IV smile)
  5. Scenario paths: E[v_t] → θ at long horizons (mean reversion)
  6. Negative rho produces negative skew (left skew, as in equity markets)
"""
import warnings

import numpy as np
import pytest

from src.pricing.black_scholes import BlackScholes, OptionType
from src.pricing.heston import heston_price, heston_implied_vol
from src.backtest.scenario import HestonScenario, HestonConfig


# ── Standard Heston parameters ───────────────────────────────────────────────
BASE = dict(S=100.0, K=100.0, T=0.5, r=0.05,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.7)


def test_heston_bs_limit_call():
    """
    As ξ → 0, Heston price converges to BS price with σ = √v₀.
    This is the fundamental consistency check between the two pricers.
    """
    S, K, T, r, v0 = 100.0, 100.0, 0.5, 0.05, 0.04
    sigma = np.sqrt(v0)  # σ = √v₀

    bs_price = float(BlackScholes.price(S, K, T, r, sigma, OptionType.CALL))
    # xi → 0 limit: use very small xi, neutral rho
    heston = heston_price(S, K, T, r, v0, kappa=2.0, theta=v0, xi=0.01, rho=0.0, option_type="call")

    assert abs(heston - bs_price) < 0.02, (
        f"Heston BS limit failed: heston={heston:.4f}, bs={bs_price:.4f}, "
        f"diff={abs(heston-bs_price):.4f}"
    )


def test_heston_bs_limit_put():
    """Same BS limit for put options."""
    S, K, T, r, v0 = 100.0, 100.0, 0.5, 0.05, 0.04
    sigma = np.sqrt(v0)

    bs_price = float(BlackScholes.price(S, K, T, r, sigma, OptionType.PUT))
    heston = heston_price(S, K, T, r, v0, kappa=2.0, theta=v0, xi=0.01, rho=0.0, option_type="put")

    assert abs(heston - bs_price) < 0.02


def test_heston_put_call_parity():
    """C - P = S - K·e^{-rT} must hold under Heston."""
    p = BASE
    call = heston_price(**{**p, "option_type": "call"})
    put  = heston_price(**{**p, "option_type": "put"})
    lhs  = call - put
    rhs  = p["S"] - p["K"] * np.exp(-p["r"] * p["T"])
    assert abs(lhs - rhs) < 1e-2, f"PCP violated: {lhs:.4f} vs {rhs:.4f}"


def test_heston_price_positive():
    """Option prices must be non-negative."""
    for ot in ["call", "put"]:
        price = heston_price(**{**BASE, "option_type": ot})
        assert price >= 0.0, f"Negative {ot} price: {price}"


def test_heston_price_above_intrinsic():
    """Heston price must be above intrinsic value."""
    p = BASE
    call = heston_price(**{**p, "option_type": "call"})
    intrinsic = max(p["S"] - p["K"] * np.exp(-p["r"] * p["T"]), 0.0)
    assert call >= intrinsic - 1e-6


def test_heston_generates_skew():
    """
    Heston with negative ρ generates negative skew:
    IV(OTM put) > IV(ATM) > IV(OTM call)
    This is the fundamental stylised fact of equity vol markets.
    """
    S, T, r, v0 = 100.0, 0.5, 0.05, 0.04
    heston_kwargs = dict(T=T, r=r, v0=v0, kappa=2.0, theta=v0, xi=0.5, rho=-0.7)

    iv_otm_put  = heston_implied_vol(S, K=90.0,  **heston_kwargs, option_type="put")
    iv_atm      = heston_implied_vol(S, K=100.0, **heston_kwargs, option_type="call")
    iv_otm_call = heston_implied_vol(S, K=110.0, **heston_kwargs, option_type="call")

    assert iv_otm_put > iv_atm, (
        f"No negative skew: IV(90p)={iv_otm_put:.3f} ≤ IV(ATM)={iv_atm:.3f}"
    )
    assert iv_atm > iv_otm_call - 0.01, (
        f"Skew not monotone: IV(ATM)={iv_atm:.3f}, IV(110c)={iv_otm_call:.3f}"
    )


def test_heston_positive_rho_positive_skew():
    """With positive ρ, calls are more expensive than equidistant puts (positive skew)."""
    S, T, r, v0 = 100.0, 0.5, 0.05, 0.04
    kwargs = dict(T=T, r=r, v0=v0, kappa=2.0, theta=v0, xi=0.4, rho=+0.6)

    iv_otm_call = heston_implied_vol(S, K=110.0, **kwargs, option_type="call")
    iv_otm_put  = heston_implied_vol(S, K=90.0,  **kwargs, option_type="put")
    # Positive rho → right skew (calls more expensive than puts)
    assert iv_otm_call > iv_otm_put - 0.01


def test_feller_condition_warning():
    """Warn when 2κθ ≤ ξ² (Feller condition violated)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = HestonConfig(kappa=0.5, theta=0.01, xi=1.5)  # 2*0.5*0.01 = 0.01 << 2.25
        assert len(w) == 1
        assert "Feller" in str(w[0].message)


def test_feller_condition_satisfied_no_warning():
    """No warning when Feller condition holds."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = HestonConfig(kappa=3.0, theta=0.04, xi=0.3)  # 2*3*0.04=0.24 >> 0.09
        assert len(w) == 0


def test_feller_ratio():
    cfg = HestonConfig(kappa=2.0, theta=0.04, xi=0.30)
    expected = 2 * 2.0 * 0.04 / (0.30 ** 2)
    assert abs(cfg.feller_ratio - expected) < 1e-10


def test_heston_scenario_mean_reversion():
    """
    E[v_T] should converge to θ as T → ∞ (mean reversion property of CIR).
    For finite T, E[v_T] = θ + (v₀ - θ)·e^{-κT}.
    """
    cfg = HestonConfig(
        S0=100.0, v0=0.01, kappa=5.0, theta=0.04,
        xi=0.1, rho=-0.5, T=2.0, n_steps=504, n_paths=5000, seed=7
    )
    scenario = HestonScenario(cfg)
    _, v_paths = scenario.generate()

    mean_v_T = float(v_paths[-1].mean())
    expected  = cfg.theta + (cfg.v0 - cfg.theta) * np.exp(-cfg.kappa * cfg.T)
    # Euler-Maruyama has finite discretisation error; allow 10% tolerance
    assert abs(mean_v_T - expected) / expected < 0.10, (
        f"E[v_T]={mean_v_T:.4f} ≠ expected {expected:.4f}"
    )


def test_heston_scenario_shape():
    cfg = HestonConfig(n_steps=50, n_paths=10)
    S, v = HestonScenario(cfg).generate()
    assert S.shape == (51, 10)
    assert v.shape == (51, 10)
    assert np.all(S > 0)
    assert np.all(v >= 0)


def test_heston_invalid_inputs():
    with pytest.raises(ValueError, match="T > 0"):
        heston_price(100, 100, 0.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.7)
    with pytest.raises(ValueError, match="v0"):
        heston_price(100, 100, 0.5, 0.05, -0.01, 2.0, 0.04, 0.3, -0.7)
    with pytest.raises(ValueError, match="xi"):
        heston_price(100, 100, 0.5, 0.05, 0.04, 2.0, 0.04, -0.1, -0.7)
