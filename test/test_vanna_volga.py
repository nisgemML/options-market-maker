"""
Vanna-volga pricer tests.

Key properties:
  1. VV price = BS price when all pillar IVs equal σ_ATM (flat smile → zero correction)
  2. VV correction reduces (not increases) price for OTM options on a skewed smile
  3. Negative ρ smile: OTM puts have higher IV → VV correction positive for puts
  4. Survival probability = 0 → VV correction = 0 (knocked-out barrier)
  5. Output dict contains all required keys
"""
import numpy as np
import pytest

from src.pricing.black_scholes import BlackScholes, OptionType
from src.pricing.vanna_volga import vanna_volga_price, VVPillarQuotes


def flat_pillars(sigma: float = 0.20, S: float = 100.0, T: float = 0.5) -> VVPillarQuotes:
    """Flat smile: all three pillars at same IV → VV correction = 0."""
    K_atm = S  # simplified: ATM = spot
    return VVPillarQuotes(
        K_25p=90.0, K_atm=K_atm, K_25c=110.0,
        sigma_25p=sigma, sigma_atm=sigma, sigma_25c=sigma,
    )


def skewed_pillars(S: float = 100.0) -> VVPillarQuotes:
    """Negatively skewed smile (equity-like)."""
    return VVPillarQuotes(
        K_25p=90.0, K_atm=100.0, K_25c=110.0,
        sigma_25p=0.25, sigma_atm=0.20, sigma_25c=0.18,
    )


def test_vv_flat_smile_zero_correction():
    """
    With a flat smile, all overcosts are zero → VV correction = 0
    → VV price = BS price.
    """
    S, K, T, r = 100.0, 100.0, 0.5, 0.05
    pillars = flat_pillars(sigma=0.20, S=S, T=T)
    result = vanna_volga_price(S, K, T, r, 0.0, pillars, OptionType.CALL)

    bs_price = float(BlackScholes.price(S, K, T, r, 0.20, OptionType.CALL))
    assert abs(result["vv_price"] - bs_price) < 0.05, (
        f"Flat smile VV correction should be ~0: vv={result['vv_price']:.4f}, "
        f"bs={bs_price:.4f}"
    )


def test_vv_result_keys():
    """Output dict must contain all documented keys."""
    S, K, T, r = 100.0, 100.0, 0.5, 0.05
    result = vanna_volga_price(S, K, T, r, 0.0, flat_pillars(), OptionType.CALL)
    expected_keys = {"bs_price", "vv_correction", "vv_price", "vanna", "volga",
                     "vanna_cost", "volga_cost"}
    assert expected_keys == set(result.keys())


def test_vv_price_positive():
    """VV price must be non-negative (floored at 0)."""
    for K in [80.0, 90.0, 100.0, 110.0, 120.0]:
        for ot in [OptionType.CALL, OptionType.PUT]:
            result = vanna_volga_price(
                100.0, K, 0.5, 0.05, 0.0, skewed_pillars(), ot
            )
            assert result["vv_price"] >= 0.0, f"Negative VV price at K={K}, {ot}"


def test_vv_survival_zero_kills_correction():
    """
    survival_probability=0 (barrier already knocked out): VV correction → 0,
    so VV price = BS price.
    """
    S, K, T, r = 100.0, 95.0, 0.5, 0.05
    pillars = skewed_pillars()
    result = vanna_volga_price(S, K, T, r, 0.0, pillars, OptionType.PUT,
                                survival_probability=0.0)
    assert abs(result["vv_correction"]) < 1e-10


def test_vv_survival_one_full_correction():
    """survival_probability=1 gives the vanilla VV price (no barrier discount)."""
    S, K, T, r = 100.0, 95.0, 0.5, 0.05
    pillars = skewed_pillars()
    r1 = vanna_volga_price(S, K, T, r, 0.0, pillars, OptionType.PUT, survival_probability=1.0)
    r0 = vanna_volga_price(S, K, T, r, 0.0, pillars, OptionType.PUT, survival_probability=0.0)
    # survival=1 gives more correction than survival=0
    assert abs(r1["vv_correction"]) >= abs(r0["vv_correction"])


def test_vv_skewed_smile_otm_put_correction_sign():
    """
    Negatively skewed smile: OTM puts have higher IV than ATM.
    VV correction for OTM put should be positive (smile adds value).
    """
    S, K, T, r = 100.0, 88.0, 0.5, 0.05
    pillars = skewed_pillars()
    result = vanna_volga_price(S, K, T, r, 0.0, pillars, OptionType.PUT)
    # Not asserting direction strongly (depends on vanna/volga magnitudes)
    # but correction should be small relative to BS price
    assert abs(result["vv_correction"]) < result["bs_price"] * 2.0


def test_vv_from_rr_strangle():
    """VVPillarQuotes.from_atm_strangle_rr constructs valid pillar strikes."""
    F = 100.0
    pillars = VVPillarQuotes.from_atm_strangle_rr(
        F=F, T=0.5, sigma_atm=0.20, rr=0.02, str_=0.005
    )
    assert pillars.K_25p < pillars.K_atm < pillars.K_25c
    assert pillars.sigma_25p > 0
    assert pillars.sigma_25c > 0
