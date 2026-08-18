"""
Vanna-Volga (VV) pricing approximation for exotic options.

The vanna-volga method (Castagna & Mercurio 2007) adjusts a BS price for
smile risk using the market prices of three liquid vanillas — typically the
Δ=25 put, Δ=50 (ATM), and Δ=25 call — from which the market cost of
vanna and volga are inferred.

Core formula (first-order approximation):
    Price_VV ≈ Price_BS(σ_ATM)
                + vanna_exotic · (market cost per unit vanna)
                + volga_exotic · (market cost per unit volga)

Where market costs are extracted by solving:
    P_25Δ_mkt = P_25Δ_BS(σ_ATM) + x₁·vanna_25Δ + x₂·volga_25Δ
    P_ATM_mkt  = P_ATM_BS(σ_ATM) + x₁·vanna_ATM  + x₂·volga_ATM
    P_25Δc_mkt = P_25Δc_BS(σ_ATM) + x₁·vanna_25Δc + x₂·volga_25Δc

This gives the per-unit costs of vanna and volga from the market smile.

Primary use case: FX barrier options, digital options, one-touch options —
where BS misprices smile risk systematically.

References:
    Castagna, A. & Mercurio, F. (2007). The vanna-volga method for implied
    volatilities. Risk, 106-111.
    Wystup, U. (2006). FX Options and Structured Products. Wiley.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .black_scholes import BlackScholes, OptionType


class VVPillarQuotes(NamedTuple):
    """
    Three liquid vanilla quotes that anchor the vanna-volga smile.

    Standard FX convention: 25-delta put, ATM (50-delta), 25-delta call.
    Strikes can be computed from delta or supplied directly.
    """
    K_25p: float     # 25-delta put strike
    K_atm: float     # ATM (50-delta) strike ≈ F·exp(½σ²T)
    K_25c: float     # 25-delta call strike
    sigma_25p: float  # market IV at K_25p
    sigma_atm: float  # ATM IV
    sigma_25c: float  # market IV at K_25c

    @classmethod
    def from_atm_strangle_rr(
        cls,
        F: float,
        T: float,
        sigma_atm: float,
        rr: float,   # 25-delta risk reversal = σ_25c - σ_25p
        str_: float, # 25-delta strangle = (σ_25c + σ_25p)/2 - σ_atm
    ) -> "VVPillarQuotes":
        """
        Build from standard FX quote conventions:
          Risk reversal: RR = σ_25c - σ_25p
          Strangle:      STR = (σ_25c + σ_25p)/2 - σ_ATM
        """
        sigma_25c = sigma_atm + str_ + rr / 2.0
        sigma_25p = sigma_atm + str_ - rr / 2.0

        # ATM-DNS strike: K_atm = F·exp(½σ²T)
        K_atm = F * np.exp(0.5 * sigma_atm ** 2 * T)

        # 25-delta strikes (Garman-Kohlhagen, zero rates for simplicity)
        from scipy.stats import norm
        sqrtT = np.sqrt(T)
        K_25c = F * np.exp(-norm.ppf(0.25) * sigma_25c * sqrtT + 0.5 * sigma_25c**2 * T)
        K_25p = F * np.exp( norm.ppf(0.25) * sigma_25p * sqrtT + 0.5 * sigma_25p**2 * T)

        return cls(
            K_25p=K_25p, K_atm=K_atm, K_25c=K_25c,
            sigma_25p=sigma_25p, sigma_atm=sigma_atm, sigma_25c=sigma_25c,
        )


def vanna_volga_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    pillars: VVPillarQuotes,
    option_type: OptionType,
    exotic_bs_price: float | None = None,
    survival_probability: float = 1.0,
) -> dict[str, float]:
    """
    Vanna-volga adjusted price for a vanilla or barrier option.

    Args:
        S:                   spot price
        K:                   option strike
        T:                   time to expiry
        r:                   domestic risk-free rate
        q:                   foreign rate / dividend yield
        pillars:             three-pillar market quotes
        option_type:         CALL or PUT
        exotic_bs_price:     BS price of exotic at σ_ATM (if None, computes vanilla)
        survival_probability: P(barrier not hit) ∈ [0,1]. For vanillas = 1.
                              For down-and-out call, approximate analytically or
                              supply from MC. Controls the VV correction weight.

    Returns:
        dict with keys: 'bs_price', 'vv_correction', 'vv_price',
                        'vanna', 'volga', 'vanna_cost', 'volga_cost'

    Notes:
        The survival_probability adjustment (Castagna & Mercurio 2007, eq. 14)
        accounts for the fact that a barrier option stops accumulating smile
        P&L if knocked out. Setting p=1 gives the full vanilla VV correction.
    """
    bs = BlackScholes
    sigma_atm = pillars.sigma_atm

    # BS price of target option at ATM vol
    if exotic_bs_price is None:
        bs_price = float(bs.price(S, K, T, r, sigma_atm, option_type, q))
    else:
        bs_price = exotic_bs_price

    # Greeks of target option at ATM vol
    vanna_x = float(bs.vanna(S, K, T, r, sigma_atm, q))
    volga_x = float(bs.volga(S, K, T, r, sigma_atm, q))

    # Greeks of the three pillars at ATM vol (σ_ATM for pillar greeks — VV approximation)
    # Market overcost = pillar_market_price - pillar_BS_price(σ_ATM)
    def pillar_overcost(K_i: float, sigma_i: float, ot: OptionType) -> float:
        p_mkt = float(bs.price(S, K_i, T, r, sigma_i, ot, q))
        p_bs  = float(bs.price(S, K_i, T, r, sigma_atm, ot, q))
        return p_mkt - p_bs

    overcost_25p = pillar_overcost(pillars.K_25p, pillars.sigma_25p, OptionType.PUT)
    overcost_atm = pillar_overcost(pillars.K_atm, pillars.sigma_atm, OptionType.CALL)  # = 0 by def
    overcost_25c = pillar_overcost(pillars.K_25c, pillars.sigma_25c, OptionType.CALL)

    # Vanna and volga of pillars at ATM vol
    vanna_25p = float(bs.vanna(S, pillars.K_25p, T, r, sigma_atm, q))
    vanna_atm = float(bs.vanna(S, pillars.K_atm, T, r, sigma_atm, q))
    vanna_25c = float(bs.vanna(S, pillars.K_25c, T, r, sigma_atm, q))

    volga_25p = float(bs.volga(S, pillars.K_25p, T, r, sigma_atm, q))
    volga_atm = float(bs.volga(S, pillars.K_atm, T, r, sigma_atm, q))
    volga_25c = float(bs.volga(S, pillars.K_25c, T, r, sigma_atm, q))

    # Solve 3×3 system: [vanna_i, volga_i] · [x_vanna, x_volga] = overcost_i
    # Using ATM as the third equation (which constrains the system)
    # Full Castagna-Mercurio system:
    A = np.array([
        [vanna_25p, volga_25p],
        [vanna_atm, volga_atm],
        [vanna_25c, volga_25c],
    ])
    b_vec = np.array([overcost_25p, overcost_atm, overcost_25c])

    # Least-squares solution (overdetermined 3×2 system)
    x, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
    cost_per_vanna, cost_per_volga = x

    # VV correction weighted by survival probability (barrier adjustment)
    vv_correction = survival_probability * (
        vanna_x * cost_per_vanna + volga_x * cost_per_volga
    )
    vv_price = bs_price + vv_correction

    return {
        "bs_price":        bs_price,
        "vv_correction":   vv_correction,
        "vv_price":        max(vv_price, 0.0),
        "vanna":           vanna_x,
        "volga":           volga_x,
        "vanna_cost":      cost_per_vanna,
        "volga_cost":      cost_per_volga,
    }
