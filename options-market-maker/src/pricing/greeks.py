"""
Greeks container and batch computation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import BlackScholes, OptionType


@dataclass
class Greeks:
    """All first and second-order Greeks for one option."""
    delta:  float
    gamma:  float
    vega:   float
    theta:  float
    rho:    float
    vanna:  float
    volga:  float

    def dollar_delta(self, notional: float) -> float:
        """Delta exposure in dollar terms."""
        return self.delta * notional

    def dollar_gamma(self, notional: float, dS: float = 1.0) -> float:
        """Dollar gamma: P&L from a 1-dollar move in spot."""
        return 0.5 * self.gamma * dS**2 * notional

    def __repr__(self) -> str:
        return (
            f"Greeks(Δ={self.delta:+.4f}, Γ={self.gamma:.4f}, "
            f"ν={self.vega:.4f}, Θ={self.theta:+.4f}/day, "
            f"ρ={self.rho:.4f}, vanna={self.vanna:.4f}, volga={self.volga:.4f})"
        )


def compute_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
) -> Greeks:
    """Compute all Greeks for a single option."""
    bs = BlackScholes
    return Greeks(
        delta=float(bs.delta(S, K, T, r, sigma, option_type, q)),
        gamma=float(bs.gamma(S, K, T, r, sigma, q)),
        vega=float(bs.vega(S, K, T, r, sigma, q)),
        theta=float(bs.theta(S, K, T, r, sigma, option_type, q)),
        rho=float(bs.rho(S, K, T, r, sigma, option_type, q)),
        vanna=float(bs.vanna(S, K, T, r, sigma, q)),
        volga=float(bs.volga(S, K, T, r, sigma, q)),
    )


def portfolio_greeks(positions: list[tuple[Greeks, float]]) -> Greeks:
    """
    Aggregate Greeks across a portfolio.

    Args:
        positions: list of (Greeks, quantity) — quantity is signed
                   (positive = long, negative = short).
    """
    delta = gamma = vega = theta = rho = vanna = volga = 0.0
    for g, qty in positions:
        delta += g.delta * qty
        gamma += g.gamma * qty
        vega  += g.vega  * qty
        theta += g.theta * qty
        rho   += g.rho   * qty
        vanna += g.vanna * qty
        volga += g.volga * qty
    return Greeks(delta=delta, gamma=gamma, vega=vega,
                  theta=theta, rho=rho, vanna=vanna, volga=volga)


def finite_diff_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float = 0.0,
    dS: float = 0.01,
    dv: float = 0.001,
) -> dict[str, float]:
    """
    Finite-difference Greeks for validation of analytical formulas.
    Used in tests to cross-check BlackScholes analytical Greeks.
    """
    bs = BlackScholes
    p  = lambda s, v: bs.price(s, K, T, r, v, option_type, q)

    price_base = p(S, sigma)
    delta_fd   = (p(S + dS, sigma) - p(S - dS, sigma)) / (2 * dS)
    gamma_fd   = (p(S + dS, sigma) - 2 * price_base + p(S - dS, sigma)) / dS**2
    vega_fd    = (p(S, sigma + dv) - p(S, sigma - dv)) / (2 * dv)
    vanna_fd   = (
        p(S + dS, sigma + dv) - p(S + dS, sigma - dv)
        - p(S - dS, sigma + dv) + p(S - dS, sigma - dv)
    ) / (4 * dS * dv)
    volga_fd   = (p(S, sigma + dv) - 2 * price_base + p(S, sigma - dv)) / dv**2

    return {
        "delta": float(delta_fd),
        "gamma": float(gamma_fd),
        "vega":  float(vega_fd),
        "vanna": float(vanna_fd),
        "volga": float(volga_fd),
    }
