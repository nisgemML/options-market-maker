"""
Options market maker — quoting engine.

Quoting logic:
  1. Price option using BS + current IV from vol surface
  2. Apply inventory skew to shift mid
  3. Set spread based on: base_spread + gamma adjustment + vega adjustment
  4. Enforce min tick and max quote size
  5. Pull quotes if risk limits breached

Risk limits:
  - Max net delta
  - Max net gamma
  - Max net vega
  - Max position per strike
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.pricing import BlackScholes, OptionType, compute_greeks
from src.pricing.surface import VolSurface
from .skew import InventorySkew
from src.hedging.portfolio import HedgePortfolio


class Quote(NamedTuple):
    """A two-sided options quote."""
    strike:      float
    expiry:      float
    option_type: OptionType
    bid:         float
    ask:         float
    bid_size:    int
    ask_size:    int
    fair_value:  float
    sigma:       float      # IV used to price
    pulled:      bool = False  # True if quotes pulled due to risk limits


@dataclass
class MMParams:
    """Market maker configuration."""
    base_spread_vol:      float = 0.005   # base spread in vol points (0.5%)
    max_delta:            float = 50.0    # max net delta exposure
    max_gamma:            float = 5.0     # max net gamma
    max_vega:             float = 500.0   # max net vega
    max_position_per_strike: int = 100    # max contracts per strike
    min_spread:           float = 0.01    # min $ spread
    quote_size:           int   = 10      # default quote size (contracts)
    tick_size:            float = 0.01    # min price increment
    inventory_alpha:      float = 0.01    # skew sensitivity
    gamma_spread_mult:    float = 0.5     # gamma spread multiplier


class MarketMaker:
    """
    Options market maker.

    Generates two-sided quotes for a given option, conditioned on:
      - Current spot price S
      - Vol surface (for fair value IV)
      - Current portfolio Greeks (for inventory skew and risk checks)
    """

    def __init__(self, params: MMParams | None = None) -> None:
        self.params = params or MMParams()
        self.skew = InventorySkew(
            alpha=self.params.inventory_alpha,
            gamma_spread_mult=self.params.gamma_spread_mult,
        )

    def _round_to_tick(self, price: float) -> float:
        tick = self.params.tick_size
        return round(round(price / tick) * tick, 10)

    def _risk_limit_breached(
        self,
        portfolio: HedgePortfolio,
        S: float,
    ) -> str | None:
        """Returns reason string if any risk limit is breached, else None."""
        g = portfolio.net_greeks(S)
        p = self.params
        if abs(portfolio.net_delta(S)) > p.max_delta:
            return f"delta {portfolio.net_delta(S):.1f} > {p.max_delta}"
        if abs(g.gamma) > p.max_gamma:
            return f"gamma {g.gamma:.2f} > {p.max_gamma}"
        if abs(g.vega) > p.max_vega:
            return f"vega {g.vega:.1f} > {p.max_vega}"
        return None

    def quote(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        portfolio: HedgePortfolio,
        vol_surface: VolSurface | None = None,
        sigma_override: float | None = None,
        q: float = 0.0,
    ) -> Quote:
        """
        Generate a two-sided quote.

        Vol priority: sigma_override > vol_surface interpolation > flat 20% fallback.
        """
        p = self.params

        # 1. Get fair vol
        if sigma_override is not None:
            sigma = sigma_override
        elif vol_surface is not None:
            sigma = vol_surface.interpolate(K, T)
        else:
            sigma = 0.20  # fallback flat vol

        # 2. Fair value
        fair_value = float(BlackScholes.price(S, K, T, r, sigma, option_type, q))

        # 3. Greeks at this option
        greeks = compute_greeks(S, K, T, r, sigma, option_type, q)

        # 4. Portfolio Greeks for skew
        net_delta = portfolio.net_delta(S)
        port_greeks = portfolio.net_greeks(S)

        # 5. Base spread: vol-point spread converted to $ spread via vega
        # d(price)/d(sigma) = vega, so $ spread = vol_spread * vega
        vol_spread = p.base_spread_vol
        base_spread = max(vol_spread * abs(greeks.vega), p.min_spread)

        # 6. Inventory-skewed quotes
        bid, ask = self.skew.adjusted_quotes(
            fair_value=fair_value,
            base_spread=base_spread,
            net_delta=net_delta,
            gamma=port_greeks.gamma,
            S=S,
        )

        # 7. Round to tick
        bid = self._round_to_tick(bid)
        ask = self._round_to_tick(ask)

        # Ensure bid < ask after rounding
        if bid >= ask:
            ask = bid + p.tick_size

        # 8. Risk limit check — pull quotes if breached
        breach = self._risk_limit_breached(portfolio, S)
        pulled = breach is not None

        return Quote(
            strike=K,
            expiry=T,
            option_type=option_type,
            bid=bid if not pulled else 0.0,
            ask=ask if not pulled else float("inf"),
            bid_size=p.quote_size if not pulled else 0,
            ask_size=p.quote_size if not pulled else 0,
            fair_value=fair_value,
            sigma=sigma,
            pulled=pulled,
        )

    def quote_strip(
        self,
        S: float,
        strikes: list[float],
        T: float,
        r: float,
        option_type: OptionType,
        portfolio: HedgePortfolio,
        vol_surface: VolSurface | None = None,
    ) -> list[Quote]:
        """Quote a strip of strikes for one expiry."""
        return [
            self.quote(S, K, T, r, option_type, portfolio, vol_surface)
            for K in strikes
        ]
