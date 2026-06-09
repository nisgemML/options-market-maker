"""
Delta hedging strategies.

Three hedge triggers:
  1. Periodic  — rehedge every N minutes/bars
  2. Band      — rehedge when |net delta| > threshold
  3. Gamma     — dynamic band: threshold = gamma_multiple * sqrt(dt) * S
                 (Zakamouline & Koekebakker 2009 approximation)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple

import numpy as np

from .portfolio import HedgePortfolio


class HedgeFrequency(Enum):
    PERIODIC = auto()   # every N steps
    BAND     = auto()   # delta band
    GAMMA    = auto()   # gamma-adjusted band


@dataclass
class HedgeRecord:
    """One hedge trade."""
    step:           int
    S:              float
    delta_before:   float
    delta_after:    float
    shares_traded:  float
    cost:           float       # transaction cost
    trigger:        str


class DeltaHedger:
    """
    Pure-functional delta hedger: each call returns a new portfolio + record.
    No mutable state — safe to run in parallel across simulations.
    """

    def __init__(
        self,
        hedge_freq: HedgeFrequency = HedgeFrequency.BAND,
        band_threshold: float = 0.05,       # for BAND: rehedge if |delta| > this
        gamma_multiple: float = 1.0,        # for GAMMA: band = multiple * sqrt(dt)*S*gamma
        periodic_n: int = 10,               # for PERIODIC: rehedge every N steps
        transaction_cost_bps: float = 1.0,  # round-trip cost in bps of notional
    ) -> None:
        self.hedge_freq = hedge_freq
        self.band_threshold = band_threshold
        self.gamma_multiple = gamma_multiple
        self.periodic_n = periodic_n
        self.tc_bps = transaction_cost_bps

    def _should_hedge(
        self,
        net_delta: float,
        step: int,
        S: float,
        dt: float,
        gamma: float,
    ) -> bool:
        if self.hedge_freq == HedgeFrequency.PERIODIC:
            return step % self.periodic_n == 0

        if self.hedge_freq == HedgeFrequency.BAND:
            return abs(net_delta) > self.band_threshold

        # GAMMA: dynamic band proportional to gamma * sqrt(dt) * S
        band = self.gamma_multiple * abs(gamma) * np.sqrt(dt) * S
        band = max(band, 0.01)  # floor to avoid zero band
        return abs(net_delta) > band

    def step(
        self,
        portfolio: HedgePortfolio,
        S: float,
        step: int,
        dt: float,
    ) -> tuple[HedgePortfolio, HedgeRecord | None]:
        """
        Evaluate hedge trigger and optionally rehedge.

        Returns:
            (updated_portfolio, HedgeRecord or None if no trade)
        """
        greeks = portfolio.net_greeks(S)
        net_delta = portfolio.net_delta(S)

        if not self._should_hedge(net_delta, step, S, dt, greeks.gamma):
            return portfolio, None

        # Target: zero net delta
        shares_to_trade = -net_delta  # buy if delta negative, sell if positive

        # Transaction cost
        notional = abs(shares_to_trade) * S
        cost = notional * self.tc_bps / 10_000

        # Update portfolio
        new_hedge = portfolio.delta_hedge + shares_to_trade
        new_cash  = portfolio.cash - shares_to_trade * S - cost

        new_portfolio = HedgePortfolio(
            positions=portfolio.positions,
            delta_hedge=new_hedge,
            cash=new_cash,
        )

        record = HedgeRecord(
            step=step,
            S=S,
            delta_before=net_delta,
            delta_after=new_portfolio.net_delta(S),
            shares_traded=shares_to_trade,
            cost=cost,
            trigger=self.hedge_freq.name,
        )

        return new_portfolio, record
