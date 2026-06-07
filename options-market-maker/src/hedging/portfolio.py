"""
Option position and portfolio abstractions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pricing import BlackScholes, OptionType, Greeks, compute_greeks


@dataclass
class OptionPosition:
    """A single option leg in the portfolio."""
    strike:      float
    expiry:      float          # years to expiry
    option_type: OptionType
    quantity:    float          # signed: positive=long, negative=short
    entry_price: float          # premium paid/received per unit
    r:           float
    sigma:       float          # IV at entry
    q:           float = 0.0

    def current_price(self, S: float, sigma: float | None = None) -> float:
        sig = sigma if sigma is not None else self.sigma
        return float(BlackScholes.price(S, self.strike, self.expiry, self.r, sig, self.option_type, self.q))

    def greeks(self, S: float, sigma: float | None = None) -> Greeks:
        sig = sigma if sigma is not None else self.sigma
        return compute_greeks(S, self.strike, self.expiry, self.r, sig, self.option_type, self.q)

    def pnl(self, S: float, sigma: float | None = None) -> float:
        """Mark-to-market P&L vs entry price."""
        return self.quantity * (self.current_price(S, sigma) - self.entry_price)

    def roll_time(self, dt: float) -> "OptionPosition":
        """Return a new position with time decayed by dt years."""
        return OptionPosition(
            strike=self.strike,
            expiry=max(self.expiry - dt, 0.0),
            option_type=self.option_type,
            quantity=self.quantity,
            entry_price=self.entry_price,
            r=self.r,
            sigma=self.sigma,
            q=self.q,
        )


@dataclass
class HedgePortfolio:
    """
    Portfolio of option positions plus a delta hedge in the underlying.

    Tracks:
      - option legs
      - delta hedge quantity (shares/futures)
      - cash account (receives/pays option premiums and hedge costs)
    """
    positions:    list[OptionPosition] = field(default_factory=list)
    delta_hedge:  float = 0.0     # shares held for hedge
    cash:         float = 0.0     # cash/PnL account

    def add_position(self, pos: OptionPosition) -> None:
        """Add an option leg, debit/credit premium to cash."""
        self.positions.append(pos)
        self.cash -= pos.quantity * pos.entry_price  # pay for longs, receive for shorts

    def net_greeks(self, S: float) -> Greeks:
        """Aggregate Greeks across all option legs (not including delta hedge)."""
        from src.pricing.greeks import portfolio_greeks
        legs = [(pos.greeks(S), pos.quantity) for pos in self.positions]
        return portfolio_greeks(legs)

    def net_delta(self, S: float) -> float:
        """Total delta including the hedge position."""
        option_delta = sum(
            pos.quantity * pos.greeks(S).delta for pos in self.positions
        )
        return option_delta + self.delta_hedge

    def mtm_value(self, S: float) -> float:
        """Mark-to-market: options value + hedge value + cash."""
        options_value = sum(
            pos.quantity * pos.current_price(S) for pos in self.positions
        )
        return options_value + self.delta_hedge * S + self.cash

    def option_pnl(self, S: float) -> float:
        return sum(pos.pnl(S) for pos in self.positions)

    def roll_time(self, dt: float) -> "HedgePortfolio":
        """Advance time by dt, decaying all option expiries."""
        new_positions = [pos.roll_time(dt) for pos in self.positions
                         if pos.expiry - dt > 0]
        return HedgePortfolio(
            positions=new_positions,
            delta_hedge=self.delta_hedge,
            cash=self.cash,
        )

    def __repr__(self) -> str:
        return (
            f"HedgePortfolio(legs={len(self.positions)}, "
            f"hedge={self.delta_hedge:.2f}, cash={self.cash:.2f})"
        )
