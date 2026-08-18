"""
Option position and portfolio abstractions.

Design: pure functional / immutable.
  - OptionPosition: frozen dataclass — state cannot be mutated after construction.
  - HedgePortfolio: frozen dataclass with tuple of positions (not list).
    All "mutations" return new instances, making the state transition
    explicit and enabling safe parallelism across backtest paths.

This mirrors Jane Street's approach to state in OCaml: all state is threaded
explicitly as function arguments rather than hidden in mutable objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.pricing import BlackScholes, OptionType, Greeks, compute_greeks

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class OptionPosition:
    """
    A single option leg.  Frozen — immutable after construction.

    Quantity convention:
      +1 = long (MM bought from client)
      -1 = short (MM sold to client)
    """
    strike:      float
    expiry:      float          # years to expiry
    option_type: OptionType
    quantity:    float          # signed
    entry_price: float          # premium paid/received per unit
    r:           float
    sigma:       float          # IV at entry
    q:           float = 0.0

    def current_price(self, S: float, sigma: float | None = None) -> float:
        sig = sigma if sigma is not None else self.sigma
        return float(BlackScholes.price(S, self.strike, self.expiry, self.r, sig,
                                        self.option_type, self.q))

    def greeks(self, S: float, sigma: float | None = None) -> Greeks:
        sig = sigma if sigma is not None else self.sigma
        return compute_greeks(S, self.strike, self.expiry, self.r, sig,
                              self.option_type, self.q)

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


@dataclass(frozen=True)
class HedgePortfolio:
    """
    Immutable portfolio of option positions plus a delta hedge in the underlying.

    All state transitions (add_position, roll_time, hedge) return new instances.
    No mutable state — safe to use across parallel simulation paths.

    Tracks:
      - positions: tuple of OptionPosition (immutable sequence)
      - delta_hedge: shares/futures held for delta neutrality
      - cash: running cash account (debited for longs, credited for shorts
              and hedge trades)
    """
    positions:    tuple[OptionPosition, ...] = field(default_factory=tuple)
    delta_hedge:  float = 0.0
    cash:         float = 0.0

    def __post_init__(self) -> None:
        # Ensure positions is always a tuple even if a list was passed in
        if isinstance(self.positions, list):
            object.__setattr__(self, "positions", tuple(self.positions))

    def add_position(self, pos: OptionPosition) -> "HedgePortfolio":
        """
        Return a new portfolio with pos netted into existing positions.

        Netting: if a position with the same (strike, expiry, option_type, r, sigma, q)
        already exists, quantities are summed rather than creating a new leg.
        This mirrors real MM practice and keeps the position count bounded
        (O(strikes × types) rather than O(fills)).

        Does NOT mutate self.
        """
        new_cash = self.cash - pos.quantity * pos.entry_price

        # Find an existing position with matching contract terms
        key = (pos.strike, pos.option_type, pos.r, pos.sigma, pos.q)
        matched_idx = None
        for i, existing in enumerate(self.positions):
            if (existing.strike == pos.strike and
                    existing.option_type == pos.option_type and
                    abs(existing.expiry - pos.expiry) < 1e-6 and
                    existing.r == pos.r and existing.sigma == pos.sigma):
                matched_idx = i
                break

        if matched_idx is None:
            new_positions = self.positions + (pos,)
        else:
            old_pos = self.positions[matched_idx]
            new_qty = old_pos.quantity + pos.quantity
            if abs(new_qty) < 1e-10:
                # Fully flat — remove the leg
                new_positions = self.positions[:matched_idx] + self.positions[matched_idx+1:]
            else:
                # Update quantity; keep entry price as volume-weighted average
                total_cost = old_pos.quantity * old_pos.entry_price + pos.quantity * pos.entry_price
                avg_entry = total_cost / new_qty if abs(new_qty) > 1e-10 else pos.entry_price
                netted = OptionPosition(
                    strike=old_pos.strike,
                    expiry=old_pos.expiry,
                    option_type=old_pos.option_type,
                    quantity=new_qty,
                    entry_price=abs(avg_entry),
                    r=old_pos.r,
                    sigma=old_pos.sigma,
                    q=old_pos.q,
                )
                new_positions = self.positions[:matched_idx] + (netted,) + self.positions[matched_idx+1:]

        return HedgePortfolio(
            positions=new_positions,
            delta_hedge=self.delta_hedge,
            cash=new_cash,
        )

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
        """Advance time by dt, decaying all option expiries (drop expired legs)."""
        new_positions = tuple(
            pos.roll_time(dt) for pos in self.positions
            if pos.expiry - dt > 0
        )
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
